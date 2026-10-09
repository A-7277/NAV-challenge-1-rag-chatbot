"""Idempotent PDF download, Docling conversion, chunking and indexing."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from langdetect import DetectorFactory, detect
from pypdf import PdfReader

from rag_app.config import Settings
from rag_app.embeddings import LocalEmbeddingModel
from rag_app.models import BoundingBox, Chunk
from rag_app.text import normalize_text, stable_chunk_id
from rag_app.vector_store import QdrantStore, chunk_from_payload

LOGGER = logging.getLogger(__name__)
DetectorFactory.seed = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_document(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "TraceablePdfRag/0.1 (educational document retrieval demo)"}
    with requests.get(url, headers=headers, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            raise ValueError(f"Expected PDF from {url}, received {content_type}")
        with partial.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    if partial.read_bytes()[:5] != b"%PDF-":
        partial.unlink(missing_ok=True)
        raise ValueError(f"Downloaded content is not a PDF: {url}")
    partial.replace(destination)


def inspect_pdf(path: Path, *, min_pages: int = 200) -> dict[str, Any]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    if page_count < min_pages:
        raise ValueError(f"{path.name} has {page_count} pages; expected at least {min_pages}")
    sampled = sorted(
        set(range(min(page_count, 5)))
        | set(range(max(0, page_count - 5), page_count))
    )
    native = sum(
        1
        for index in sampled
        if len((reader.pages[index].extract_text() or "").strip()) >= 50
    )
    ratio = native / len(sampled) if sampled else 0.0
    return {
        "page_count": page_count,
        "sampled_pages": len(sampled),
        "native_text_sample_ratio": round(ratio, 3),
        "estimated_ocr_pages": round(page_count * (1 - ratio)),
    }


def _docling_converter(settings: Settings):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        HeadingHierarchyOptions,
        PdfPipelineOptions,
        TableFormerMode,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    docling_artifacts = settings.model_cache_dir / "docling"
    options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_page_images=False,
        generate_parsed_pages=True,
        ocr_options=TesseractCliOcrOptions(lang="eng"),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=8),
    )
    # Setting an empty artifacts_path forces offline lookup and prevents Docling
    # from downloading its first-run models. Only opt into it for a prepared bundle.
    if docling_artifacts.exists() and any(docling_artifacts.iterdir()):
        options.artifacts_path = docling_artifacts
    options.table_structure_options.mode = TableFormerMode.FAST
    options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _extract_provenance(doc_items: list[Any]) -> tuple[list[int], list[BoundingBox]]:
    pages: set[int] = set()
    boxes: list[BoundingBox] = []
    seen: set[tuple] = set()
    for item in doc_items:
        for prov in getattr(item, "prov", []) or []:
            page = int(prov.page_no)
            pages.add(page)
            bbox = prov.bbox
            key = (page, float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
            if key in seen:
                continue
            seen.add(key)
            origin = getattr(getattr(bbox, "coord_origin", None), "value", "BOTTOMLEFT")
            boxes.append(
                BoundingBox(
                    page=page,
                    left=round(float(bbox.l), 2),
                    top=round(float(bbox.t), 2),
                    right=round(float(bbox.r), 2),
                    bottom=round(float(bbox.b), 2),
                    origin=str(origin),
                )
            )
    return sorted(pages), boxes


def convert_and_chunk(
    pdf_path: Path,
    *,
    title: str,
    document_sha256: str,
    settings: Settings,
    max_tokens: int = 700,
    overlap_tokens: int = 105,
    page_range: tuple[int, int] | None = None,
) -> list[Chunk]:
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    converter = _docling_converter(settings)
    # Newer Docling versions reject page_range=None via pydantic; only pass it
    # when an explicit range is requested.
    if page_range is not None:
        result = converter.convert(pdf_path, page_range=page_range)
    else:
        result = converter.convert(pdf_path)
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=settings.embedding_model,
        max_tokens=max_tokens - overlap_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    raw_chunks = list(chunker.chunk(result.document))
    detected_language = "unknown"
    language_sample = " ".join(chunk.text for chunk in raw_chunks)[:5000]
    if language_sample.strip():
        try:
            detected_language = detect(language_sample)
        except Exception:  # short/noisy OCR samples can be undetectable
            pass

    output: list[Chunk] = []
    previous_tail: list[int] = []
    previous_pages: list[int] = []
    previous_boxes: list[BoundingBox] = []
    for index, raw_chunk in enumerate(raw_chunks):
        contextualized = normalize_text(chunker.contextualize(raw_chunk))
        current_ids = tokenizer.get_tokenizer().encode(
            contextualized, add_special_tokens=False
        )
        prefix = previous_tail[-overlap_tokens:] if index else []
        allowed = max_tokens - len(prefix)
        combined = prefix + current_ids[:allowed]
        text = normalize_text(tokenizer.get_tokenizer().decode(combined))
        previous_tail = current_ids
        pages, boxes = _extract_provenance(list(raw_chunk.meta.doc_items))
        if prefix:
            pages = sorted(set(previous_pages + pages))
            boxes = list(dict.fromkeys(previous_boxes + boxes))
        output.append(
            Chunk(
                chunk_id=stable_chunk_id(document_sha256, index, text),
                text=text,
                doc_id=document_sha256[:16],
                filename=pdf_path.name,
                title=title,
                sha256=document_sha256,
                page_numbers=pages,
                bounding_boxes=boxes,
                headings=list(raw_chunk.meta.headings or []),
                language=detected_language,
                extraction_method="docling+tesseract",
                token_count=len(combined),
                overlap_tokens=len(prefix),
            )
        )
        previous_pages, previous_boxes = _extract_provenance(list(raw_chunk.meta.doc_items))
    return output


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.payload(), ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> list[Chunk]:
    return [
        chunk_from_payload(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ingest_manifest(
    manifest_path: Path,
    *,
    settings: Settings,
    selected_id: str | None = None,
    skip_download: bool = False,
) -> list[dict[str, Any]]:
    settings.ensure_directories()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", [])
    if not selected_id and len(documents) < 10:
        raise ValueError("The full corpus manifest must contain at least ten documents")
    if selected_id:
        documents = [item for item in documents if item["id"] == selected_id]
        if not documents:
            raise ValueError(f"Unknown document id: {selected_id}")
    embedder = LocalEmbeddingModel(
        settings.embedding_model, str(settings.model_cache_dir / "huggingface")
    )
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    store.ensure_collection(embedder.dimension)
    report_path = settings.artifact_dir / "ingestion-report.json"
    existing_reports: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        existing_reports = {
            str(item["id"]): item
            for item in json.loads(report_path.read_text(encoding="utf-8"))
        }
    processed_reports: list[dict[str, Any]] = []
    for document in documents:
        started = time.perf_counter()
        pdf_path = settings.data_dir / "corpus" / document["filename"]
        try:
            if not skip_download:
                download_document(document["url"], pdf_path)
            inspection = inspect_pdf(pdf_path, min_pages=int(document.get("min_pages", 200)))
            sha256 = file_sha256(pdf_path)
            expected = document.get("sha256")
            if expected and sha256 != expected:
                raise ValueError(f"Checksum mismatch for {pdf_path.name}")
            cache_path = settings.artifact_dir / "chunks" / f"{sha256}.jsonl"
            cache_hit = cache_path.exists()
            if cache_hit:
                chunks = _read_chunks(cache_path)
            else:
                chunks = convert_and_chunk(
                    pdf_path,
                    title=document["title"],
                    document_sha256=sha256,
                    settings=settings,
                )
            if not chunks:
                raise ValueError("Docling produced no chunks")
            _write_chunks(cache_path, chunks)
            vectors = embedder.encode_documents([chunk.text for chunk in chunks])
            store.replace_document(sha256[:16], chunks, vectors)
            report = {
                "id": document["id"],
                "title": document["title"],
                "filename": pdf_path.name,
                "status": "complete",
                "sha256": sha256,
                "url": document["url"],
                "chunks": len(chunks),
                "cache_hit": cache_hit,
                "duration_seconds": round(time.perf_counter() - started, 2),
                **inspection,
            }
        except Exception as exc:
            LOGGER.exception("Failed to ingest %s", document.get("id"))
            report = {
                "id": document.get("id"),
                "title": document.get("title"),
                "filename": document.get("filename"),
                "status": "failed",
                "error": str(exc),
                "duration_seconds": round(time.perf_counter() - started, 2),
            }
        existing_reports[str(report["id"])] = report
        processed_reports.append(report)
        report_path.write_text(
            json.dumps(list(existing_reports.values()), indent=2), encoding="utf-8"
        )
    reports = list(existing_reports.values())
    lock_entries = [
        {
            key: item.get(key)
            for key in ("id", "filename", "url", "sha256", "page_count")
        }
        for item in reports
        if item.get("status") == "complete"
    ]
    (settings.artifact_dir / "corpus-lock.json").write_text(
        json.dumps(lock_entries, indent=2), encoding="utf-8"
    )
    return processed_reports if selected_id else reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus.yaml"))
    parser.add_argument("--document-id")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reports = ingest_manifest(
        args.manifest,
        settings=Settings.load(),
        selected_id=args.document_id,
        skip_download=args.skip_download,
    )
    print(json.dumps(reports, indent=2))
    if any(item["status"] == "failed" for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
