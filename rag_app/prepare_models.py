"""Pre-download every model needed by ingestion and warm query execution."""

from __future__ import annotations

import subprocess
import sys

from rag_app.config import Settings
from rag_app.embeddings import LocalEmbeddingModel, LocalReranker


def main() -> None:
    settings = Settings.load()
    settings.ensure_directories()
    docling_dir = settings.model_cache_dir / "docling"
    docling_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["docling-tools", "models", "download", "--output-dir", str(docling_dir)],
        check=False,
    )
    if result.returncode != 0:
        # docling-tools can fail if an optional OCR backend (e.g. RapidOCR) has a
        # version mismatch. This project uses Tesseract, so the failure is harmless
        # and layout/formula models downloaded before the crash are still usable.
        print(
            "WARNING: docling-tools models download exited with a non-zero status "
            "(likely a RapidOCR version mismatch). Tesseract-based OCR is unaffected. "
            "Continuing with embedding and reranker model download.",
            file=sys.stderr,
        )
    cache = str(settings.model_cache_dir / "huggingface")
    embedder = LocalEmbeddingModel(settings.embedding_model, cache)
    reranker = LocalReranker(settings.reranker_model, cache)
    print(f"Embedding dimension: {embedder.dimension}")
    print(f"Reranker smoke score: {reranker.score('climate risk', ['climate risk'])[0]:.4f}")
    print(f"Models prepared under {settings.model_cache_dir}")


if __name__ == "__main__":
    main()
