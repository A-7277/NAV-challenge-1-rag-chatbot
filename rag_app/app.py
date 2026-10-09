"""Streamlit interface for chat, retrieval inspection and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict

import streamlit as st
import yaml

from rag_app.config import Settings
from rag_app.metrics import load_latency_summary
from rag_app.pipeline import RagPipeline

st.set_page_config(page_title="Traceable Climate RAG", page_icon="📚", layout="wide")
settings = Settings.load()
settings.ensure_directories()


@st.cache_resource(show_spinner="Loading local retrieval models…")
def get_pipeline() -> RagPipeline:
    return RagPipeline(settings)


def render_source(item) -> None:
    pages = ", ".join(str(page) for page in item.chunk.page_numbers) or "unknown"
    label = item.source_label or "source"
    with st.expander(
        f"[{label}] {item.chunk.filename} · page(s) {pages} · score {item.score:.3f}"
    ):
        if item.chunk.headings:
            st.caption(" › ".join(item.chunk.headings))
        st.write(item.chunk.text)
        st.json(
            {
                "dense_score": round(item.dense_score, 4),
                "rerank_score": (
                    round(item.rerank_score, 4) if item.rerank_score is not None else None
                ),
                "language": item.chunk.language,
                "extraction_method": item.chunk.extraction_method,
                "bounding_boxes": [asdict(box) for box in item.chunk.bounding_boxes[:8]],
            },
            expanded=False,
        )


st.title("Traceable Climate Report Assistant")
st.caption("Local embeddings + Qdrant retrieval · grounded generation · page-level provenance")
chat_tab, retrieval_tab, documents_tab, evaluation_tab = st.tabs(
    ["Chat", "Retrieval Inspector", "Documents", "Evaluation"]
)

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask a question about the indexed IPCC reports")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and grounding the answer…"):
                    result = get_pipeline().ask(question)
                st.markdown(result.answer)
                st.caption(
                    f"Total {result.timings.total_ms / 1000:.2f}s · "
                    f"retrieval {result.timings.retrieval_ms:.0f}ms · "
                    f"reranking {result.timings.reranking_ms:.0f}ms · "
                    f"generation {result.timings.generation_ms:.0f}ms"
                )
                if result.warning:
                    st.warning(result.warning)
                for source in result.sources:
                    render_source(source)
                st.session_state.latest_result = result
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer}
                )
            except Exception as exc:
                st.error(f"The query could not be completed: {exc}")

with retrieval_tab:
    result = st.session_state.get("latest_result")
    if not result:
        st.info("Ask a question in Chat to inspect its retrieval trace.")
    else:
        st.subheader(result.question)
        chart_data = {
            item.source_label: {
                "dense": item.dense_score,
                "reranker": item.rerank_score or 0.0,
            }
            for item in result.sources
        }
        st.bar_chart(chart_data)
        for source in result.sources:
            render_source(source)

with documents_tab:
    manifest = yaml.safe_load(settings.corpus_manifest.read_text(encoding="utf-8"))
    configured_documents = manifest.get("documents", [])
    st.subheader("Ingestion pipeline")
    st.code(
        "PDF download + validation → Docling/Tesseract extraction → "
        "deterministic chunks → BGE embeddings → Qdrant HNSW",
        language=None,
    )
    st.caption(
        "Each chunk retains its document, filename, page numbers, extraction method, "
        "headings, language, and available bounding boxes."
    )
    report_path = settings.artifact_dir / "ingestion-report.json"
    if not report_path.exists():
        st.warning("No ingestion report found. Run `rag-ingest` first.")
        st.dataframe(configured_documents, use_container_width=True, hide_index=True)
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        complete = sum(item.get("status") == "complete" for item in report)
        pages = sum(item.get("page_count", 0) for item in report)
        chunks = sum(item.get("chunks", 0) for item in report)
        left, middle, right = st.columns(3)
        left.metric("Documents", f"{complete}/{len(configured_documents)}")
        middle.metric("Pages", pages)
        right.metric("Chunks", chunks)
        st.dataframe(report, use_container_width=True, hide_index=True)

with evaluation_tab:
    evaluation_path = settings.artifact_dir / "evaluation-report.json"
    latency = load_latency_summary(settings.artifact_dir / "query-metrics.jsonl")
    left, middle, right = st.columns(3)
    left.metric("Session queries", latency["queries"])
    middle.metric("Latency p50", f"{latency['p50_ms'] / 1000:.2f}s")
    right.metric("Latency p95", f"{latency['p95_ms'] / 1000:.2f}s")
    if evaluation_path.exists():
        report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        st.subheader("Saved evaluation")
        st.json(report["summary"])
        st.dataframe(report["records"], use_container_width=True, hide_index=True)
    else:
        st.info("Run `rag-evaluate` to create the labeled evaluation report.")
