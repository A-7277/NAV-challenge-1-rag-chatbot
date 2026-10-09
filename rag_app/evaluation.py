"""Retrieval and optional end-to-end evaluation for the labeled question set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from rag_app.config import Settings
from rag_app.embeddings import LocalEmbeddingModel, LocalReranker
from rag_app.pipeline import RagPipeline
from rag_app.retrieval import deduplicate
from rag_app.text import invalid_citations
from rag_app.vector_store import QdrantStore


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def evaluate(
    question_path: Path, *, settings: Settings, with_generation: bool = False
) -> dict:
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    cache = str(settings.model_cache_dir / "huggingface")
    embedder = LocalEmbeddingModel(settings.embedding_model, cache)
    reranker = LocalReranker(settings.reranker_model, cache)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    pipeline = RagPipeline(settings) if with_generation else None
    records = []
    reciprocal_ranks: list[float] = []
    hits = 0
    latencies = []
    valid_citation_answers = 0
    refusal_hits = 0
    for item in questions:
        started = time.perf_counter()
        vector = embedder.encode_query(item["question"])
        candidates = store.search(vector, limit=settings.top_k)
        scores = reranker.score(item["question"], [result.chunk.text for result in candidates])
        for result, score in zip(candidates, scores, strict=True):
            result.rerank_score = score
        ranked = deduplicate(candidates, limit=settings.final_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        filenames = [result.chunk.filename for result in ranked]
        expected = set(item["expected_filenames"])
        first_rank = next(
            (rank for rank, filename in enumerate(filenames, start=1) if filename in expected), None
        )
        if item["answerable"]:
            hits += int(first_rank is not None)
            reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        record = {
            "id": item["id"],
            "question": item["question"],
            "filenames": filenames,
            "first_relevant_rank": first_rank,
            "retrieval_latency_ms": round(elapsed_ms, 2),
        }
        if pipeline:
            answer = pipeline.ask(item["question"])
            record["answer"] = answer.answer
            record["refused"] = answer.refused
            if not invalid_citations(answer.answer, len(answer.sources)):
                valid_citation_answers += 1
            if not item["answerable"] and answer.refused:
                refusal_hits += 1
        records.append(record)
    answerable_count = sum(1 for item in questions if item["answerable"])
    unanswerable_count = len(questions) - answerable_count
    summary = {
        "question_count": len(questions),
        "recall_at_5": round(hits / answerable_count, 4) if answerable_count else 0.0,
        "mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "retrieval_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "retrieval_p95_ms": round(_percentile(latencies, 0.95), 2),
        "citation_label_validity": (
            round(valid_citation_answers / len(questions), 4) if pipeline else None
        ),
        "refusal_accuracy": (
            round(refusal_hits / unanswerable_count, 4)
            if pipeline and unanswerable_count
            else None
        ),
        "hallucination_rate": None,
        "hallucination_note": (
            "Populate after manual claim-level review; do not infer it from citations."
        ),
    }
    report = {"summary": summary, "records": records}
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifact_dir / "evaluation-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", type=Path, default=Path("data/evaluation/questions.json")
    )
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        args.questions, settings=Settings.load(), with_generation=args.with_generation
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
