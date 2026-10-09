"""End-to-end query orchestration with independently timed stages."""

from __future__ import annotations

import time

from rag_app.config import Settings
from rag_app.embeddings import LocalEmbeddingModel, LocalReranker
from rag_app.generation import GroundedGenerator
from rag_app.metrics import append_query_metric
from rag_app.models import AnswerResult, StageTimings
from rag_app.retrieval import deduplicate
from rag_app.vector_store import QdrantStore


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        cache = str(settings.model_cache_dir / "huggingface")
        self.settings = settings
        self.embedder = LocalEmbeddingModel(settings.embedding_model, cache)
        self.reranker = LocalReranker(settings.reranker_model, cache)
        self.store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
        self.generator = GroundedGenerator(
            settings.groq_api_key, settings.groq_model, settings.max_answer_tokens
        )

    def ask(self, question: str) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        total_started = time.perf_counter()
        timings = StageTimings()

        started = time.perf_counter()
        query_vector = self.embedder.encode_query(question)
        timings.embedding_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        candidates = self.store.search(query_vector, limit=self.settings.top_k)
        timings.retrieval_ms = (time.perf_counter() - started) * 1000

        if not candidates or candidates[0].dense_score < self.settings.min_relevance_score:
            result = AnswerResult(
                question=question,
                answer="Insufficient evidence in the indexed reports.",
                sources=candidates[: self.settings.final_k],
                timings=timings,
                refused=True,
            )
        else:
            started = time.perf_counter()
            scores = self.reranker.score(question, [item.chunk.text for item in candidates])
            for candidate, score in zip(candidates, scores, strict=True):
                candidate.rerank_score = score
            sources = deduplicate(candidates, limit=self.settings.final_k)
            timings.reranking_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            try:
                answer, warning = self.generator.generate(question, sources)
                refused = answer.startswith("Insufficient evidence")
            except Exception as exc:
                answer = "Generation is unavailable. Review the retrieved evidence below."
                warning = str(exc)
                refused = False
            timings.generation_ms = (time.perf_counter() - started) * 1000
            result = AnswerResult(
                question=question,
                answer=answer,
                sources=sources,
                timings=timings,
                refused=refused,
                warning=warning,
            )

        timings.total_ms = (time.perf_counter() - total_started) * 1000
        append_query_metric(self.settings.artifact_dir / "query-metrics.jsonl", result)
        return result
