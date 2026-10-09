"""Two-stage retrieval with overlap-aware deduplication."""

from __future__ import annotations

from rag_app.embeddings import LocalEmbeddingModel, LocalReranker
from rag_app.models import RetrievedChunk
from rag_app.vector_store import QdrantStore


def _page_overlap(left: RetrievedChunk, right: RetrievedChunk) -> bool:
    return bool(set(left.chunk.page_numbers) & set(right.chunk.page_numbers))


def deduplicate(results: list[RetrievedChunk], *, limit: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    for candidate in sorted(results, key=lambda item: item.score, reverse=True):
        duplicate = any(
            candidate.chunk.doc_id == existing.chunk.doc_id
            and _page_overlap(candidate, existing)
            and candidate.chunk.text[:180] in existing.chunk.text
            for existing in selected
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) == limit:
            break
    for index, item in enumerate(selected, start=1):
        item.source_label = f"S{index}"
    return selected


class Retriever:
    def __init__(
        self,
        embedder: LocalEmbeddingModel,
        store: QdrantStore,
        reranker: LocalReranker,
        *,
        top_k: int = 12,
        final_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.top_k = top_k
        self.final_k = final_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        vector = self.embedder.encode_query(question)
        candidates = self.store.search(vector, limit=self.top_k)
        scores = self.reranker.score(question, [item.chunk.text for item in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = score
        return deduplicate(candidates, limit=self.final_k)

