"""Small serializable models shared by ingestion, retrieval and the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    page: int
    left: float
    top: float
    right: float
    bottom: float
    origin: str = "BOTTOMLEFT"


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    filename: str
    title: str
    sha256: str
    page_numbers: list[int]
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    language: str = "unknown"
    extraction_method: str = "docling"
    token_count: int = 0
    overlap_tokens: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "title": self.title,
            "sha256": self.sha256,
            "page_numbers": self.page_numbers,
            "page_start": min(self.page_numbers) if self.page_numbers else None,
            "page_end": max(self.page_numbers) if self.page_numbers else None,
            "bounding_boxes": [asdict(box) for box in self.bounding_boxes],
            "headings": self.headings,
            "language": self.language,
            "extraction_method": self.extraction_method,
            "token_count": self.token_count,
            "overlap_tokens": self.overlap_tokens,
        }


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    rerank_score: float | None = None
    source_label: str | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.dense_score


@dataclass(slots=True)
class StageTimings:
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    timings: StageTimings
    refused: bool = False
    warning: str | None = None

