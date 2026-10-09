"""Qdrant persistence with explicit metadata round-tripping."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np

from rag_app.models import BoundingBox, Chunk, RetrievedChunk


class QdrantStore:
    def __init__(self, url: str, collection_name: str) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, timeout=60)
        self.collection_name = collection_name

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import (
            Distance,
            HnswConfigDiff,
            OptimizersConfigDiff,
            PayloadSchemaType,
            VectorParams,
        )

        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            configured_size = info.config.params.vectors.size
            if configured_size != dimension:
                raise ValueError(
                    f"Collection dimension {configured_size} does not match "
                    f"model dimension {dimension}"
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=1_000),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1_000),
        )
        for field in ("doc_id", "filename", "language"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def replace_document(self, doc_id: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
            PointStruct,
        )

        if len(chunks) != len(vectors):
            raise ValueError("Chunk/vector count mismatch")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            points = [
                PointStruct(
                    id=str(uuid.UUID(chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=chunk.payload(),
                )
                for chunk, vector in zip(
                    chunks[start : start + batch_size],
                    vectors[start : start + batch_size],
                    strict=True,
                )
            ]
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

    def search(self, query_vector: np.ndarray, *, limit: int) -> list[RetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=chunk_from_payload(point.payload or {}), dense_score=point.score)
            for point in response.points
        ]

    def collection_stats(self) -> dict[str, int | str]:
        info = self.client.get_collection(self.collection_name)
        return {
            "status": str(info.status),
            "points_count": int(info.points_count or 0),
            "indexed_vectors_count": int(info.indexed_vectors_count or 0),
        }


def chunk_from_payload(payload: dict) -> Chunk:
    boxes = [BoundingBox(**box) for box in payload.get("bounding_boxes", [])]
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        text=str(payload["text"]),
        doc_id=str(payload["doc_id"]),
        filename=str(payload["filename"]),
        title=str(payload.get("title", payload["filename"])),
        sha256=str(payload.get("sha256", "")),
        page_numbers=[int(page) for page in payload.get("page_numbers", [])],
        bounding_boxes=boxes,
        headings=[str(value) for value in payload.get("headings", [])],
        language=str(payload.get("language", "unknown")),
        extraction_method=str(payload.get("extraction_method", "docling")),
        token_count=int(payload.get("token_count", 0)),
        overlap_tokens=int(payload.get("overlap_tokens", 0)),
    )
