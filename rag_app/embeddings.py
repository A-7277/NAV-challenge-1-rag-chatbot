"""Lazy local embedding and reranking adapters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class LocalEmbeddingModel:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
        return self._model

    def encode_documents(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._load().encode_document(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self._load().encode_query(
            query, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())


class LocalReranker:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, cache_folder=self.cache_dir)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        raw_scores = self._load().predict(
            [(query, passage) for passage in passages], show_progress_bar=False
        )
        # MS MARCO models return relevance logits. Their absolute scale is not a
        # probability, but ordering is exactly what reranking needs.
        return [float(score) for score in raw_scores]
