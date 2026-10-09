"""Environment-backed configuration with no import-time network side effects."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _as_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    artifact_dir: Path
    model_cache_dir: Path
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    reranker_model: str
    groq_api_key: str
    groq_model: str
    top_k: int
    final_k: int
    min_relevance_score: float
    max_answer_tokens: int

    @classmethod
    def load(cls, env_file: Path | None = None) -> Settings:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(env_file or project_root / ".env", override=False)
        return cls(
            project_root=project_root,
            data_dir=Path(os.getenv("DATA_DIR", project_root / "data")).resolve(),
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", project_root / "artifacts")).resolve(),
            model_cache_dir=Path(
                os.getenv("MODEL_CACHE_DIR", project_root / "models")
            ).resolve(),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "ipcc_reports"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            top_k=_as_int("TOP_K", 12),
            final_k=_as_int("FINAL_K", 5),
            min_relevance_score=_as_float("MIN_RELEVANCE_SCORE", 0.20),
            max_answer_tokens=_as_int("MAX_ANSWER_TOKENS", 220),
        )

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.artifact_dir, self.model_cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

