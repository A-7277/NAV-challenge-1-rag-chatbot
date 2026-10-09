"""Append-only query metrics used by the UI and evaluator."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from rag_app.models import AnswerResult


def append_query_metric(path: Path, result: AnswerResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "question": result.question,
        "refused": result.refused,
        "source_count": len(result.sources),
        "source_ids": [item.chunk.chunk_id for item in result.sources],
        "timings": asdict(result.timings),
        "warning": result.warning,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_latency_summary(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {"queries": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(float(json.loads(line)["timings"]["total_ms"]))
    if not values:
        return {"queries": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered) + 0.5) - 1))
    return {
        "queries": len(values),
        "p50_ms": round(statistics.median(values), 2),
        "p95_ms": round(ordered[p95_index], 2),
    }

