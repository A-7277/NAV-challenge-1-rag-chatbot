import json

from rag_app.metrics import load_latency_summary


def test_latency_summary_uses_nearest_rank_p95(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"timings": {"total_ms": value}}) for value in [10, 20, 30, 40]
        ),
        encoding="utf-8",
    )
    summary = load_latency_summary(path)
    assert summary == {"queries": 4, "p50_ms": 25.0, "p95_ms": 40.0}

