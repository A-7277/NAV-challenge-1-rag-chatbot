import json
from types import SimpleNamespace

from rag_app.models import Chunk, RetrievedChunk
from rag_app.pipeline import RagPipeline


class FakeEmbedder:
    def encode_query(self, _question):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, candidates):
        self.candidates = candidates

    def search(self, _vector, limit):
        return self.candidates[:limit]


class FakeReranker:
    def score(self, _question, passages):
        return [float(len(passages) - index) for index, _ in enumerate(passages)]


class FakeGenerator:
    def generate(self, _question, _sources):
        return "The assessed warming is evidence-backed [S1].", None


def candidate(score=0.8):
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="1" * 32,
            text="The report assesses global warming impacts.",
            doc_id="sr15",
            filename="sr15.pdf",
            title="Global Warming of 1.5 C",
            sha256="a" * 64,
            page_numbers=[51],
        ),
        dense_score=score,
    )


def pipeline(tmp_path, candidates):
    instance = RagPipeline.__new__(RagPipeline)
    instance.settings = SimpleNamespace(
        artifact_dir=tmp_path,
        top_k=12,
        final_k=5,
        min_relevance_score=0.3,
    )
    instance.embedder = FakeEmbedder()
    instance.store = FakeStore(candidates)
    instance.reranker = FakeReranker()
    instance.generator = FakeGenerator()
    return instance


def test_pipeline_answers_with_labeled_evidence_and_records_metrics(tmp_path):
    result = pipeline(tmp_path, [candidate()]).ask("What does the report assess?")

    assert result.answer.endswith("[S1].")
    assert result.sources[0].source_label == "S1"
    assert result.refused is False
    metric = json.loads((tmp_path / "query-metrics.jsonl").read_text(encoding="utf-8"))
    assert metric["source_ids"] == ["1" * 32]
    assert metric["timings"]["total_ms"] >= 0


def test_pipeline_refuses_when_best_dense_match_is_below_threshold(tmp_path):
    result = pipeline(tmp_path, [candidate(score=0.2)]).ask("An unrelated question")

    assert result.refused is True
    assert result.answer == "Insufficient evidence in the indexed reports."
    assert result.sources[0].source_label is None
