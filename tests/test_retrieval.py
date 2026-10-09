from rag_app.models import Chunk, RetrievedChunk
from rag_app.retrieval import deduplicate


def result(identifier: int, page: int, score: float, text: str = "passage") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{identifier:032x}",
        text=text,
        doc_id="same-doc",
        filename="report.pdf",
        title="Report",
        sha256="a" * 64,
        page_numbers=[page],
    )
    return RetrievedChunk(chunk=chunk, dense_score=score, rerank_score=score)


def test_deduplicate_removes_contained_passage_on_same_page_and_labels_sources():
    ranked = deduplicate(
        [
            result(1, 3, 0.9, "a useful passage with details"),
            result(2, 3, 0.8, "a useful passage"),
            result(3, 4, 0.7, "different page"),
        ],
        limit=5,
    )
    assert [item.chunk.chunk_id for item in ranked] == [f"{1:032x}", f"{3:032x}"]
    assert [item.source_label for item in ranked] == ["S1", "S2"]

