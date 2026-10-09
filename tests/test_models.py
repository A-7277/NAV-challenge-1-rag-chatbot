from rag_app.models import BoundingBox, Chunk, RetrievedChunk


def make_chunk(chunk_id: str = "0" * 32) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="Grounded passage",
        doc_id="doc-1",
        filename="report.pdf",
        title="Report",
        sha256="a" * 64,
        page_numbers=[4, 5],
        bounding_boxes=[BoundingBox(4, 1.0, 2.0, 3.0, 4.0)],
        headings=["Chapter", "Section"],
        language="en",
        token_count=42,
        overlap_tokens=6,
    )


def test_chunk_payload_contains_traceable_metadata():
    payload = make_chunk().payload()
    assert payload["page_start"] == 4
    assert payload["page_end"] == 5
    assert payload["bounding_boxes"][0]["page"] == 4
    assert payload["headings"] == ["Chapter", "Section"]


def test_retrieved_chunk_prefers_rerank_score():
    item = RetrievedChunk(make_chunk(), dense_score=0.9, rerank_score=0.2)
    assert item.score == 0.2

