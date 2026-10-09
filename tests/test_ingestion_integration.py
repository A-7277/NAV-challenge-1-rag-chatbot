import os
from pathlib import Path

import pytest

pytest.importorskip("pypdf")

from rag_app.config import Settings
from rag_app.ingestion import convert_and_chunk, file_sha256

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_HEAVY_TESTS") != "1",
    reason="Set RUN_HEAVY_TESTS=1 after installing Docling and Tesseract",
)
def test_image_only_pdf_is_ocrd_with_page_provenance(tmp_path: Path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 1600), "white")
    ImageDraw.Draw(image).text(
        (100, 200), "Observed warming has affected natural systems.", fill="black"
    )
    pdf = tmp_path / "scan.pdf"
    image.save(pdf, "PDF", resolution=150)
    settings = Settings.load()
    chunks = convert_and_chunk(
        pdf,
        title="OCR fixture",
        document_sha256=file_sha256(pdf),
        settings=settings,
        page_range=(1, 1),
    )
    assert chunks
    assert any("warming" in chunk.text.lower() for chunk in chunks)
    assert all(chunk.page_numbers == [1] for chunk in chunks)
    assert any(chunk.bounding_boxes for chunk in chunks)
