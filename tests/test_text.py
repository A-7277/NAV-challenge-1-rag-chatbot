from rag_app.text import (
    cited_source_numbers,
    factual_paragraphs_without_citations,
    invalid_citations,
    normalize_text,
    remove_repeated_margin_lines,
    stable_chunk_id,
    token_windows,
)


def test_normalization_is_deterministic_and_repairs_line_break_hyphens():
    assert normalize_text("  climate  inter-\nvention\r\n\r\n\r\nworks ") == (
        "climate intervention\n\nworks"
    )


def test_repeated_margin_lines_are_removed_without_removing_body():
    pages = [
        "IPCC REPORT\nFirst body\nRepeated body phrase\nClosing note A\n12",
        "IPCC REPORT\nSecond body\nRepeated body phrase\nClosing note B\n13",
        "IPCC REPORT\nThird body\nRepeated body phrase\nClosing note C\n14",
    ]
    cleaned = remove_repeated_margin_lines(pages)
    assert all("IPCC REPORT" not in page for page in cleaned)
    assert all("Repeated body phrase" in page for page in cleaned)


def test_token_windows_have_requested_overlap():
    windows = list(token_windows(list(range(20)), max_tokens=8, overlap_tokens=2))
    assert windows[0] == (list(range(8)), 0)
    assert windows[1][0][:2] == [6, 7]
    assert windows[1][1] == 2


def test_stable_chunk_ids_change_with_content_or_position():
    first = stable_chunk_id("abc", 0, "text")
    assert first == stable_chunk_id("abc", 0, "text")
    assert first != stable_chunk_id("abc", 1, "text")
    assert first != stable_chunk_id("abc", 0, "other")


def test_citation_validation_and_coverage():
    answer = (
        "Warming has documented impacts [S1].\n\n"
        "A longer factual paragraph deliberately has no supporting source."
    )
    assert cited_source_numbers(answer) == {1}
    assert invalid_citations(answer + " [S7]", source_count=2) == {7}
    assert factual_paragraphs_without_citations(answer) == [
        "A longer factual paragraph deliberately has no supporting source."
    ]
