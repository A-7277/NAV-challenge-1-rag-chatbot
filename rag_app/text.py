"""Deterministic text normalization, overlap and citation utilities."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=[a-z])")
_CITATION = re.compile(r"\[S(\d+)\]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub("", text)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def stable_chunk_id(document_sha256: str, index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_sha256}:{index}:{text}".encode()
    ).hexdigest()
    return digest[:32]


def remove_repeated_margin_lines(
    pages: Sequence[str], *, min_fraction: float = 0.6, edge_lines: int = 2
) -> list[str]:
    """Remove repeated first/last lines while preserving body repetitions."""
    if not pages:
        return []
    candidates: Counter[str] = Counter()
    normalized_pages: list[list[str]] = []
    for page in pages:
        lines = [line for line in normalize_text(page).splitlines() if line]
        normalized_pages.append(lines)
        edge = set(lines[:edge_lines] + lines[-edge_lines:])
        candidates.update(line for line in edge if 2 < len(line) < 160)
    threshold = max(2, int(len(pages) * min_fraction + 0.999))
    repeated = {line for line, count in candidates.items() if count >= threshold}
    return ["\n".join(line for line in lines if line not in repeated) for lines in normalized_pages]


def token_windows(
    token_ids: Sequence[int], *, max_tokens: int = 700, overlap_tokens: int = 105
) -> Iterable[tuple[list[int], int]]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")
    step = max_tokens - overlap_tokens
    start = 0
    while start < len(token_ids):
        window = list(token_ids[start : start + max_tokens])
        yield window, min(overlap_tokens, start)
        if start + max_tokens >= len(token_ids):
            break
        start += step


def cited_source_numbers(answer: str) -> set[int]:
    return {int(match) for match in _CITATION.findall(answer)}


def invalid_citations(answer: str, source_count: int) -> set[int]:
    return {number for number in cited_source_numbers(answer) if not 1 <= number <= source_count}


def factual_paragraphs_without_citations(answer: str) -> list[str]:
    paragraphs = [part.strip() for part in answer.split("\n\n") if part.strip()]
    return [
        paragraph
        for paragraph in paragraphs
        if len(paragraph.split()) >= 8
        and not paragraph.lower().startswith(("sources", "insufficient evidence"))
        and not _CITATION.search(paragraph)
    ]
