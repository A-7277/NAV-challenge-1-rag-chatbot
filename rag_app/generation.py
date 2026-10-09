"""Grounded generation and strict citation checking."""

from __future__ import annotations

from rag_app.models import RetrievedChunk
from rag_app.text import factual_paragraphs_without_citations, invalid_citations

SYSTEM_PROMPT = """You answer questions only from the supplied IPCC passages.
Rules:
1. Never use outside knowledge or invent facts.
2. Cite every factual paragraph with one or more labels such as [S1].
3. Use only source labels present in the supplied context.
4. If the passages do not answer the question, reply exactly:
   Insufficient evidence in the indexed reports.
5. Be concise and do not add a bibliography; the application renders source details.
"""


def build_context(sources: list[RetrievedChunk]) -> str:
    blocks = []
    for item in sources:
        pages = ", ".join(str(page) for page in item.chunk.page_numbers) or "unknown"
        blocks.append(
            f"[{item.source_label}] {item.chunk.filename}, page(s) {pages}\n"
            f"Headings: {' > '.join(item.chunk.headings) or 'n/a'}\n"
            f"Passage: {item.chunk.text}"
        )
    return "\n\n".join(blocks)


class GroundedGenerator:
    def __init__(self, api_key: str, model: str, max_tokens: int = 220) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, sources: list[RetrievedChunk]) -> tuple[str, str | None]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        from groq import Groq

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{build_context(sources)}\n\nQuestion: {question}",
                },
            ],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        answer = (response.choices[0].message.content or "").strip()
        invalid = invalid_citations(answer, len(sources))
        uncited = factual_paragraphs_without_citations(answer)
        warnings = []
        if invalid:
            warnings.append(f"Removed invalid citation labels: {sorted(invalid)}")
            for number in invalid:
                answer = answer.replace(f"[S{number}]", "")
        if uncited and not answer.startswith("Insufficient evidence"):
            warnings.append("One or more factual paragraphs lack citations")
        return answer, "; ".join(warnings) or None

