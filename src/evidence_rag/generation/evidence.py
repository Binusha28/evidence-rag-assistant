"""Structured evidence models for grounded RAG answers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    """A retrieved passage with traceable document provenance."""

    evidence_id: str
    chunk_id: str
    source: str
    page_number: int
    text: str
    retrieval_score: float

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("Evidence ID must not be empty")
        if not self.chunk_id.strip():
            raise ValueError("Chunk ID must not be empty")
        if not self.source.strip():
            raise ValueError("Source must not be empty")
        if self.page_number <= 0:
            raise ValueError("Page number must be positive")
        if not self.text.strip():
            raise ValueError("Evidence text must not be empty")

    @property
    def citation(self) -> str:
        """Return the citation marker used in generated answers."""
        return f"[{self.evidence_id}]"

    def format_for_prompt(self) -> str:
        """Format evidence for inclusion in an LLM prompt."""
        return (
            f"{self.citation}\n"
            f"Source: {self.source}\n"
            f"Page: {self.page_number}\n"
            f"Chunk ID: {self.chunk_id}\n"
            f"Content: {self.text}"
        )


def build_evidence(
    results: list,
) -> list[Evidence]:
    """Convert cited search results into structured evidence."""
    evidence_items = []

    for index, result in enumerate(results, start=1):
        evidence_items.append(
            Evidence(
                evidence_id=f"E{index}",
                chunk_id=result.chunk_id,
                source=result.source,
                page_number=result.page_number,
                text=result.text,
                retrieval_score=result.score,
            )
        )

    return evidence_items
