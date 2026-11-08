"""Validate citation identifiers in generated research answers."""

import re
from dataclasses import dataclass

from evidence_rag.generation.evidence import Evidence


class CitationValidationError(ValueError):
    """Raised when generated answer citations are invalid or missing."""

    def __init__(self, message: str, feedback: str) -> None:
        super().__init__(message)
        self.feedback = feedback


def normalize_citations(answer: str) -> str:
    """Convert parenthesized citations to bracketed format."""
    return re.sub(r"\((E\d+)\)", r"[\1]", answer)


@dataclass(frozen=True)
class CitationValidationResult:
    cited_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]
    uncited_ids: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return True when no unknown citation identifiers are present."""
        return not self.invalid_ids


def validate_citations(
    answer: str,
    evidence: list[Evidence] | tuple[Evidence, ...],
) -> CitationValidationResult:
    """Check whether answer citations match supplied evidence IDs."""

    cited_ids = tuple(dict.fromkeys(re.findall(r"\[E\d+\]", answer)))

    available_ids = {item.citation for item in evidence}

    valid_ids = tuple(citation for citation in cited_ids if citation in available_ids)

    invalid_ids = tuple(citation for citation in cited_ids if citation not in available_ids)

    uncited_ids = tuple(item.citation for item in evidence if item.citation not in cited_ids)

    return CitationValidationResult(
        cited_ids=cited_ids,
        valid_ids=valid_ids,
        invalid_ids=invalid_ids,
        uncited_ids=uncited_ids,
    )
