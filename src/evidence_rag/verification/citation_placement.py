"""Detect misplaced evidence citations in generated answers."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CitationPlacementResult:
    leading_citations: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.leading_citations


def check_citation_placement(answer: str) -> CitationPlacementResult:
    """Detect citations at the beginning of an answer or sentence."""
    pattern = r"(?:^|(?<=[.!?])\s+)\s*(\[E\d+\])"

    matches = tuple(re.findall(pattern, answer.strip()))

    return CitationPlacementResult(leading_citations=matches)
