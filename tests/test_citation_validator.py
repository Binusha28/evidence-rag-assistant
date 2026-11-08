"""Tests for citation identifier validation."""

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.citation_validator import (
    normalize_citations,
    validate_citations,
)


def make_evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"paper.pdf:p1:{evidence_id}",
        source="paper.pdf",
        page_number=1,
        text="Example research evidence.",
        retrieval_score=0.8,
    )


def test_valid_citations():
    result = validate_citations(
        "Research supports this [E1] and this [E2].",
        [make_evidence("E1"), make_evidence("E2")],
    )

    assert result.is_valid
    assert result.valid_ids == ("[E1]", "[E2]")
    assert result.invalid_ids == ()


def test_fabricated_citation():
    result = validate_citations(
        "The model claims this [E9].",
        [make_evidence("E1")],
    )

    assert not result.is_valid
    assert result.invalid_ids == ("[E9]",)


def test_uncited_evidence():
    result = validate_citations(
        "Only one source is cited [E1].",
        [make_evidence("E1"), make_evidence("E2")],
    )

    assert result.uncited_ids == ("[E2]",)


def test_duplicate_citations():
    result = validate_citations(
        "First claim [E1]. Second claim [E1].",
        [make_evidence("E1")],
    )

    assert result.cited_ids == ("[E1]",)


def test_no_citations():
    result = validate_citations(
        "An answer without citations.",
        [make_evidence("E1")],
    )

    assert result.cited_ids == ()
    assert result.uncited_ids == ("[E1]",)


def test_normalize_parenthesized_citations():
    answer = "The model found evidence (E1) and (E2)."

    result = normalize_citations(answer)

    assert result == "The model found evidence [E1] and [E2]."


def test_normalize_preserves_existing_citations():
    answer = "The model found evidence [E1] and (E2)."

    result = normalize_citations(answer)

    assert result == "The model found evidence [E1] and [E2]."


def test_normalize_does_not_modify_other_references():
    answer = "The study (2022) reported results (E3)."

    result = normalize_citations(answer)

    assert result == "The study (2022) reported results [E3]."
