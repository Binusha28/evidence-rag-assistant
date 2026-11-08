"""Tests for unsupported clauses inside partially cited sentences."""

from evidence_rag.verification.claim_extractor import (
    extract_uncited_sentences,
)


def test_uncited_clause_after_cited_claim_is_detected():
    answer = "CNNs generate feature maps [E1], and ViTs achieve 100% accuracy."

    uncited = extract_uncited_sentences(answer)

    assert uncited, (
        "An uncited factual clause must not be hidden by a citation attached to another clause."
    )


def test_fully_cited_comparison_is_not_flagged():
    answer = "CNNs generate feature maps [E1], while ViTs use image patches [E2]."

    assert extract_uncited_sentences(answer) == []


def test_multiple_citations_for_single_claim_are_not_flagged():
    answer = "Both studies investigate image classification [E1][E2]."

    assert extract_uncited_sentences(answer) == []
