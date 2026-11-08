"""Tests for the combined answer verification report."""

from unittest.mock import Mock

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.answer_verification import (
    verify_generated_answer,
)
from evidence_rag.verification.semantic_verifier import ClaimVerificationResult


def make_evidence():
    return [
        Evidence(
            evidence_id="E1",
            chunk_id="test:p1:c0",
            source="test.pdf",
            page_number=1,
            text="CNNs identify retinal lesions.",
            retrieval_score=1.0,
        )
    ]


def make_verifier(label="entailment"):
    verifier = Mock()
    verifier.verify.return_value = ClaimVerificationResult(
        claim="CNNs identify retinal lesions.",
        evidence="CNNs identify retinal lesions.",
        label=label,
        confidence=0.98,
    )
    return verifier


def test_fully_supported_answer():
    report = verify_generated_answer(
        answer="CNNs identify retinal lesions [E1].",
        evidence=make_evidence(),
        verifier=make_verifier(),
    )

    assert report.is_verified is True
    assert len(report.assessments) == 1
    assert report.uncited_sentences == ()


def test_uncited_claim_fails_verification():
    report = verify_generated_answer(
        answer=("CNNs identify retinal lesions [E1]. The model achieves 100% accuracy."),
        evidence=make_evidence(),
        verifier=make_verifier(),
    )

    assert report.is_verified is False
    assert report.uncited_sentences == ("The model achieves 100% accuracy.",)


def test_neutral_claim_fails_verification():
    report = verify_generated_answer(
        answer="CNNs identify retinal lesions [E1].",
        evidence=make_evidence(),
        verifier=make_verifier(label="neutral"),
    )

    assert report.is_verified is False


def test_contradictory_claim_fails_verification():
    report = verify_generated_answer(
        answer="CNNs identify retinal lesions [E1].",
        evidence=make_evidence(),
        verifier=make_verifier(label="contradiction"),
    )

    assert report.is_verified is False


def test_missing_evidence_fails_verification():
    report = verify_generated_answer(
        answer="CNNs identify retinal lesions [E99].",
        evidence=make_evidence(),
        verifier=make_verifier(),
    )

    assert report.is_verified is False
    assert report.assessments[0].label == "missing_evidence"


def test_leading_citation_fails_verification():
    report = verify_generated_answer(
        answer="[E1] CNNs identify retinal lesions.",
        evidence=make_evidence(),
        verifier=make_verifier(),
    )

    assert report.is_verified is False
    assert report.leading_citations == ("[E1]",)


def test_empty_answer_fails_verification():
    report = verify_generated_answer(
        answer="",
        evidence=make_evidence(),
        verifier=make_verifier(),
    )

    assert report.is_verified is False
    assert report.assessments == ()
