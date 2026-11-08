"""Tests for verification decisions."""

import pytest

from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
)
from evidence_rag.verification.claim_verification_pipeline import (
    ClaimEvidenceAssessment,
)
from evidence_rag.verification.verification_decision import (
    decide_verification,
)


def make_report(
    label="entailment",
    uncited_sentences=(),
    leading_citations=(),
):
    assessment = ClaimEvidenceAssessment(
        claim="CNNs identify retinal lesions.",
        citation="[E1]",
        label=label,
        confidence=0.98,
    )

    return AnswerVerificationReport(
        assessments=(assessment,),
        uncited_sentences=uncited_sentences,
        leading_citations=leading_citations,
    )


def test_supported_answer_passes():
    decision = decide_verification(make_report())

    assert decision.status == "passed_checks"
    assert decision.safe_to_present_as_verified is True
    assert decision.reasons == ()


def test_uncited_claim_requires_review():
    decision = decide_verification(
        make_report(uncited_sentences=("The model achieves 100% accuracy.",))
    )

    assert decision.status == "needs_review"
    assert decision.safe_to_present_as_verified is False
    assert any("uncited" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    "label",
    ["neutral", "contradiction", "missing_evidence", "unknown"],
)
def test_non_entailment_requires_review(label):
    decision = decide_verification(make_report(label=label))

    assert decision.status == "needs_review"
    assert decision.safe_to_present_as_verified is False
    assert decision.reasons


def test_leading_citation_requires_review():
    decision = decide_verification(make_report(leading_citations=("[E1]",)))

    assert decision.status == "needs_review"
    assert any("placement" in reason for reason in decision.reasons)


def test_no_assessments_requires_review():
    report = AnswerVerificationReport(
        assessments=(),
        uncited_sentences=(),
        leading_citations=(),
    )

    decision = decide_verification(report)

    assert decision.status == "needs_review"
    assert decision.safe_to_present_as_verified is False
    assert any("No cited claims" in reason for reason in decision.reasons)
