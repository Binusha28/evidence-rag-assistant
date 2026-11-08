"""Regression tests for feedback on compound scientific claims."""

from evidence_rag.verification.answer_verification import AnswerVerificationReport
from evidence_rag.verification.claim_verification_pipeline import ClaimEvidenceAssessment
from evidence_rag.verification.verification_feedback import build_verification_feedback


def make_report(assessments):
    return AnswerVerificationReport(
        assessments=tuple(assessments),
        uncited_sentences=(),
        leading_citations=(),
    )


def assessment(claim, citation, label):
    return ClaimEvidenceAssessment(
        claim=claim,
        citation=citation,
        label=label,
        confidence=0.9,
    )


def test_compound_claim_receives_splitting_feedback():
    claim = "AI uses OCT, supports detection, and assists diagnosis."

    report = make_report(
        [
            assessment(claim, "[E1]", "entailment"),
            assessment(claim, "[E2]", "neutral"),
            assessment(claim, "[E3]", "neutral"),
        ]
    )

    feedback = build_verification_feedback(report)

    assert "compound statements" in feedback
    assert "separate, self-contained factual sentences" in feedback
    assert "omit unsupported claims" in feedback


def test_single_citation_keeps_standard_feedback():
    report = make_report(
        [
            assessment("AI improves screening.", "[E1]", "neutral"),
        ]
    )

    feedback = build_verification_feedback(report)

    assert "Claim requiring revision" in feedback
    assert "compound statements" not in feedback


def test_fully_supported_multi_citation_claim_not_flagged():
    claim = "The model processes retinal images."

    report = make_report(
        [
            assessment(claim, "[E1]", "entailment"),
            assessment(claim, "[E2]", "entailment"),
        ]
    )

    feedback = build_verification_feedback(report)

    assert "compound statements" not in feedback
