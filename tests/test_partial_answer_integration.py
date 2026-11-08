"""Integration tests for optional partial-answer recovery."""

from unittest.mock import Mock, patch

from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.generation.safe_answer import FALLBACK_MESSAGE
from evidence_rag.generation.verified_answer import VerifiedAnswerResult
from evidence_rag.verification.answer_verification import AnswerVerificationReport
from evidence_rag.verification.citation_validator import CitationValidationError
from evidence_rag.verification.claim_verification_pipeline import ClaimEvidenceAssessment
from evidence_rag.verification.verification_decision import VerificationDecision

SUPPORTED = "CNNs analyse retinal images [E1]."
UNSUPPORTED = "AI guarantees perfect diagnosis [E2]."


def make_result():
    report = AnswerVerificationReport(
        assessments=(
            ClaimEvidenceAssessment(
                claim="CNNs analyse retinal images.",
                citation="[E1]",
                label="entailment",
                confidence=0.95,
            ),
            ClaimEvidenceAssessment(
                claim="AI guarantees perfect diagnosis.",
                citation="[E2]",
                label="neutral",
                confidence=0.99,
            ),
        ),
        uncited_sentences=(),
        leading_citations=(),
    )

    decision = VerificationDecision(
        status="needs_review",
        reasons=("Unsupported claim.",),
        safe_to_present_as_verified=False,
    )

    return VerifiedAnswerResult(
        answer=f"{SUPPORTED} {UNSUPPORTED}",
        verification=report,
        decision=decision,
    )


def fresh_report(accepted):
    return AnswerVerificationReport(
        assessments=(
            ClaimEvidenceAssessment(
                claim="CNNs analyse retinal images.",
                citation="[E1]",
                label="entailment" if accepted else "neutral",
                confidence=0.95,
            ),
        ),
        uncited_sentences=(),
        leading_citations=(),
    )


def test_partial_recovery_requires_fresh_verification():
    generator = Mock()
    generator.generate.return_value = make_result()

    retry = RetryAnswerGenerator(
        generator,
        max_attempts=2,
        allow_partial_answers=True,
    )

    with patch(
        "evidence_rag.generation.retry_answer.verify_generated_answer",
        return_value=fresh_report(True),
    ) as verify:
        result = retry.generate("What do CNNs do?", [])

    assert result.attempts == 2
    assert result.response.status == "partially_verified"
    assert result.response.text == SUPPORTED
    assert UNSUPPORTED not in result.response.text
    verify.assert_called_once()


def test_failed_fresh_verification_preserves_fallback():
    generator = Mock()
    generator.generate.return_value = make_result()

    retry = RetryAnswerGenerator(
        generator,
        allow_partial_answers=True,
    )

    with patch(
        "evidence_rag.generation.retry_answer.verify_generated_answer",
        return_value=fresh_report(False),
    ):
        result = retry.generate("What do CNNs do?", [])

    assert result.response.status == "insufficient_verification"
    assert result.response.text == FALLBACK_MESSAGE


def test_partial_recovery_disabled_by_default():
    generator = Mock()
    generator.generate.return_value = make_result()

    retry = RetryAnswerGenerator(generator)

    with patch(
        "evidence_rag.generation.retry_answer.verify_generated_answer",
    ) as verify:
        result = retry.generate("What do CNNs do?", [])

    assert result.response.status == "insufficient_verification"
    verify.assert_not_called()


def test_final_citation_error_prevents_recovery():
    generator = Mock()
    generator.generate.side_effect = [
        make_result(),
        CitationValidationError("Invalid citation", "Use valid evidence citations."),
    ]

    retry = RetryAnswerGenerator(
        generator,
        allow_partial_answers=True,
    )

    with patch(
        "evidence_rag.generation.retry_answer.verify_generated_answer",
    ) as verify:
        result = retry.generate("What do CNNs do?", [])

    assert result.response.status == "insufficient_verification"
    assert result.response.text == FALLBACK_MESSAGE
    verify.assert_not_called()
