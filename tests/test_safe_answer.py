"""Tests for safe answer presentation."""

from evidence_rag.generation.safe_answer import (
    FALLBACK_MESSAGE,
    prepare_safe_answer,
)
from evidence_rag.generation.verified_answer import VerifiedAnswerResult
from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
)
from evidence_rag.verification.verification_decision import (
    VerificationDecision,
)


def make_result(accepted: bool) -> VerifiedAnswerResult:
    report = AnswerVerificationReport(
        assessments=(),
        uncited_sentences=(),
        leading_citations=(),
    )

    decision = VerificationDecision(
        status="passed_checks" if accepted else "needs_review",
        reasons=() if accepted else ("Unsupported claim.",),
        safe_to_present_as_verified=accepted,
    )

    return VerifiedAnswerResult(
        answer="CNNs identify retinal lesions [E1].",
        verification=report,
        decision=decision,
    )


def test_accepted_answer_is_displayed():
    result = prepare_safe_answer(make_result(accepted=True))

    assert result.status == "passed_checks"
    assert result.text == "CNNs identify retinal lesions [E1]."
    assert result.reasons == ()


def test_rejected_answer_uses_fallback():
    result = prepare_safe_answer(make_result(accepted=False))

    assert result.status == "insufficient_verification"
    assert result.text == FALLBACK_MESSAGE
    assert result.reasons == ("Unsupported claim.",)


def test_rejected_answer_does_not_expose_generated_text():
    original = make_result(accepted=False)
    result = prepare_safe_answer(original)

    assert result.text != original.answer
