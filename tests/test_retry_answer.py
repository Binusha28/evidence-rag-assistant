"""Tests for bounded answer regeneration."""

from unittest.mock import Mock

import pytest

from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.generation.safe_answer import FALLBACK_MESSAGE
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


def test_first_attempt_success():
    generator = Mock()
    generator.generate.return_value = make_result(True)

    retry = RetryAnswerGenerator(generator)
    result = retry.generate("How do CNNs work?", [])

    assert result.attempts == 1
    assert result.response.status == "passed_checks"
    assert generator.generate.call_count == 1


def test_second_attempt_success():
    generator = Mock()
    generator.generate.side_effect = [
        make_result(False),
        make_result(True),
    ]

    retry = RetryAnswerGenerator(generator)
    result = retry.generate("How do CNNs work?", [])

    assert result.attempts == 2
    assert result.response.status == "passed_checks"
    assert generator.generate.call_count == 2


def test_both_attempts_fail():
    generator = Mock()
    generator.generate.return_value = make_result(False)

    retry = RetryAnswerGenerator(generator)
    result = retry.generate("How do CNNs work?", [])

    assert result.attempts == 2
    assert result.response.status == "insufficient_verification"
    assert result.response.text == FALLBACK_MESSAGE
    assert generator.generate.call_count == 2


def test_single_attempt_limit():
    generator = Mock()
    generator.generate.return_value = make_result(False)

    retry = RetryAnswerGenerator(
        generator,
        max_attempts=1,
    )

    result = retry.generate("How do CNNs work?", [])

    assert result.attempts == 1
    assert generator.generate.call_count == 1
    assert result.response.text == FALLBACK_MESSAGE


@pytest.mark.parametrize("attempts", [0, -1, 3, 10])
def test_invalid_attempt_limit(attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        RetryAnswerGenerator(
            verified_generator=Mock(),
            max_attempts=attempts,
        )
