"""Test retry handling for invalid and missing citations."""

from unittest.mock import Mock

from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.verification.citation_validator import (
    CitationValidationError,
)


def citation_error(message: str) -> CitationValidationError:
    return CitationValidationError(
        message=message,
        feedback="Use only valid evidence citations.",
    )


def test_invalid_citation_triggers_retry():
    generator = Mock()
    accepted = Mock()
    accepted.is_accepted = True
    accepted.answer = "Supported statement [E1]."

    generator.generate.side_effect = [
        citation_error("Unknown citation: [E9]"),
        accepted,
    ]

    retry = RetryAnswerGenerator(
        verified_generator=generator,
        max_attempts=2,
    )

    result = retry.generate(
        question="What does the evidence show?",
        evidence=[],
    )

    assert result.attempts == 2
    assert result.response.status == "passed_checks"
    assert generator.generate.call_args.kwargs["feedback"] == ("Use only valid evidence citations.")


def test_repeated_invalid_citations_return_fallback():
    generator = Mock()
    generator.generate.side_effect = [
        citation_error("Unknown citation: [E9]"),
        citation_error("Unknown citation: [E8]"),
    ]

    retry = RetryAnswerGenerator(
        verified_generator=generator,
        max_attempts=2,
    )

    result = retry.generate(
        question="What does the evidence show?",
        evidence=[],
    )

    assert result.attempts == 2
    assert result.response.status == "insufficient_verification"
    assert "Unknown citation: [E8]" in result.response.reasons


def test_missing_citations_trigger_retry():
    generator = Mock()
    generator.generate.side_effect = [
        citation_error("No evidence citations"),
        citation_error("No evidence citations"),
    ]

    retry = RetryAnswerGenerator(
        verified_generator=generator,
        max_attempts=2,
    )

    result = retry.generate(
        question="Summarise the research.",
        evidence=[],
    )

    assert result.attempts == 2
    assert result.response.status == "insufficient_verification"
