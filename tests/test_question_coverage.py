"""Unit tests for research-question coverage assessment."""

from unittest.mock import Mock

import pytest

from evidence_rag.verification.question_coverage import (
    QuestionCoverageChecker,
)

QUESTION = "What are the two main components of the Transformer architecture?"


@pytest.mark.parametrize(
    ("model_response", "expected_status"),
    [
        ("COMPLETE", "complete"),
        ("INCOMPLETE", "incomplete"),
        ("UNCERTAIN", "uncertain"),
        ("The answer is complete.", "uncertain"),
    ],
)
def test_coverage_classification(model_response, expected_status):
    llm = Mock()
    llm.generate.return_value = model_response

    checker = QuestionCoverageChecker(llm)

    result = checker.check(
        QUESTION,
        "The Transformer has an encoder and a decoder [E1].",
    )

    assert result.status == expected_status
    assert result.is_complete == (expected_status == "complete")
    llm.generate.assert_called_once()


def test_empty_answer_is_uncertain_without_model_call():
    llm = Mock()
    checker = QuestionCoverageChecker(llm)

    result = checker.check(QUESTION, "")

    assert result.status == "uncertain"
    assert not result.is_complete
    llm.generate.assert_not_called()


def test_model_failure_is_uncertain():
    llm = Mock()
    llm.generate.side_effect = RuntimeError("Ollama unavailable")

    checker = QuestionCoverageChecker(llm)

    result = checker.check(QUESTION, "The encoder has six layers [E1].")

    assert result.status == "uncertain"
    assert not result.is_complete


def test_checker_includes_question_and_answer_in_prompt():
    llm = Mock()
    llm.generate.return_value = "INCOMPLETE"

    checker = QuestionCoverageChecker(llm)

    answer = "The encoder has six layers [E1]."
    result = checker.check(QUESTION, answer)

    prompt = llm.generate.call_args.kwargs["prompt"]

    assert QUESTION in prompt
    assert answer in prompt
    assert result.status == "incomplete"
