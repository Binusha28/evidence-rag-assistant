"""Integration tests for citation-validation retry handling."""

from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.generation.safe_answer import SafeAnswer
from evidence_rag.verification.citation_validator import CitationValidationError


class FakeVerifiedGenerator:
    """Simulate a citation failure followed by a verified response."""

    def __init__(self):
        self.calls = []

    def generate(self, question, evidence, feedback=None):
        self.calls.append(feedback)

        if len(self.calls) == 1:
            raise CitationValidationError(
                message="The generated answer has no citations.",
                feedback="Add a citation to every factual sentence.",
            )

        return FakeVerifiedResult()


class FakeVerifiedResult:
    """Simulate a successful verification result."""

    is_accepted = True


def test_retry_after_citation_error(monkeypatch):
    generator = FakeVerifiedGenerator()

    expected_response = SafeAnswer(
        text="A CNN generates a feature map [E1].",
        status="verified",
        reasons=(),
    )

    monkeypatch.setattr(
        "evidence_rag.generation.retry_answer.prepare_safe_answer",
        lambda result: expected_response,
    )

    retry = RetryAnswerGenerator(
        verified_generator=generator,
        max_attempts=2,
    )

    result = retry.generate(
        question="How does a CNN process fundus images?",
        evidence=[
            Evidence(
                evidence_id="E1",
                chunk_id="test-chunk",
                source="research.pdf",
                page_number=3,
                text="A CNN generates a feature map.",
                retrieval_score=1.0,
            )
        ],
    )

    assert result.attempts == 2
    assert result.response == expected_response
    assert generator.calls == [
        None,
        "Add a citation to every factual sentence.",
    ]
