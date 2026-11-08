"""Tests for the verified answer generator."""

from unittest.mock import Mock

import pytest

from evidence_rag.generation.answer_generator import GeneratedAnswer
from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.verification.semantic_verifier import ClaimVerificationResult


@pytest.fixture
def evidence():
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


def make_generator(answer, evidence, label="entailment"):
    answer_generator = Mock()
    answer_generator.generate.return_value = GeneratedAnswer(
        question="How do CNNs detect lesions?",
        answer=answer,
        evidence=evidence,
    )

    verifier = Mock()
    verifier.verify.return_value = ClaimVerificationResult(
        claim="CNNs identify retinal lesions.",
        evidence=evidence[0].text,
        label=label,
        confidence=0.98,
    )

    return VerifiedAnswerGenerator(
        answer_generator=answer_generator,
        verifier=verifier,
    )


def test_supported_answer_is_accepted(evidence):
    generator = make_generator(
        "CNNs identify retinal lesions [E1].",
        evidence,
    )

    result = generator.generate(
        "How do CNNs detect lesions?",
        evidence,
    )

    assert result.is_accepted is True
    assert result.decision.status == "passed_checks"
    assert len(result.verification.assessments) == 1


def test_uncited_claim_is_rejected(evidence):
    generator = make_generator(
        "CNNs identify retinal lesions [E1]. The model achieves 100% accuracy.",
        evidence,
    )

    result = generator.generate(
        "How do CNNs detect lesions?",
        evidence,
    )

    assert result.is_accepted is False
    assert result.decision.status == "needs_review"
    assert result.verification.uncited_sentences


@pytest.mark.parametrize("label", ["neutral", "contradiction"])
def test_unsupported_claim_is_rejected(evidence, label):
    generator = make_generator(
        "CNNs identify retinal lesions [E1].",
        evidence,
        label=label,
    )

    result = generator.generate(
        "How do CNNs detect lesions?",
        evidence,
    )

    assert result.is_accepted is False
    assert result.verification.assessments[0].label == label
