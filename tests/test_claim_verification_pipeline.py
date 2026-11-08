"""Tests for the claim-to-evidence verification pipeline."""

from unittest.mock import Mock

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.claim_verification_pipeline import (
    verify_answer_claims,
)
from evidence_rag.verification.semantic_verifier import ClaimVerificationResult


def make_evidence(evidence_id: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"test:p1:{evidence_id}",
        source="test.pdf",
        page_number=1,
        text=text,
        retrieval_score=1.0,
    )


def test_supported_claim():
    evidence = [make_evidence("E1", "CNNs identify retinal lesions.")]
    verifier = Mock()
    verifier.verify.return_value = ClaimVerificationResult(
        claim="CNNs identify retinal lesions.",
        evidence=evidence[0].text,
        label="entailment",
        confidence=0.98,
    )

    result = verify_answer_claims(
        "CNNs identify retinal lesions [E1].",
        evidence,
        verifier,
    )

    assert len(result) == 1
    assert result[0].citation == "[E1]"
    assert result[0].label == "entailment"


def test_missing_evidence():
    result = verify_answer_claims(
        "CNNs identify lesions [E99].",
        [],
        Mock(),
    )

    assert len(result) == 1
    assert result[0].label == "missing_evidence"


def test_multiple_citations():
    evidence = [
        make_evidence("E1", "CNNs extract image features."),
        make_evidence("E2", "CNNs support retinal classification."),
    ]
    verifier = Mock()
    verifier.verify.return_value = ClaimVerificationResult(
        claim="CNNs extract features and classify images.",
        evidence="Test evidence.",
        label="entailment",
        confidence=0.95,
    )

    result = verify_answer_claims(
        "CNNs extract features and classify images [E1][E2].",
        evidence,
        verifier,
    )

    assert len(result) == 2
    assert {item.citation for item in result} == {"[E1]", "[E2]"}
    assert verifier.verify.call_count == 2


def test_uncited_sentence_is_not_assessed():
    result = verify_answer_claims(
        "CNNs analyse retinal images.",
        [make_evidence("E1", "CNNs analyse retinal images.")],
        Mock(),
    )

    assert result == []
