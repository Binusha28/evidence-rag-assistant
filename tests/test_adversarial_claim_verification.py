"""Adversarial tests for citation-aware claim verification."""

from unittest.mock import Mock

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.claim_extractor import (
    extract_cited_claims,
    extract_uncited_sentences,
)
from evidence_rag.verification.claim_verification_pipeline import (
    verify_answer_claims,
)
from evidence_rag.verification.semantic_verifier import (
    ClaimVerificationResult,
)


def make_evidence(evidence_id: str, text: str) -> Evidence:
    """Create evidence without loading real ML models."""
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"test:p1:{evidence_id}",
        source="test.pdf",
        page_number=1,
        text=text,
        retrieval_score=1.0,
    )


def make_verifier(labels: dict[str, str]) -> Mock:
    """Return controlled NLI labels for specific claims."""

    verifier = Mock()

    def verify(claim: str, evidence: str) -> ClaimVerificationResult:
        label = labels.get(claim, "neutral")

        return ClaimVerificationResult(
            claim=claim,
            evidence=evidence,
            label=label,
            confidence=0.99,
        )

    verifier.verify.side_effect = verify
    return verifier


def test_independent_comparison_claims_are_verified_separately():
    answer = "CNNs generate feature maps [E1], while ViTs use image patches [E2]."
    evidence = [
        make_evidence("E1", "CNNs generate feature maps."),
        make_evidence("E2", "ViTs use image patches."),
    ]
    verifier = make_verifier(
        {
            "CNNs generate feature maps": "entailment",
            "ViTs use image patches.": "entailment",
        }
    )

    results = verify_answer_claims(answer, evidence, verifier)

    assert len(results) == 2
    assert [item.citation for item in results] == [
        "[E1]",
        "[E2]",
    ]
    assert all(item.label == "entailment" for item in results)

    verifier.verify.assert_any_call(
        claim="CNNs generate feature maps",
        evidence="CNNs generate feature maps.",
    )
    verifier.verify.assert_any_call(
        claim="ViTs use image patches.",
        evidence="ViTs use image patches.",
    )


def test_unsupported_comparison_claim_is_not_accepted():
    answer = "CNNs generate feature maps [E1], while ViTs always achieve perfect accuracy [E2]."
    evidence = [
        make_evidence("E1", "CNNs generate feature maps."),
        make_evidence("E2", "ViTs use image patches."),
    ]
    verifier = make_verifier(
        {
            "CNNs generate feature maps": "entailment",
            "ViTs always achieve perfect accuracy.": "neutral",
        }
    )

    results = verify_answer_claims(answer, evidence, verifier)

    assert len(results) == 2
    assert [item.label for item in results] == [
        "entailment",
        "neutral",
    ]


def test_negation_is_preserved_in_extracted_claim():
    answer = "The study did not report perfect accuracy [E1]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].text == ("The study did not report perfect accuracy.")
    assert claims[0].citation_ids == ("[E1]",)


def test_contradictory_claim_is_reported():
    answer = "The model achieves perfect accuracy [E1]."
    evidence = [
        make_evidence(
            "E1",
            "The model achieves an accuracy of 85%.",
        )
    ]
    verifier = make_verifier({"The model achieves perfect accuracy.": "contradiction"})

    results = verify_answer_claims(answer, evidence, verifier)

    assert len(results) == 1
    assert results[0].label == "contradiction"


def test_shared_citations_are_preserved():
    answer = "Both papers discuss image classification [E1][E2]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]", "[E2]")


def test_uncited_sentence_is_detected():
    answer = "CNNs generate feature maps [E1]. The model achieves perfect accuracy."

    uncited = extract_uncited_sentences(answer)

    assert uncited == ["The model achieves perfect accuracy."]


def test_unknown_citation_is_reported_as_missing_evidence():
    answer = "CNNs generate feature maps [E99]."

    results = verify_answer_claims(
        answer,
        [make_evidence("E1", "CNNs generate feature maps.")],
        Mock(),
    )

    assert len(results) == 1
    assert results[0].citation == "[E99]"
    assert results[0].label == "missing_evidence"


def test_ambiguous_comparison_is_not_split():
    answer = "CNNs generate feature maps [E1], and ViTs use patches [E2]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]", "[E2]")


def test_repeated_citation_is_deduplicated():
    answer = "CNNs generate feature maps [E1][E1]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]",)


def test_comparison_claims_keep_correct_evidence_mapping():
    answer = "CNNs generate feature maps [E1], whereas ViTs use patch embeddings [E2]."
    evidence = [
        make_evidence("E1", "CNNs generate feature maps."),
        make_evidence("E2", "ViTs use patch embeddings."),
    ]
    verifier = make_verifier(
        {
            "CNNs generate feature maps": "entailment",
            "ViTs use patch embeddings.": "entailment",
        }
    )

    results = verify_answer_claims(answer, evidence, verifier)

    assert len(results) == 2

    verifier.verify.assert_any_call(
        claim="CNNs generate feature maps",
        evidence="CNNs generate feature maps.",
    )
    verifier.verify.assert_any_call(
        claim="ViTs use patch embeddings.",
        evidence="ViTs use patch embeddings.",
    )
