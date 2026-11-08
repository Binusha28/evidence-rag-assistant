"""Tests for hybrid semantic verification diagnostics."""

from evidence_rag.verification.hybrid_verifier import (
    HybridSemanticVerifier,
)
from evidence_rag.verification.semantic_verifier import (
    ClaimVerificationResult,
)


class FakeVerifier:
    """Deterministic NLI substitute for fast unit tests."""

    def verify(self, claim: str, evidence: str) -> ClaimVerificationResult:
        label = "entailment" if evidence == "The image produces a feature map." else "neutral"

        return ClaimVerificationResult(
            claim=claim,
            evidence=evidence,
            label=label,
            confidence=0.95,
        )


def test_hybrid_diagnosis_finds_sentence_entailment():
    verifier = HybridSemanticVerifier(verifier=FakeVerifier())

    result = verifier.diagnose(
        claim="The image produces a feature map.",
        evidence=("The image produces a feature map. The model estimates lesion regions."),
    )

    assert result.full_passage.label == "neutral"
    assert result.best_sentence_entailment is not None
    assert result.best_sentence_entailment.label == "entailment"


def test_hybrid_verify_preserves_original_decision():
    verifier = HybridSemanticVerifier(verifier=FakeVerifier())

    result = verifier.verify(
        claim="The image produces a feature map.",
        evidence=("The image produces a feature map. The model estimates lesion regions."),
    )

    assert result.label == "neutral"


def test_hybrid_diagnosis_without_entailment():
    verifier = HybridSemanticVerifier(verifier=FakeVerifier())

    result = verifier.diagnose(
        claim="The model achieves perfect accuracy.",
        evidence="The model estimates lesion regions.",
    )

    assert result.best_sentence_entailment is None


def test_context_review_detects_disagreement():
    verifier = HybridSemanticVerifier(verifier=FakeVerifier())

    result = verifier.diagnose(
        claim="The image produces a feature map.",
        evidence=("The image produces a feature map. The model estimates lesion regions."),
    )

    assert result.needs_context_review is True


def test_context_review_not_triggered_without_entailment():
    verifier = HybridSemanticVerifier(verifier=FakeVerifier())

    result = verifier.diagnose(
        claim="The model achieves perfect accuracy.",
        evidence="The model estimates lesion regions.",
    )

    assert result.needs_context_review is False
