"""Regression tests for NLI claim normalization."""

from unittest.mock import Mock

import numpy as np

from evidence_rag.verification.semantic_verifier import (
    SemanticClaimVerifier,
    normalize_claim_text,
)


def test_normalize_space_before_period():
    assert (
        normalize_claim_text("Vision Transformers outperform ResNets .")
        == "Vision Transformers outperform ResNets."
    )


def test_normalize_multiple_spaces_and_newlines():
    assert normalize_claim_text("CNNs  generate\n feature maps .") == "CNNs generate feature maps."


def test_normalize_preserves_meaningful_negation():
    assert (
        normalize_claim_text("The evidence does not support the claim .")
        == "The evidence does not support the claim."
    )


def test_verifier_receives_normalized_claim():
    verifier = SemanticClaimVerifier.__new__(SemanticClaimVerifier)

    verifier.model = Mock()
    verifier.model.predict.return_value = np.array([[0.01, 0.98, 0.01]])
    verifier.labels = {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }

    result = verifier.verify(
        claim="Vision Transformers outperform ResNets .",
        evidence="Vision Transformers outperform ResNets.",
    )

    verifier.model.predict.assert_called_once_with(
        [
            (
                "Vision Transformers outperform ResNets.",
                "Vision Transformers outperform ResNets.",
            )
        ],
        show_progress_bar=False,
        apply_softmax=True,
    )

    assert result.label == "entailment"
    assert result.claim == "Vision Transformers outperform ResNets."
