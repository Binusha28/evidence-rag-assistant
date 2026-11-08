"""Tests for semantic claim verification."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evidence_rag.verification.semantic_verifier import (
    SemanticClaimVerifier,
)


@pytest.fixture
def verifier():
    with patch("evidence_rag.verification.semantic_verifier.CrossEncoder") as mock_encoder:
        model = MagicMock()
        model.config.id2label = {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
        }
        mock_encoder.return_value = model

        instance = SemanticClaimVerifier()
        yield instance


def test_entailment_prediction(verifier):
    verifier.model.predict.return_value = np.array([[0.01, 0.98, 0.01]])

    result = verifier.verify(
        claim="CNNs identify retinal lesions.",
        evidence="CNNs identify retinal lesions in fundus images.",
    )

    assert result.label == "entailment"
    assert result.confidence == pytest.approx(0.98)


def test_neutral_prediction(verifier):
    verifier.model.predict.return_value = np.array([[0.01, 0.01, 0.98]])

    result = verifier.verify(
        claim="The model achieves 99% accuracy.",
        evidence="The model detects retinal lesions.",
    )

    assert result.label == "neutral"


def test_contradiction_prediction(verifier):
    verifier.model.predict.return_value = np.array([[0.98, 0.01, 0.01]])

    result = verifier.verify(
        claim="CNNs cannot identify lesions.",
        evidence="CNNs identify retinal lesions.",
    )

    assert result.label == "contradiction"


def test_empty_claim_rejected(verifier):
    with pytest.raises(ValueError, match="Claim must not be empty"):
        verifier.verify(
            claim=" ",
            evidence="Valid evidence.",
        )


def test_empty_evidence_rejected(verifier):
    with pytest.raises(ValueError, match="Evidence must not be empty"):
        verifier.verify(
            claim="Valid claim.",
            evidence=" ",
        )
