"""Semantic claim verification using natural language inference."""

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


@dataclass(frozen=True)
class ClaimVerificationResult:
    claim: str
    evidence: str
    label: str
    confidence: float


def normalize_claim_text(text: str) -> str:
    """Normalize whitespace without changing the claim's meaning."""

    normalized = re.sub(r"\s+", " ", text).strip()

    # Remove extraction artifacts before sentence punctuation.
    normalized = re.sub(
        r"\s+([.!?,;:])",
        r"\1",
        normalized,
    )

    return normalized


class SemanticClaimVerifier:
    """Assess whether an evidence passage supports a claim."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model = CrossEncoder(model_name, device=device)
        self.labels = self.model.config.id2label

    def verify(
        self,
        claim: str,
        evidence: str,
    ) -> ClaimVerificationResult:
        normalized_claim = normalize_claim_text(claim)

        if not normalized_claim:
            raise ValueError("Claim must not be empty")

        if not evidence.strip():
            raise ValueError("Evidence must not be empty")

        scores = np.asarray(
            self.model.predict(
                [(evidence, normalized_claim)],
                show_progress_bar=False,
                apply_softmax=True,
            )
        ).reshape(-1)

        label_index = int(np.argmax(scores))
        label = self.labels[label_index].lower()
        confidence = float(scores[label_index])

        return ClaimVerificationResult(
            claim=normalized_claim,
            evidence=evidence,
            label=label,
            confidence=confidence,
        )
