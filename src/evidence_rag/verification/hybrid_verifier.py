"""Compare full-passage and sentence-level NLI predictions."""

from dataclasses import dataclass

from evidence_rag.verification.claim_extractor import split_sentences
from evidence_rag.verification.semantic_verifier import (
    ClaimVerificationResult,
    SemanticClaimVerifier,
)


@dataclass(frozen=True)
class HybridVerificationResult:
    """Diagnostic results for a claim and its cited passage."""

    full_passage: ClaimVerificationResult
    sentence_results: tuple[ClaimVerificationResult, ...]

    @property
    def best_sentence_entailment(self) -> ClaimVerificationResult | None:
        """Return the strongest sentence-level entailment, if any."""
        entailed = [result for result in self.sentence_results if result.label == "entailment"]

        if not entailed:
            return None

        return max(entailed, key=lambda result: result.confidence)

    @property
    def needs_context_review(self) -> bool:
        """Identify disagreement between full-passage and sentence NLI."""
        return self.full_passage.label == "neutral" and self.best_sentence_entailment is not None


class HybridSemanticVerifier:
    """Evaluate claims using full and sentence-level evidence."""

    def __init__(
        self,
        verifier: SemanticClaimVerifier | None = None,
    ) -> None:
        self.verifier = verifier or SemanticClaimVerifier()

    def diagnose(
        self,
        claim: str,
        evidence: str,
    ) -> HybridVerificationResult:
        """Run NLI on the full passage and its individual sentences."""
        full_result = self.verifier.verify(
            claim=claim,
            evidence=evidence,
        )

        sentence_results = tuple(
            self.verifier.verify(
                claim=claim,
                evidence=sentence,
            )
            for sentence in split_sentences(evidence)
            if sentence.strip()
        )

        return HybridVerificationResult(
            full_passage=full_result,
            sentence_results=sentence_results,
        )

    def verify(
        self,
        claim: str,
        evidence: str,
    ) -> ClaimVerificationResult:
        """Preserve the existing conservative verification decision."""
        return self.verifier.verify(
            claim=claim,
            evidence=evidence,
        )
