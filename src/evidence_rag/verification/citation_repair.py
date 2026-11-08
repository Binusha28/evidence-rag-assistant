"""Conservative citation repair using semantic entailment."""

import re
from dataclasses import dataclass

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.claim_extractor import split_sentences
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier


@dataclass(frozen=True)
class CitationRepairResult:
    """Result of attempting to repair missing evidence citations."""

    original_answer: str
    repaired_answer: str
    repaired_sentences: tuple[str, ...]
    unrepaired_sentences: tuple[str, ...]

    @property
    def changed(self) -> bool:
        """Return True if citation repair modified the answer."""
        return self.original_answer != self.repaired_answer


class SemanticCitationRepair:
    """Attach citations only when evidence semantically entails a claim."""

    def __init__(
        self,
        verifier: SemanticClaimVerifier,
        min_confidence: float = 0.95,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        self.verifier = verifier
        self.min_confidence = min_confidence

    def repair(
        self,
        answer: str,
        evidence: list[Evidence] | tuple[Evidence, ...],
    ) -> CitationRepairResult:
        """Attempt to add citations to independently supported sentences."""

        if not answer.strip() or not evidence:
            return CitationRepairResult(
                original_answer=answer,
                repaired_answer=answer,
                repaired_sentences=(),
                unrepaired_sentences=(),
            )

        sentences = split_sentences(answer)

        repaired_sentences: list[str] = []
        unrepaired_sentences: list[str] = []
        output_sentences: list[str] = []

        for sentence in sentences:
            # Preserve all existing citation identifiers, including
            # invalid identifiers. The normal citation validator
            # is responsible for rejecting unknown citations.
            if re.search(r"\[E\d+\]", sentence):
                output_sentences.append(sentence)
                continue

            best_evidence = None
            best_confidence = 0.0

            # Evaluate the uncited claim against every evidence passage.
            for item in evidence:
                result = self.verifier.verify(
                    claim=sentence,
                    evidence=item.text,
                )

                # Only consider evidence classified as entailment.
                if (
                    result.label == "entailment"
                    and result.confidence >= self.min_confidence
                    and result.confidence > best_confidence
                ):
                    best_evidence = item
                    best_confidence = result.confidence

            if best_evidence is None:
                # Do not fabricate support for unsupported claims.
                output_sentences.append(sentence)
                unrepaired_sentences.append(sentence)
                continue

            # Insert the citation before terminal punctuation.
            if sentence.endswith((".", "!", "?")):
                repaired = f"{sentence[:-1].rstrip()} {best_evidence.citation}{sentence[-1]}"
            else:
                repaired = f"{sentence.rstrip()} {best_evidence.citation}"

            output_sentences.append(repaired)
            repaired_sentences.append(repaired)

        return CitationRepairResult(
            original_answer=answer,
            repaired_answer=" ".join(output_sentences),
            repaired_sentences=tuple(repaired_sentences),
            unrepaired_sentences=tuple(unrepaired_sentences),
        )
