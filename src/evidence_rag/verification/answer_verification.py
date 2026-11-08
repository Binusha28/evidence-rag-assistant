"""Produce a combined verification report for a generated answer."""

from dataclasses import dataclass

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.citation_placement import (
    check_citation_placement,
)
from evidence_rag.verification.claim_extractor import (
    extract_uncited_sentences,
)
from evidence_rag.verification.claim_verification_pipeline import (
    ClaimEvidenceAssessment,
    verify_answer_claims,
)
from evidence_rag.verification.semantic_verifier import (
    SemanticClaimVerifier,
)


@dataclass(frozen=True)
class AnswerVerificationReport:
    assessments: tuple[ClaimEvidenceAssessment, ...]
    uncited_sentences: tuple[str, ...]
    leading_citations: tuple[str, ...]

    @property
    def is_verified(self) -> bool:
        """Conservative pass/fail result for the current checks."""
        return (
            bool(self.assessments)
            and all(item.label == "entailment" for item in self.assessments)
            and not self.uncited_sentences
            and not self.leading_citations
        )


def verify_generated_answer(
    answer: str,
    evidence: list[Evidence] | tuple[Evidence, ...],
    verifier: SemanticClaimVerifier,
) -> AnswerVerificationReport:
    """Run citation placement, coverage and semantic checks."""
    placement = check_citation_placement(answer)

    assessments = verify_answer_claims(
        answer=answer,
        evidence=evidence,
        verifier=verifier,
    )

    uncited = extract_uncited_sentences(answer)

    return AnswerVerificationReport(
        assessments=tuple(assessments),
        uncited_sentences=tuple(uncited),
        leading_citations=placement.leading_citations,
    )
