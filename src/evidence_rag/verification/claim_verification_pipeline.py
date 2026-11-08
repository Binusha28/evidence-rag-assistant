"""Connect cited claims to their supporting evidence."""

from dataclasses import dataclass

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.claim_extractor import extract_cited_claims
from evidence_rag.verification.semantic_verifier import (
    SemanticClaimVerifier,
)


@dataclass(frozen=True)
class ClaimEvidenceAssessment:
    claim: str
    citation: str
    label: str
    confidence: float


def verify_answer_claims(
    answer: str,
    evidence: list[Evidence] | tuple[Evidence, ...],
    verifier: SemanticClaimVerifier,
) -> list[ClaimEvidenceAssessment]:
    """Assess each cited claim against its referenced passage."""
    evidence_by_citation = {item.citation: item for item in evidence}

    assessments = []

    for claim in extract_cited_claims(answer):
        for citation in claim.citation_ids:
            item = evidence_by_citation.get(citation)

            if item is None:
                assessments.append(
                    ClaimEvidenceAssessment(
                        claim=claim.text,
                        citation=citation,
                        label="missing_evidence",
                        confidence=0.0,
                    )
                )
                continue

            result = verifier.verify(
                claim=claim.text,
                evidence=item.text,
            )

            assessments.append(
                ClaimEvidenceAssessment(
                    claim=claim.text,
                    citation=citation,
                    label=result.label,
                    confidence=result.confidence,
                )
            )

    return assessments
