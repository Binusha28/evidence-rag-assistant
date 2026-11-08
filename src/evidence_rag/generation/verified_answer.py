"""Generate answers with a conservative verification decision."""

from dataclasses import dataclass

from evidence_rag.generation.answer_generator import EvidenceAnswerGenerator
from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
    verify_generated_answer,
)
from evidence_rag.verification.citation_repair import (
    SemanticCitationRepair,
)
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier
from evidence_rag.verification.verification_decision import (
    VerificationDecision,
    decide_verification,
)


@dataclass(frozen=True)
class VerifiedAnswerResult:
    """Generated answer and its evidence-verification outcome."""

    answer: str
    verification: AnswerVerificationReport
    decision: VerificationDecision

    @property
    def is_accepted(self) -> bool:
        """Whether the answer passed the available verification checks."""
        return self.decision.safe_to_present_as_verified


class VerifiedAnswerGenerator:
    """Wrap an existing answer generator with evidence verification."""

    def __init__(
        self,
        answer_generator: EvidenceAnswerGenerator,
        verifier: SemanticClaimVerifier,
        citation_repair: SemanticCitationRepair | None = None,
    ) -> None:
        self.answer_generator = answer_generator
        self.verifier = verifier
        self.citation_repair = citation_repair

    def generate(
        self,
        question: str,
        evidence: list[Evidence] | tuple[Evidence, ...],
        feedback: str | None = None,
    ) -> VerifiedAnswerResult:
        """Generate, optionally repair citations, and verify the answer."""

        if feedback is None:
            generated = self.answer_generator.generate(
                question=question,
                evidence=evidence,
            )
        else:
            generated = self.answer_generator.generate(
                question=question,
                evidence=evidence,
                feedback=feedback,
            )

        answer = generated.answer

        # Citation repair is optional and disabled by default.
        # The repairer only adds citations to claims predicted
        # as entailed by the supplied evidence.
        if self.citation_repair is not None and generated.evidence:
            repair_result = self.citation_repair.repair(
                answer=answer,
                evidence=generated.evidence,
            )
            answer = repair_result.repaired_answer

        # Always verify the final answer, including any repaired claims.
        report = verify_generated_answer(
            answer=answer,
            evidence=generated.evidence,
            verifier=self.verifier,
        )

        decision = decide_verification(report)

        return VerifiedAnswerResult(
            answer=answer,
            verification=report,
            decision=decision,
        )
