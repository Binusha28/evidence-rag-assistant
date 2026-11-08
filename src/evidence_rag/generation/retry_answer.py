"""Bounded feedback-guided retry for verified answer generation."""

from dataclasses import dataclass

from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.partial_answer import (
    recover_verified_sentences,
)
from evidence_rag.generation.safe_answer import (
    FALLBACK_MESSAGE,
    SafeAnswer,
    prepare_safe_answer,
)
from evidence_rag.generation.verified_answer import (
    VerifiedAnswerGenerator,
)
from evidence_rag.verification.answer_verification import (
    verify_generated_answer,
)
from evidence_rag.verification.citation_validator import (
    CitationValidationError,
)
from evidence_rag.verification.verification_feedback import (
    build_verification_feedback,
)


@dataclass(frozen=True)
class RetryAnswerResult:
    response: SafeAnswer
    attempts: int


class RetryAnswerGenerator:
    """Retry generation using verification or citation-validation feedback."""

    def __init__(
        self,
        verified_generator: VerifiedAnswerGenerator,
        max_attempts: int = 2,
        allow_partial_answers: bool = False,
    ) -> None:
        if max_attempts < 1 or max_attempts > 2:
            raise ValueError("max_attempts must be 1 or 2")

        self.verified_generator = verified_generator
        self.max_attempts = max_attempts
        self.allow_partial_answers = allow_partial_answers

    def generate(
        self,
        question: str,
        evidence: list[Evidence] | tuple[Evidence, ...],
    ) -> RetryAnswerResult:
        feedback = None
        safe = None
        last_result = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if feedback is None:
                    result = self.verified_generator.generate(
                        question=question,
                        evidence=evidence,
                    )
                else:
                    result = self.verified_generator.generate(
                        question=question,
                        evidence=evidence,
                        feedback=feedback,
                    )

            except CitationValidationError as error:
                feedback = error.feedback
                last_result = None

                safe = SafeAnswer(
                    text=FALLBACK_MESSAGE,
                    status="insufficient_verification",
                    reasons=(str(error),),
                )

                continue

            last_result = result
            safe = prepare_safe_answer(result)

            if result.is_accepted:
                return RetryAnswerResult(
                    response=safe,
                    attempts=attempt,
                )

            if attempt < self.max_attempts:
                feedback = build_verification_feedback(result.verification)

        if self.allow_partial_answers and last_result is not None and not last_result.is_accepted:
            candidate = recover_verified_sentences(
                answer=last_result.answer,
                report=last_result.verification,
            )

            if candidate is not None:
                fresh_report = verify_generated_answer(
                    answer=candidate,
                    evidence=evidence,
                    verifier=self.verified_generator.verifier,
                )

                if fresh_report.is_verified:
                    return RetryAnswerResult(
                        response=SafeAnswer(
                            text=candidate,
                            status="partially_verified",
                            reasons=(
                                "Only independently verified sentences "
                                "were retained from the generated answer.",
                            ),
                        ),
                        attempts=self.max_attempts,
                    )

        return RetryAnswerResult(
            response=safe,
            attempts=self.max_attempts,
        )
