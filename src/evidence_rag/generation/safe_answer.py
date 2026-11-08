"""Safely present answers based on verification results."""

from dataclasses import dataclass

from evidence_rag.generation.verified_answer import VerifiedAnswerResult

FALLBACK_MESSAGE = (
    "I could not produce an answer that passed the available "
    "evidence verification checks. Please review the retrieved "
    "sources or refine your question."
)


@dataclass(frozen=True)
class SafeAnswer:
    text: str
    status: str
    reasons: tuple[str, ...]


def prepare_safe_answer(
    result: VerifiedAnswerResult,
) -> SafeAnswer:
    """Only expose the generated answer when checks pass."""
    if result.is_accepted:
        return SafeAnswer(
            text=result.answer,
            status="passed_checks",
            reasons=(),
        )

    return SafeAnswer(
        text=FALLBACK_MESSAGE,
        status="insufficient_verification",
        reasons=result.decision.reasons,
    )
