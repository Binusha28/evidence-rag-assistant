"""Metrics for evaluating retrieval with incomplete relevance judgments."""

from collections.abc import Mapping, Sequence


def judgment_coverage_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int = 5,
) -> float:
    """Fraction of the top-k positions with an explicit relevance judgment."""
    if k <= 0:
        raise ValueError("k must be positive")

    judged_count = sum(chunk_id in relevance for chunk_id in retrieved_ids[:k])
    return judged_count / k


def judged_precision_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int = 5,
) -> float:
    """Fraction of judged top-k passages that are relevant.

    Returns zero when none of the retrieved passages has been judged.
    This is a descriptive metric, not a replacement for Precision@K.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    judged_grades = [relevance[chunk_id] for chunk_id in retrieved_ids[:k] if chunk_id in relevance]

    if not judged_grades:
        return 0.0

    relevant_count = sum(grade >= 1 for grade in judged_grades)
    return relevant_count / len(judged_grades)


def unjudged_chunk_ids(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int = 5,
) -> list[str]:
    """Return top-k chunk IDs missing from the relevance judgments."""
    if k <= 0:
        raise ValueError("k must be positive")

    return [chunk_id for chunk_id in retrieved_ids[:k] if chunk_id not in relevance]
