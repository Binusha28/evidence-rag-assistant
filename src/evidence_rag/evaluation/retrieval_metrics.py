import math
from collections.abc import Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
) -> float:
    """Calculate Precision@K using binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_results = retrieved_ids[:k]

    relevant_count = sum(relevance.get(chunk_id, 0) >= 1 for chunk_id in top_results)

    return relevant_count / k


def dcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
) -> float:
    """Calculate Discounted Cumulative Gain."""
    if k <= 0:
        raise ValueError("k must be positive")

    score = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        grade = relevance.get(chunk_id, 0)
        gain = (2**grade - 1) / math.log2(rank + 1)
        score += gain

    return score


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if k <= 0:
        raise ValueError("k must be positive")

    actual_dcg = dcg_at_k(retrieved_ids, relevance, k)

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]

    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1)
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
) -> float:
    """Fraction of known relevant chunks retrieved within top K.

    Assumes the relevance mapping contains the complete set
    of relevant chunks for the query.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    relevant_ids = {chunk_id for chunk_id, grade in relevance.items() if grade >= 1}

    if not relevant_ids:
        return 0.0

    retrieved_relevant = set(retrieved_ids[:k]) & relevant_ids

    return len(retrieved_relevant) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
) -> float:
    """Reciprocal rank of the first relevant chunk within top K."""
    if k <= 0:
        raise ValueError("k must be positive")

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if relevance.get(chunk_id, 0) >= 1:
            return 1.0 / rank

    return 0.0
