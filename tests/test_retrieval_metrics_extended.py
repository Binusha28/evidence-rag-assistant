"""Tests for Recall@K and Reciprocal Rank@K."""

import pytest

from evidence_rag.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_all_relevant_retrieved():
    retrieved = ["A", "B", "C"]
    relevance = {"A": 1, "B": 0, "C": 2}

    assert recall_at_k(retrieved, relevance, k=3) == 1.0


def test_recall_partial_retrieval():
    retrieved = ["B", "C", "D", "A", "F"]
    relevance = {"A": 2, "C": 1, "E": 1}

    assert recall_at_k(retrieved, relevance, k=5) == pytest.approx(2 / 3)


def test_recall_no_relevant_chunks():
    assert recall_at_k(["A", "B"], {"A": 0}, k=2) == 0.0


def test_recall_no_relevant_results_retrieved():
    retrieved = ["A", "B"]
    relevance = {"C": 1}

    assert recall_at_k(retrieved, relevance, k=2) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank_at_k(["A", "B"], {"A": 1}, k=2) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank_at_k(["B", "A"], {"A": 1}, k=2) == 0.5


def test_reciprocal_rank_no_relevant_result():
    assert reciprocal_rank_at_k(["A", "B"], {"C": 1}, k=2) == 0.0


def test_reciprocal_rank_respects_k():
    assert reciprocal_rank_at_k(["B", "A"], {"A": 1}, k=1) == 0.0


def test_duplicate_retrieval_does_not_inflate_recall():
    assert (
        recall_at_k(
            ["A", "A", "B"],
            {"A": 1, "B": 1},
            k=3,
        )
        == 1.0
    )


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(["A"], {"A": 1}, k=k)

    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_at_k(["A"], {"A": 1}, k=k)
