import pytest

from evidence_rag.evaluation.retrieval_metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
)


def test_precision_at_k() -> None:
    retrieved = ["A", "B", "C", "D", "E"]
    relevance = {"A": 2, "B": 0, "C": 1, "D": 0, "E": 2}

    assert precision_at_k(retrieved, relevance, k=5) == 0.6


def test_precision_no_relevant_results() -> None:
    retrieved = ["A", "B", "C"]
    relevance = {"A": 0, "B": 0, "C": 0}

    assert precision_at_k(retrieved, relevance, k=3) == 0.0


def test_precision_missing_judgments() -> None:
    retrieved = ["A", "B"]
    relevance = {"A": 2}

    assert precision_at_k(retrieved, relevance, k=2) == 0.5


def test_precision_fewer_results_than_k() -> None:
    retrieved = ["A"]
    relevance = {"A": 2}

    assert precision_at_k(retrieved, relevance, k=5) == 0.2


def test_dcg_perfect_order() -> None:
    retrieved = ["A", "B", "C"]
    relevance = {"A": 2, "B": 1, "C": 0}

    assert dcg_at_k(retrieved, relevance, k=3) == pytest.approx(3 + 1 / 1.584962500721156)


def test_ndcg_perfect_ranking() -> None:
    retrieved = ["A", "B", "C"]
    relevance = {"A": 2, "B": 1, "C": 0}

    assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_imperfect_ranking() -> None:
    retrieved = ["B", "A", "C"]
    relevance = {"A": 2, "B": 1, "C": 0}

    score = ndcg_at_k(retrieved, relevance, k=3)

    assert 0 < score < 1


def test_ndcg_no_relevant_documents() -> None:
    retrieved = ["A", "B"]
    relevance = {"A": 0, "B": 0}

    assert ndcg_at_k(retrieved, relevance, k=2) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["A"], {"A": 1}, k=k)

    with pytest.raises(ValueError, match="k must be positive"):
        dcg_at_k(["A"], {"A": 1}, k=k)

    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k(["A"], {"A": 1}, k=k)
