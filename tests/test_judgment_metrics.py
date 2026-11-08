"""Tests for retrieval judgment coverage and judged precision."""

import pytest

from evidence_rag.evaluation.judgment_metrics import (
    judged_precision_at_k,
    judgment_coverage_at_k,
    unjudged_chunk_ids,
)


def test_full_judgment_coverage():
    retrieved = ["a", "b", "c"]
    relevance = {"a": 2, "b": 1, "c": 0}

    assert judgment_coverage_at_k(retrieved, relevance, k=3) == 1.0


def test_partial_judgment_coverage():
    retrieved = ["a", "b", "c", "d", "e"]
    relevance = {"a": 2, "c": 0, "e": 1}

    assert judgment_coverage_at_k(retrieved, relevance, k=5) == 0.6


def test_judged_precision():
    retrieved = ["a", "b", "c", "d", "e"]
    relevance = {"a": 2, "c": 0, "e": 1}

    assert judged_precision_at_k(retrieved, relevance, k=5) == pytest.approx(2 / 3)


def test_no_judged_results():
    retrieved = ["a", "b"]
    relevance = {"c": 2}

    assert judgment_coverage_at_k(retrieved, relevance, k=2) == 0.0
    assert judged_precision_at_k(retrieved, relevance, k=2) == 0.0


def test_unjudged_chunk_ids():
    retrieved = ["a", "b", "c", "d"]
    relevance = {"a": 2, "c": 0}

    assert unjudged_chunk_ids(retrieved, relevance, k=4) == ["b", "d"]


@pytest.mark.parametrize(
    "metric",
    [
        judgment_coverage_at_k,
        judged_precision_at_k,
        unjudged_chunk_ids,
    ],
)
def test_invalid_k(metric):
    with pytest.raises(ValueError, match="k must be positive"):
        metric(["a"], {"a": 2}, k=0)


def test_empty_results():
    assert judgment_coverage_at_k([], {}, k=5) == 0.0
    assert judged_precision_at_k([], {}, k=5) == 0.0
    assert unjudged_chunk_ids([], {}, k=5) == []
