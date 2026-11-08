"""Tests for multi-query retrieval and rank fusion."""

import pytest

from evidence_rag.retrieval.bm25_search import (
    BM25Retriever,
    BM25SearchResult,
)
from evidence_rag.retrieval.multi_query_search import (
    MultiQueryBM25Retriever,
    reciprocal_rank_fusion,
)


def test_rrf_combines_repeated_passages():
    first = [
        BM25SearchResult("CNN passage", 5.0, 0),
        BM25SearchResult("ViT passage", 3.0, 1),
    ]

    second = [
        BM25SearchResult("ViT passage", 8.0, 1),
        BM25SearchResult("Other passage", 2.0, 2),
    ]

    results = reciprocal_rank_fusion(
        [first, second],
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].index == 1
    assert len({item.index for item in results}) == 3


def test_rrf_preserves_original_indices():
    ranked = [
        [
            BM25SearchResult("First", 2.0, 7),
            BM25SearchResult("Second", 1.0, 12),
        ]
    ]

    results = reciprocal_rank_fusion(
        ranked,
        top_k=2,
    )

    assert [item.index for item in results] == [7, 12]


def test_multi_query_search_returns_relevant_passages():
    retriever = BM25Retriever()

    retriever.index(
        [
            "CNN feature maps support retinal lesion detection.",
            "Vision Transformers use image patches and embeddings.",
            "Transformer self attention processes image tokens.",
            "Unrelated passage about financial accounting.",
        ]
    )

    multi = MultiQueryBM25Retriever(retriever)

    results = multi.search(
        queries=[
            "CNN retinal lesion feature maps",
            "Vision Transformer image patches",
        ],
        top_k=3,
        candidates_per_query=3,
    )

    assert results
    assert len({item.index for item in results}) == len(results)
    assert all(0 <= item.index < 4 for item in results)


@pytest.mark.parametrize(
    "queries",
    [
        [],
        [""],
        ["CNN", "  "],
    ],
)
def test_multi_query_rejects_invalid_queries(queries):
    retriever = BM25Retriever()
    multi = MultiQueryBM25Retriever(retriever)

    with pytest.raises(ValueError):
        multi.search(queries)


def test_rrf_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)


def test_rrf_rejects_invalid_rrf_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=0)
