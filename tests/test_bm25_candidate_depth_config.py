"""Tests for independently configurable multi-query BM25 depth."""

from unittest.mock import Mock, patch

import pytest

from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)
from evidence_rag.ingestion.pipeline import DocumentChunk


def test_default_bm25_depth_preserves_legacy_behavior():
    config = RAGConfig(top_k=2, candidate_k=5)

    assert config.bm25_candidates_per_query is None


@pytest.mark.parametrize("depth", [0, -1])
def test_invalid_bm25_depth_is_rejected(depth):
    with pytest.raises(ValueError):
        RAGConfig(bm25_candidates_per_query=depth)


def test_custom_bm25_depth_is_independent_of_rerank_pool():
    reranker = Mock()
    reranker.rerank.return_value = [
        Mock(index=0, score=0.9),
        Mock(index=1, score=0.8),
    ]

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        config=RAGConfig(
            top_k=2,
            candidate_k=5,
            bm25_candidates_per_query=3,
        ),
        reranker=reranker,
    )

    texts = [
        "Retinal lesion detection.",
        "Fundus image classification.",
        "CNN feature extraction.",
        "Diabetic retinopathy screening.",
        "Ophthalmic image analysis.",
    ]

    pipeline.chunks = [
        DocumentChunk(
            chunk_id=f"paper.pdf:p1:c{i}",
            source="paper.pdf",
            page_number=1,
            text=text,
            start_word=0,
            end_word=len(text.split()),
        )
        for i, text in enumerate(texts)
    ]

    candidates = [Mock(index=i, text=text, score=float(5 - i)) for i, text in enumerate(texts)]

    multi_retriever = Mock()
    multi_retriever.search.return_value = candidates

    with patch(
        "evidence_rag.generation.rag_pipeline.MultiQueryBM25Retriever",
        return_value=multi_retriever,
    ):
        evidence = pipeline.retrieve_multi_query_evidence(
            ["retinal lesions", "fundus classification"]
        )

    kwargs = multi_retriever.search.call_args.kwargs

    assert kwargs["top_k"] == 5
    assert kwargs["candidates_per_query"] == 3
    assert len(evidence) == 2
    assert len(reranker.rerank.call_args.kwargs["passages"]) == 5
