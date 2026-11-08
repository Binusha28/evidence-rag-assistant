"""Regression tests for multi-query reranking candidate pools."""

from unittest.mock import Mock

from evidence_rag.generation.rag_pipeline import RAGConfig, ResearchRAGPipeline
from evidence_rag.ingestion.pipeline import DocumentChunk


def test_multi_query_reranker_receives_larger_candidate_pool():
    reranker = Mock()

    reranker.rerank.return_value = [
        Mock(index=2, score=0.95),
        Mock(index=0, score=0.85),
    ]

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        config=RAGConfig(top_k=2, candidate_k=5),
        reranker=reranker,
    )

    texts = [
        "Retinal CNN feature extraction.",
        "Vision Transformer image patches.",
        "OCT lesion segmentation.",
        "Fundus image preprocessing.",
        "Glaucoma visual field analysis.",
    ]

    pipeline.chunks = [
        DocumentChunk(
            chunk_id=f"research.pdf:p1:c{i}",
            source="research.pdf",
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

    from unittest.mock import patch

    with patch(
        "evidence_rag.generation.rag_pipeline.MultiQueryBM25Retriever",
        return_value=multi_retriever,
    ):
        evidence = pipeline.retrieve_multi_query_evidence(["retinal imaging", "OCT segmentation"])

    assert multi_retriever.search.call_args.kwargs["top_k"] == 5

    reranker.rerank.assert_called_once()

    assert reranker.rerank.call_args.kwargs["top_k"] == 2
    assert len(reranker.rerank.call_args.kwargs["passages"]) == 5

    assert [item.chunk_id for item in evidence] == [
        "research.pdf:p1:c2",
        "research.pdf:p1:c0",
    ]

    assert [item.citation for item in evidence] == [
        "[E1]",
        "[E2]",
    ]
