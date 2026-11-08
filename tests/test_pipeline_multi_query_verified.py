"""Tests for verified multi-query pipeline integration."""

from unittest.mock import Mock

import pytest

from evidence_rag.generation.rag_pipeline import ResearchRAGPipeline


def test_multi_query_verified_requires_retry_generator():
    pipeline = ResearchRAGPipeline(llm_client=None)

    with pytest.raises(
        RuntimeError,
        match="Verified answering is not configured",
    ):
        pipeline.ask_multi_query_verified(
            question="Compare CNN and ViT.",
            queries=["CNN features", "ViT patches"],
        )


def test_multi_query_verified_rejects_empty_question():
    retry_generator = Mock()

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        retry_generator=retry_generator,
    )

    with pytest.raises(
        ValueError,
        match="Question must not be empty",
    ):
        pipeline.ask_multi_query_verified(
            question="   ",
            queries=["CNN features", "ViT patches"],
        )

    retry_generator.generate.assert_not_called()


def test_multi_query_verified_passes_evidence_to_retry():
    retry_generator = Mock()
    expected_result = object()

    retry_generator.generate.return_value = expected_result

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        retry_generator=retry_generator,
    )

    evidence = [Mock(), Mock()]

    pipeline.retrieve_multi_query_evidence = Mock(
        return_value=evidence,
    )

    question = "How do CNNs differ from Vision Transformers?"

    queries = [
        "CNN retinal feature maps",
        "Vision Transformer patch embeddings",
    ]

    result = pipeline.ask_multi_query_verified(
        question=question,
        queries=queries,
    )

    pipeline.retrieve_multi_query_evidence.assert_called_once_with(
        queries=queries,
    )

    retry_generator.generate.assert_called_once_with(
        question=question,
        evidence=evidence,
    )

    assert result is expected_result


def test_multi_query_verified_propagates_retrieval_errors():
    retry_generator = Mock()

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        retry_generator=retry_generator,
    )

    pipeline.retrieve_multi_query_evidence = Mock(
        side_effect=ValueError("Queries must not be empty"),
    )

    with pytest.raises(
        ValueError,
        match="Queries must not be empty",
    ):
        pipeline.ask_multi_query_verified(
            question="Compare CNN and ViT.",
            queries=[],
        )

    retry_generator.generate.assert_not_called()
