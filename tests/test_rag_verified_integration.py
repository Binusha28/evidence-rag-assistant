"""Tests for connecting RAG retrieval to verified answer generation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evidence_rag.generation.rag_pipeline import ResearchRAGPipeline


def test_ask_verified_passes_retrieved_evidence_to_retry_generator():
    retry_generator = Mock()

    expected_result = SimpleNamespace(
        response=SimpleNamespace(
            text="Verified research answer [E1].",
            status="passed_checks",
            reasons=(),
        ),
        attempts=1,
    )

    retry_generator.generate.return_value = expected_result

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        retry_generator=retry_generator,
    )

    evidence = [Mock()]
    pipeline.retrieve_evidence = Mock(return_value=evidence)

    question = "How do Vision Transformers process images?"

    result = pipeline.ask_verified(question)

    retry_generator.generate.assert_called_once_with(
        question=question,
        evidence=evidence,
    )

    assert result is expected_result
    assert result.response.status == "passed_checks"
    assert result.attempts == 1


def test_ask_verified_requires_retry_generator():
    pipeline = ResearchRAGPipeline(llm_client=None)

    with pytest.raises(
        RuntimeError,
        match="Verified answering is not configured",
    ):
        pipeline.ask_verified("How do CNNs extract image features?")


def test_ask_verified_propagates_retrieval_errors():
    retry_generator = Mock()

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        retry_generator=retry_generator,
    )

    with pytest.raises(
        RuntimeError,
        match="Load a PDF",
    ):
        pipeline.ask_verified("How do CNNs extract image features?")

    retry_generator.generate.assert_not_called()
