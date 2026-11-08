"""Tests for the local Ollama language-model client."""

from unittest.mock import MagicMock, patch

import pytest

from evidence_rag.generation.llm_client import (
    LLMConfig,
    LLMGenerationError,
    OllamaLLMClient,
    clean_model_output,
)


def test_default_configuration():
    config = LLMConfig()

    assert config.model == "qwen3:4b"
    assert config.temperature == 0.1
    assert config.max_tokens == 1024


@pytest.mark.parametrize(
    "values",
    [
        {"model": ""},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"max_tokens": 0},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        LLMConfig(**values)


@patch("evidence_rag.generation.llm_client.ollama.chat")
def test_generate_answer(mock_chat):
    mock_chat.return_value = MagicMock(
        message=MagicMock(content="RAG combines retrieval and generation.")
    )

    client = OllamaLLMClient()
    answer = client.generate(
        "What is RAG?",
        system_prompt="Answer concisely.",
    )

    assert answer == "RAG combines retrieval and generation."

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["model"] == "qwen2.5:3b"
    assert kwargs["think"] is False
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"


def test_empty_prompt():
    client = OllamaLLMClient()

    with pytest.raises(ValueError, match="Prompt must not be empty"):
        client.generate(" ")


@patch("evidence_rag.generation.llm_client.ollama.chat")
def test_empty_response(mock_chat):
    mock_chat.return_value = MagicMock(message=MagicMock(content=""))

    client = OllamaLLMClient()

    with pytest.raises(LLMGenerationError, match="empty response"):
        client.generate("What is RAG?")


@patch("evidence_rag.generation.llm_client.ollama.chat")
def test_custom_configuration(mock_chat):
    mock_chat.return_value = MagicMock(message=MagicMock(content="Test response"))

    config = LLMConfig(
        model="custom-model",
        temperature=0.3,
        max_tokens=100,
    )

    client = OllamaLLMClient(config)
    client.generate("Test question")

    kwargs = mock_chat.call_args.kwargs

    assert kwargs["model"] == "custom-model"
    assert kwargs["options"]["temperature"] == 0.3
    assert kwargs["options"]["num_predict"] == 100


def test_clean_model_output_removes_thinking():
    content = "<think>I need to reason about this.</think>RAG combines retrieval and generation."

    assert clean_model_output(content) == ("RAG combines retrieval and generation.")


def test_clean_model_output_handles_missing_opening_tag():
    content = "Let me reason about this first.</think>RAG retrieves evidence before answering."

    assert clean_model_output(content) == ("RAG retrieves evidence before answering.")


def test_rejects_truncated_generation(monkeypatch):
    from types import SimpleNamespace

    from evidence_rag.generation.llm_client import (
        LLMGenerationError,
        OllamaLLMClient,
    )

    response = SimpleNamespace(
        done_reason="length",
        message=SimpleNamespace(content="Incomplete response"),
    )

    monkeypatch.setattr(
        "evidence_rag.generation.llm_client.ollama.chat",
        lambda **kwargs: response,
    )

    client = OllamaLLMClient()

    with pytest.raises(LLMGenerationError, match="generation limit"):
        client.generate("Explain diabetic retinopathy.")
