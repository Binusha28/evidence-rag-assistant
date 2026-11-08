"""Local language-model client powered by Ollama."""

import re
from dataclasses import dataclass

import ollama

DEFAULT_MODEL = "qwen2.5:3b"


class LLMGenerationError(RuntimeError):
    """Raised when the local language model cannot generate an answer."""


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Model name must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


def clean_model_output(content: str) -> str:
    """Remove explicitly marked reasoning from model output."""
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Handle responses with a closing tag but no opening tag.
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(
            r"</think>",
            cleaned,
            flags=re.IGNORECASE,
        )[-1]

    # Do not expose incomplete reasoning as a final answer.
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        return ""

    return cleaned.strip()


class OllamaLLMClient:
    """Generate responses using a locally running Ollama model."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:
            response = ollama.chat(
                model=self.config.model,
                messages=messages,
                think=False,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
        except (ollama.ResponseError, ollama.RequestError) as exc:
            raise LLMGenerationError(
                "Unable to generate an answer with Ollama. "
                "Check that Ollama is running and the model is installed."
            ) from exc
        if response.done_reason == "length":
            raise LLMGenerationError(
                "The model reached its generation limit before completing "
                "the answer. Try shorter evidence passages or increase "
                "the generation limit."
            )
        content = response.message.content

        if not content or not content.strip():
            raise LLMGenerationError("The language model returned an empty response.")

        cleaned = clean_model_output(content)

        if not cleaned:
            raise LLMGenerationError("The model returned no usable answer after cleaning.")

        return cleaned


def test_clean_model_output_removes_thinking():
    content = "<think>I need to reason about this.</think>RAG combines retrieval and generation."

    assert clean_model_output(content) == ("RAG combines retrieval and generation.")


def test_clean_model_output_handles_missing_opening_tag():
    content = "Let me reason about this first.</think>RAG retrieves evidence before answering."

    assert clean_model_output(content) == ("RAG retrieves evidence before answering.")
