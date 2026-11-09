"""Assess whether an answer addresses the original research question."""

from dataclasses import dataclass

from evidence_rag.generation.llm_client import OllamaLLMClient


@dataclass(frozen=True)
class QuestionCoverageResult:
    status: str
    reason: str

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


class QuestionCoverageChecker:
    """Conservatively assess answer completeness using a local LLM."""

    def __init__(self, llm_client: OllamaLLMClient) -> None:
        self.llm_client = llm_client

    def check(self, question: str, answer: str) -> QuestionCoverageResult:
        if not question.strip() or not answer.strip():
            return QuestionCoverageResult(
                status="uncertain",
                reason="Question or answer is empty.",
            )

        prompt = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Determine whether the answer directly addresses every "
            "explicitly requested part of the research question.\n"
            "Judge coverage only, not factual correctness.\n"
            "Do not use outside knowledge to fill missing information.\n"
            "For questions requesting multiple items, every requested "
            "item must be identified or addressed.\n"
            "If the answer discusses only one of two requested components, "
            "classify it as INCOMPLETE.\n"
            "Respond with exactly one word: COMPLETE, INCOMPLETE, or UNCERTAIN."
        )

        system_prompt = (
            "You are a strict research-answer coverage evaluator. "
            "Assess whether the supplied answer addresses the supplied "
            "question. Do not generate an answer or add factual information. "
            "Return only COMPLETE, INCOMPLETE, or UNCERTAIN."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )
        except Exception:
            return QuestionCoverageResult(
                status="uncertain",
                reason="Coverage assessment could not be completed.",
            )

        status = response.strip().lower()

        if status not in {"complete", "incomplete", "uncertain"}:
            return QuestionCoverageResult(
                status="uncertain",
                reason="Coverage evaluator returned an invalid response.",
            )

        return QuestionCoverageResult(
            status=status,
            reason={
                "complete": "The answer addresses the requested question.",
                "incomplete": "The answer does not address every requested part.",
                "uncertain": "Answer coverage could not be determined reliably.",
            }[status],
        )
