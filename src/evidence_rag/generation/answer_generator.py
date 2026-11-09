"""Generate evidence-grounded answers from retrieved research passages."""

from dataclasses import dataclass

from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.verification.citation_validator import (
    CitationValidationError,
    normalize_citations,
    validate_citations,
)

SYSTEM_PROMPT = """
You are a careful, evidence-grounded scientific research assistant.

STRICT EVIDENCE RULES:
1. Answer using only the supplied evidence passages.
2. Never invent scientific findings, performance metrics, or comparisons.
3. Every factual sentence must end with at least one supporting citation.
4. Use citation identifiers exactly as provided, such as [E1] or [E2].
5. Place citations after the statement they support, never at the
   beginning of a sentence.
6. Never write "[E1] shows" or "[E2] demonstrates".
   Instead write "The cited study reports ... [E1]."
7. Do not cite a passage unless it directly supports the statement.
8. Distinguish datasets, imaging modalities, methods, and populations.
9. Never attribute OCTA findings to retinal fundus photographs.
10. Do not claim that a method outperforms another unless the evidence
    explicitly contains that comparison.
11. When evidence is incomplete, explain the limitation rather than
    filling gaps with assumptions.
12. Prefer a short, well-supported answer over a long speculative one.
13. Do not add introductory or concluding factual sentences without
    citations.
14. Write self-contained factual sentences that can be verified independently.
15. Avoid ambiguous references such as "these regions", "this method",
    "they", or "it" when referring to scientific findings.
16. Reuse precise scientific terminology from the supporting evidence.
17. Do not add details merely to make a claim sound more complete.
18. For comparisons involving different evidence sources, write
    separate factual sentences for each source.
19. Each factual sentence must be independently supported by every
    evidence passage cited in that sentence.
20. Do not combine facts from different papers using "while",
    "whereas", "but", or "in contrast" in a single sentence.
21. Never place multiple independently supported claims in one sentence
    merely to make the answer shorter.
22. Prefer one clearly supported scientific claim per sentence.
23. Preserve the certainty and tense of the source evidence. Words such as 'may', 'could', 'suggests', and 'expected' must not become definitive claims.
24. Never present proposed, anticipated, or potential clinical benefits as experimentally demonstrated outcomes.
25. Prefer the narrowest scientifically meaningful claim directly supported by its cited passage.
26. Return only the final answer.
CORRECT CITATION EXAMPLE:
"The input fundus image is fed into the CNN network to obtain
a feature map [E1]."

INCORRECT CITATION EXAMPLES:
"[E1] demonstrates that CNNs identify lesions."
"CNNs identify lesions. [E1] reports high accuracy."
"A CNN processes retinal images to detect lesions.
The system classifies diseases [E1]."
""".strip()


INSUFFICIENT_EVIDENCE = (
    "The provided documents do not contain enough information to answer this question."
)


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    answer: str
    evidence: tuple[Evidence, ...]


class EvidenceAnswerGenerator:
    """Generate research answers using retrieved, cited evidence."""

    def __init__(self, llm_client: OllamaLLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        evidence: list[Evidence] | tuple[Evidence, ...],
        feedback: str | None = None,
    ) -> GeneratedAnswer:
        """Generate an answer and validate its citation identifiers."""

        if not question.strip():
            raise ValueError("Question must not be empty")

        if not evidence:
            return GeneratedAnswer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE,
                evidence=(),
            )

        evidence_context = "\n\n".join(item.format_for_prompt() for item in evidence)

        prompt = (
            f"Research question:\n{question}\n\n"
            f"Available evidence:\n{evidence_context}\n\n"
            "Write a concise, scientifically accurate answer using only "
            "the available evidence.\n"
            "QUESTION COVERAGE REQUIREMENTS:\n0. Answer the exact research question directly in the first factual sentence.\n   If asked to name or identify components, explicitly name those components\n   rather than describing only their internal mechanisms.\n   Include a supporting evidence citation for that first sentence.\n"
            "1. Identify the main components of the research question.\n"
            "2. Address each component when directly supported by evidence.\n"
            "3. For comparison questions, explain both subjects separately "
            "before describing supported similarities or differences.\n"
            "4. Do not replace a requested methodological comparison "
            "with an unrelated performance comparison.\n"
            "5. If evidence supports only part of the question, answer "
            "that part and explicitly identify what remains unsupported.\n"
            "6. Never invent missing details to make an answer complete.\n"
            "Explain the relevant mechanism or methodology first.\n"
            "Include findings only when directly supported by a passage.\n"
            "Prefer 2–4 short factual sentences, or fewer when the "
            "evidence is limited.\n"
            "Each sentence should express exactly one narrow claim.\n"
            "Use one supporting citation per sentence whenever possible.\n"
            "Do not combine multiple applications, methods, outcomes, "
            "or clinical benefits into a single sentence.\n"
            "Prefer wording closely aligned with the cited passage.\n"
            "Do not add a broader interpretation to a directly supported fact.\n"
            "If only one claim is clearly supported, provide that one "
            "claim rather than expanding it with uncertain details.\n"
            "REQUIRED: Every factual sentence must contain a citation "
            "at its end, before the final punctuation.\n"
            "Example: The input fundus image is fed into the CNN network "
            "to obtain a feature map [E1].\n"
            "Never begin a sentence with a citation identifier.\n"
            "Do not write '[E1] shows' or '[E2] demonstrates'.\n"
            "Use the exact citation identifiers provided in the evidence.\n"
            "Do not confuse retinal fundus images with OCTA images.\n"
            "Do not invent accuracy values or performance comparisons.\n"
            "Write each factual sentence as a self-contained claim.\n"
            "CLAIM GRANULARITY REQUIREMENTS:\n"
            "1. Express each independently supported claim in its own sentence.\n"
            "2. When comparing different papers, describe each method in a "
            "separate sentence with its own supporting citation.\n"
            "3. Do not combine independently supported claims with 'while', "
            "'whereas', 'but', or 'in contrast' in the same sentence.\n"
            "4. Every cited passage must support the entire factual sentence "
            "to which its citation is attached.\n"
            "5. If two facts require different sources, use two sentences "
            "instead of a combined sentence.\n"
            "Avoid ambiguous references such as 'these regions' "
            "or 'this method'.\n"
            "Use precise terminology from the supporting evidence.\n"
            "Do not add details that the evidence does not explicitly support.\n"
            "If a claim cannot be supported, omit it.\n"
            "If the evidence is insufficient, clearly explain the limitation.\n"
            "Return only the final answer."
        )

        if feedback:
            prompt += (
                "\n\nIMPORTANT: The previous answer failed verification.\n"
                "Correct the following problems:\n"
                f"{feedback}\n\n"
                "REGENERATION REQUIREMENTS:\n"
                "1. Write a completely revised answer using only the evidence.\n"
                "2. Every factual sentence MUST end with its own supporting "
                "citation before the final punctuation.\n"
                "3. Do not leave introductory or explanatory sentences uncited.\n"
                "4. Use terminology directly supported by the cited evidence.\n"
                "5. Remove unsupported claims, but retain relevant claims "
                "that are directly supported by the evidence.\n"
                "6. Preserve coverage of the research question wherever "
                "the supplied evidence supports it.\n"
                "7. For comparison questions, retain supported information "
                "about both subjects.\n"
                "8. Do not introduce unsupported claims to fill missing gaps.\n"
                "9. Check every factual sentence for a valid citation.\n"
                "10. If a combined comparison sentence fails verification, split "
                "it into separate independently supported sentences.\n"
                "11. Do not cite one source for a sentence that also contains "
                "claims supported only by another source.\n"
            )

        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        answer = normalize_citations(answer)

        validation = validate_citations(answer, evidence)

        if validation.invalid_ids:
            raise CitationValidationError(
                message=(
                    "The generated answer contains unknown citations: "
                    + ", ".join(validation.invalid_ids)
                ),
                feedback=(
                    "The previous answer used citation identifiers that "
                    "were not provided in the evidence. "
                    "Use only the available citation identifiers. "
                    "Every factual sentence must have a valid citation."
                ),
            )

        if not validation.cited_ids:
            raise CitationValidationError(
                message=("The generated answer does not contain any evidence citations."),
                feedback=(
                    "The previous answer contained no valid evidence "
                    "citations. Regenerate the answer with a supporting "
                    "citation at the end of every factual sentence."
                ),
            )

        return GeneratedAnswer(
            question=question,
            answer=answer,
            evidence=tuple(evidence),
        )
