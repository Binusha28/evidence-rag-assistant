"""Diagnose generated answers and semantic verification failures."""

from pathlib import Path

from evidence_rag.generation.answer_generator import EvidenceAnswerGenerator
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.rag_pipeline import RAGConfig, ResearchRAGPipeline
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.verification.citation_repair import SemanticCitationRepair
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier


def main() -> None:
    llm_client = OllamaLLMClient()

    verifier = SemanticClaimVerifier(
        model_name="cross-encoder/nli-deberta-v3-base",
        device="cpu",
    )

    verified_generator = VerifiedAnswerGenerator(
        answer_generator=EvidenceAnswerGenerator(llm_client),
        verifier=verifier,
        citation_repair=SemanticCitationRepair(
            verifier=verifier,
            min_confidence=0.95,
        ),
    )

    pipeline = ResearchRAGPipeline(
        llm_client=llm_client,
        config=RAGConfig(
            top_k=3,
            candidate_k=10,
            chunk_size=100,
            overlap=20,
        ),
    )

    pipeline.load_documents(
        [
            Path("data/raw/research.pdf"),
            Path("data/raw/vit_paper.pdf"),
        ]
    )

    question = (
        "How do CNN-based retinal lesion detection methods "
        "differ from Vision Transformers in image processing?"
    )

    evidence = pipeline.retrieve_evidence(question)

    print("\n=== EVIDENCE ===")
    for item in evidence:
        print(f"\n{item.citation} | {item.source} | Page {item.page_number}")
        print(item.text)

    print("\n=== GENERATING ANSWER ===")

    result = verified_generator.generate(
        question=question,
        evidence=evidence,
    )

    print("\n=== GENERATED ANSWER ===")
    print(result.answer)

    print("\n=== VERIFICATION DECISION ===")
    print("Accepted:", result.is_accepted)
    print("Decision:", result.decision)

    print("\n=== VERIFICATION REPORT ===")
    print(result.verification)

    print("\n=== DIAGNOSIS COMPLETE ===")


if __name__ == "__main__":
    main()
