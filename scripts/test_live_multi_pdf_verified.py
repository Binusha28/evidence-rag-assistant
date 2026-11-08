"""Live end-to-end test of verified multi-PDF research answering."""

from pathlib import Path

from evidence_rag.generation.answer_generator import EvidenceAnswerGenerator
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)
from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.verification.citation_repair import SemanticCitationRepair
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier


def main() -> None:
    pdf_paths = [
        Path("data/raw/research.pdf"),
        Path("data/raw/vit_paper.pdf"),
    ]

    for path in pdf_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing PDF: {path}")

    print("\n=== INITIALISING MODELS ===")

    llm_client = OllamaLLMClient()

    verifier = SemanticClaimVerifier(
        model_name="cross-encoder/nli-deberta-v3-base",
        device="cpu",
    )

    citation_repair = SemanticCitationRepair(
        verifier=verifier,
        min_confidence=0.95,
    )

    answer_generator = EvidenceAnswerGenerator(
        llm_client=llm_client,
    )

    verified_generator = VerifiedAnswerGenerator(
        answer_generator=answer_generator,
        verifier=verifier,
        citation_repair=citation_repair,
    )

    retry_generator = RetryAnswerGenerator(
        verified_generator=verified_generator,
        max_attempts=2,
    )

    pipeline = ResearchRAGPipeline(
        llm_client=llm_client,
        config=RAGConfig(
            top_k=3,
            candidate_k=10,
            chunk_size=100,
            overlap=20,
        ),
        retry_generator=retry_generator,
    )

    print("\n=== INDEXING RESEARCH DOCUMENTS ===")

    chunk_count = pipeline.load_documents(pdf_paths)

    print("Documents indexed:", len(pdf_paths))
    print("Total chunks:", chunk_count)

    question = (
        "How do CNN-based retinal lesion detection methods "
        "differ from Vision Transformers in image processing?"
    )

    print("\n=== RESEARCH QUESTION ===")
    print(question)

    evidence = pipeline.retrieve_evidence(question)

    print("\n=== RETRIEVED EVIDENCE ===")

    for item in evidence:
        print(f"{item.citation} | {item.source} | Page {item.page_number}")
        print("Text:", item.text[:300])
        print("-" * 60)

    print("\n=== GENERATING VERIFIED ANSWER ===")

    result = pipeline.ask_verified(question)

    print("\n=== FINAL ANSWER ===")
    print(result.response.text)

    print("\n=== VERIFICATION RESULT ===")
    print("Status:", result.response.status)
    print("Attempts:", result.attempts)

    print("\n=== VERIFICATION REASONS ===")

    if result.response.reasons:
        for reason in result.response.reasons:
            print("-", reason)
    else:
        print("No failure reasons reported.")

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    main()
