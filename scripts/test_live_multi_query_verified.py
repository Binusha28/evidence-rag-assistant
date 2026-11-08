"""Live end-to-end test for verified multi-query RAG."""

from pathlib import Path

from evidence_rag.generation.answer_generator import EvidenceAnswerGenerator
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)
from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier


def main() -> None:
    question = (
        "How do CNN-based retinal lesion detection methods differ "
        "from Vision Transformers in image processing?"
    )

    queries = [
        "CNN retinal lesion detection feature maps region proposals",
        "Vision Transformer image patches embeddings self attention",
    ]

    llm_client = OllamaLLMClient()

    verifier = SemanticClaimVerifier(
        device="cpu",
    )

    answer_generator = EvidenceAnswerGenerator(
        llm_client,
    )

    verified_generator = VerifiedAnswerGenerator(
        answer_generator=answer_generator,
        verifier=verifier,
    )

    retry_generator = RetryAnswerGenerator(
        verified_generator=verified_generator,
        max_attempts=2,
    )

    pipeline = ResearchRAGPipeline(
        llm_client=llm_client,
        config=RAGConfig(
            top_k=4,
            candidate_k=10,
            chunk_size=100,
            overlap=20,
        ),
        retry_generator=retry_generator,
    )

    count = pipeline.load_documents(
        [
            Path("data/raw/research.pdf"),
            Path("data/raw/vit_paper.pdf"),
        ]
    )

    print("\n=== DOCUMENTS INDEXED ===")
    print("Total chunks:", count)

    evidence = pipeline.retrieve_multi_query_evidence(queries)

    print("\n=== RETRIEVED EVIDENCE ===")
    for item in evidence:
        print(f"{item.citation} | {item.source} | Page {item.page_number}")

    result = pipeline.ask_multi_query_verified(
        question=question,
        queries=queries,
    )

    print("\n=== VERIFIED ANSWER ===")
    print(result.response.text)

    print("\n=== VERIFICATION STATUS ===")
    print(result.response.status)

    print("\n=== RETRY ATTEMPTS ===")
    print(result.attempts)

    print("\n=== VERIFICATION REASONS ===")
    for reason in result.response.reasons:
        print("-", reason)


if __name__ == "__main__":
    main()
