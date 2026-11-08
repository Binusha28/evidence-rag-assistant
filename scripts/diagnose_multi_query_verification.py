"""Diagnose claim-level failures in multi-query verified RAG."""

from pathlib import Path

from evidence_rag.generation.answer_generator import EvidenceAnswerGenerator
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier
from evidence_rag.verification.verification_feedback import (
    build_verification_feedback,
)


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

    verified_generator = VerifiedAnswerGenerator(
        answer_generator=EvidenceAnswerGenerator(llm_client),
        verifier=verifier,
    )

    pipeline = ResearchRAGPipeline(
        llm_client=llm_client,
        config=RAGConfig(
            top_k=4,
            candidate_k=10,
            chunk_size=100,
            overlap=20,
        ),
    )

    count = pipeline.load_documents(
        [
            Path("data/raw/research.pdf"),
            Path("data/raw/vit_paper.pdf"),
        ]
    )

    print("\n=== INDEXED DOCUMENTS ===")
    print("Total chunks:", count)

    evidence = pipeline.retrieve_multi_query_evidence(
        queries=queries,
    )

    print("\n=== RETRIEVED EVIDENCE ===")

    for item in evidence:
        print("\n" + "=" * 70)
        print(f"{item.citation} | {item.source} | Page {item.page_number}")
        print("Chunk ID:", item.chunk_id)
        print("Text:")
        print(item.text)

    print("\n=== GENERATING VERIFIED ANSWER ===")

    result = verified_generator.generate(
        question=question,
        evidence=evidence,
    )

    print("\n=== GENERATED ANSWER ===")
    print(result.answer)

    print("\n=== VERIFICATION DECISION ===")
    print("Accepted:", result.is_accepted)
    print("Decision:", result.decision)

    report = result.verification

    print("\n=== CLAIM-LEVEL ASSESSMENTS ===")

    if not report.assessments:
        print("No claim assessments were produced.")

    for number, assessment in enumerate(
        report.assessments,
        start=1,
    ):
        print("\n" + "-" * 70)
        print("Claim number:", number)
        print("Claim:", assessment.claim)
        print("Citation:", assessment.citation)
        print("Label:", assessment.label)
        print("Full assessment:", assessment)

    print("\n=== UNCITED SENTENCES ===")

    if report.uncited_sentences:
        for sentence in report.uncited_sentences:
            print("-", sentence)
    else:
        print("None")

    print("\n=== LEADING CITATIONS ===")

    if report.leading_citations:
        for citation in report.leading_citations:
            print("-", citation)
    else:
        print("None")

    print("\n=== VERIFICATION FEEDBACK ===")
    print(build_verification_feedback(report))

    print("\n=== FINAL VERIFICATION RESULT ===")
    print("Verified:", report.is_verified)

    print("\n=== DIAGNOSIS COMPLETE ===")


if __name__ == "__main__":
    main()
