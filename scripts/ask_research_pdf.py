"""Ask a research question using the local evidence-based RAG pipeline."""

from pathlib import Path

from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.rag_pipeline import ResearchRAGPipeline
from evidence_rag.retrieval.reranker import CrossEncoderReranker


def main() -> None:
    pdf_path = Path("data/raw/research.pdf")

    question = (
        "How do convolutional neural networks help detect "
        "diabetic retinopathy from retinal fundus images?"
    )

    pipeline = ResearchRAGPipeline(
        llm_client=OllamaLLMClient(),
        reranker=CrossEncoderReranker(),
    )
    print("\nLoading research PDF...")
    count = pipeline.load_document(pdf_path)
    print(f"Indexed passages: {count}")

    print(f"\nQuestion: {question}")
    print("\nGenerating evidence-grounded answer...\n")

    result = pipeline.ask(question)

    print("ANSWER")
    print(result.answer)

    print("\nEVIDENCE SOURCES")

    for evidence in result.evidence:
        print(f"{evidence.citation} {evidence.source}, page {evidence.page_number}")
        print(f"Chunk ID: {evidence.chunk_id}")
        print()


if __name__ == "__main__":
    main()
