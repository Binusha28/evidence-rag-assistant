"""Test retrieval and source provenance across real research PDFs."""

from pathlib import Path

from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)


def main() -> None:
    pdf_paths = sorted(Path("data/raw").glob("*.pdf"))

    if len(pdf_paths) < 2:
        print("Please place at least two PDFs inside data/raw/")
        return

    print("\n=== LOADING RESEARCH DOCUMENTS ===")

    for path in pdf_paths:
        print("PDF:", path.name)

    pipeline = ResearchRAGPipeline(
        llm_client=None,
        config=RAGConfig(
            top_k=3,
            candidate_k=10,
            chunk_size=100,
            overlap=20,
        ),
    )

    count = pipeline.load_documents(pdf_paths)

    print("\nTotal indexed chunks:", count)
    print("Total documents:", len(pdf_paths))

    question = input("\nEnter a research question: ").strip()

    if not question:
        print("No question entered.")
        return

    evidence = pipeline.retrieve_evidence(question)

    print("\n=== RETRIEVED EVIDENCE ===")

    for item in evidence:
        print("\nCitation:", item.citation)
        print("Source:", item.source)
        print("Page:", item.page_number)
        print("Score:", item.retrieval_score)
        print("Text:", item.text[:600])
        print("-" * 60)


if __name__ == "__main__":
    main()
