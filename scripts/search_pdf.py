import argparse
from pathlib import Path

from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.document_search import DocumentSearcher
from evidence_rag.retrieval.embeddings import EmbeddingModel
from evidence_rag.retrieval.semantic_search import SemanticRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search evidence in a research PDF.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF")
    parser.add_argument("query", help="Research question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("Extracting and chunking PDF...")
    chunks = ingest_pdf(args.pdf)
    original_count = len(chunks)

    chunks = filter_reference_chunks(chunks)

    print(f"Original chunks: {original_count}")
    print(f"Chunks after filtering: {len(chunks)}")
    print(f"Removed chunks: {original_count - len(chunks)}")

    if not chunks:
        raise ValueError("No extractable text found in this PDF")

    print(f"Created {len(chunks)} text chunks.")

    print("Loading embedding model...")
    model = EmbeddingModel()

    retriever = SemanticRetriever(model)
    searcher = DocumentSearcher(retriever)

    print("Generating embeddings...")
    searcher.index(chunks)

    print("Searching for relevant evidence...")
    results = searcher.search(args.query, top_k=args.top_k)

    print("\nSEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page_number}")
        print(f"Similarity: {result.score:.3f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Evidence: {result.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
