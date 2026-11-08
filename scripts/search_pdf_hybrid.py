import argparse
from pathlib import Path

from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.document_search import DocumentSearcher
from evidence_rag.retrieval.embeddings import EmbeddingModel
from evidence_rag.retrieval.hybrid_search import HybridRetriever
from evidence_rag.retrieval.semantic_search import SemanticRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a research PDF using hybrid retrieval.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Extracting and chunking research PDF...")
    chunks = ingest_pdf(args.pdf)

    original_count = len(chunks)

    chunks = filter_reference_chunks(chunks)
    after_reference_filter = len(chunks)

    chunks = filter_low_information_chunks(chunks)

    print(f"Original chunks: {original_count}")
    print(f"Removed reference-like chunks: {original_count - after_reference_filter}")
    print(f"Removed low-information chunks: {after_reference_filter - len(chunks)}")
    print(f"Final indexed chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No searchable evidence found in this PDF")

    print("Loading semantic embedding model...")
    embedding_model = EmbeddingModel()

    semantic_retriever = SemanticRetriever(embedding_model)
    bm25_retriever = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
    )

    searcher = DocumentSearcher(hybrid_retriever)

    print("Building hybrid search index...")
    searcher.index(chunks)

    print("Searching for relevant evidence...")
    results = searcher.search(args.query, top_k=args.top_k)

    print("\nHYBRID SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page_number}")
        print(f"RRF Score: {result.score:.5f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Evidence: {result.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
