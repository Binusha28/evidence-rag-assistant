import json
from pathlib import Path

from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.embeddings import EmbeddingModel
from evidence_rag.retrieval.hybrid_search import HybridRetriever
from evidence_rag.retrieval.semantic_search import SemanticRetriever


def main() -> None:
    questions_path = Path("data/processed/retrieval_questions.json")

    with questions_path.open(encoding="utf-8") as file:
        benchmark = json.load(file)

    questions = benchmark["questions"]

    print("Loading research PDF...")

    chunks = ingest_pdf("data/raw/research.pdf")
    chunks = filter_reference_chunks(chunks)
    chunks = filter_low_information_chunks(chunks)

    if not chunks:
        raise ValueError("No searchable chunks found")

    passages = [chunk.text for chunk in chunks]

    print(f"Indexed chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    semantic = SemanticRetriever(EmbeddingModel())
    bm25 = BM25Retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    hybrid.index(passages)

    retrievers = {
        "Semantic": semantic,
        "BM25": bm25,
        "Hybrid RRF": hybrid,
    }

    print("\nMULTI-QUESTION RETRIEVAL INSPECTION")
    print("=" * 70)

    for question in questions:
        query_id = question["query_id"]
        query = question["query"]

        print(f"\nQUESTION {query_id}")
        print(f"Query: {query}")
        print("-" * 70)

        for name, retriever in retrievers.items():
            results = retriever.search(query, top_k=5)

            print(f"\n{name} — Top 5")

            for rank, result in enumerate(results, start=1):
                chunk = chunks[result.index]

                preview = " ".join(chunk.text.split())[:180]

                print(
                    f"{rank}. {chunk.chunk_id} "
                    f"| Page: {chunk.page_number} "
                    f"| Score: {result.score:.4f}"
                )
                print(f"   Preview: {preview}...")

    print("\nCandidate inspection completed successfully.")


if __name__ == "__main__":
    main()
