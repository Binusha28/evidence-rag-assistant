from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.embeddings import EmbeddingModel
from evidence_rag.retrieval.hybrid_search import HybridRetriever
from evidence_rag.retrieval.semantic_search import SemanticRetriever


def display_results(name, results, chunks):
    print(f"\n{name}")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        chunk = chunks[result.index]

        print(f"\nRank: {rank}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Score: {result.score:.5f}")
        print(f"Text: {chunk.text[:220]}")
        print("-" * 70)


def main():
    query = (
        "How do deep learning algorithms analyse retinal "
        "fundus photographs to detect diabetic retinopathy?"
    )

    print("Loading research PDF...")
    chunks = filter_reference_chunks(ingest_pdf("data/raw/research.pdf"))

    print(f"Indexed chunks: {len(chunks)}")

    passages = [chunk.text for chunk in chunks]

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    semantic = SemanticRetriever(embedding_model)
    bm25 = BM25Retriever()

    hybrid = HybridRetriever(semantic, bm25)

    print("Building indexes...")
    hybrid.index(passages)

    semantic_results = semantic.search(query, top_k=10)
    bm25_results = bm25.search(query, top_k=10)
    hybrid_results = hybrid.search(query, top_k=10)

    display_results(
        "SEMANTIC SEARCH",
        semantic_results,
        chunks,
    )

    display_results(
        "BM25 SEARCH",
        bm25_results,
        chunks,
    )

    display_results(
        "HYBRID SEARCH",
        hybrid_results,
        chunks,
    )


if __name__ == "__main__":
    main()
