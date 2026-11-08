import json
from pathlib import Path

from evidence_rag.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
)
from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.embeddings import EmbeddingModel
from evidence_rag.retrieval.hybrid_search import HybridRetriever
from evidence_rag.retrieval.reranker import CrossEncoderReranker
from evidence_rag.retrieval.semantic_search import SemanticRetriever


def main() -> None:
    evaluation_path = Path("data/processed/retrieval_relevance.json")

    with evaluation_path.open(encoding="utf-8") as file:
        evaluation = json.load(file)

    query = evaluation["query"]
    relevance = evaluation["judgments"]

    print("Loading and filtering PDF...")

    chunks = ingest_pdf("data/raw/research.pdf")
    chunks = filter_reference_chunks(chunks)
    chunks = filter_low_information_chunks(chunks)

    if not chunks:
        raise ValueError("No searchable chunks found")

    passages = [chunk.text for chunk in chunks]

    print(f"Indexed chunks: {len(chunks)}")

    embedding_model = EmbeddingModel()
    semantic = SemanticRetriever(embedding_model)
    bm25 = BM25Retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    hybrid.index(passages)

    print("Loading cross-encoder...")
    reranker = CrossEncoderReranker()

    candidate_count = min(15, len(chunks))
    top_k = 5

    candidate_sources = {
        "Semantic": semantic,
        "Hybrid RRF": hybrid,
    }

    print("\nRERANKING SOURCE COMPARISON")
    print("=" * 65)

    for source_name, retriever in candidate_sources.items():
        candidates = retriever.search(
            query,
            top_k=candidate_count,
        )

        candidate_passages = [result.text for result in candidates]

        reranked = reranker.rerank(
            query,
            candidate_passages,
            top_k=top_k,
        )

        retrieved_ids = [chunks[candidates[result.index].index].chunk_id for result in reranked]

        precision = precision_at_k(
            retrieved_ids,
            relevance,
            k=top_k,
        )

        ndcg = ndcg_at_k(
            retrieved_ids,
            relevance,
            k=top_k,
        )

        print(f"\n{source_name} + Cross-Encoder")
        print(f"Precision@{top_k}: {precision:.3f}")
        print(f"nDCG@{top_k}: {ndcg:.3f}")

        for rank, result in enumerate(reranked, start=1):
            chunk_id = retrieved_ids[rank - 1]
            grade = relevance.get(chunk_id, "Unjudged")

            print(f"{rank}. {chunk_id} | Score: {result.score:.4f} | Relevance: {grade}")

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()
