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
    # --------------------------------------------------
    # 1. Load evaluation dataset
    # --------------------------------------------------
    evaluation_path = Path("data/processed/retrieval_relevance.json")

    with evaluation_path.open(encoding="utf-8") as file:
        evaluation = json.load(file)

    query = evaluation["query"]
    relevance = evaluation["judgments"]

    top_k = 5

    # --------------------------------------------------
    # 2. Load and filter research PDF
    # --------------------------------------------------
    print("Loading research PDF...")

    chunks = ingest_pdf("data/raw/research.pdf")
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

    passages = [chunk.text for chunk in chunks]

    # --------------------------------------------------
    # 3. Initialize retrieval models
    # --------------------------------------------------
    print("\nLoading semantic embedding model...")

    embedding_model = EmbeddingModel()

    semantic = SemanticRetriever(embedding_model)
    bm25 = BM25Retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    # --------------------------------------------------
    # 4. Build retrieval indexes
    # --------------------------------------------------
    print("Building retrieval indexes...")

    hybrid.index(passages)

    retrievers = {
        "Semantic": semantic,
        "BM25": bm25,
        "Hybrid RRF": hybrid,
    }

    # --------------------------------------------------
    # 5. Evaluate baseline retrieval methods
    # --------------------------------------------------
    print("\nRETRIEVAL EVALUATION")
    print("=" * 65)
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print("-" * 65)

    for name, retriever in retrievers.items():
        results = retriever.search(
            query,
            top_k=top_k,
        )

        retrieved_ids = [chunks[result.index].chunk_id for result in results]

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

        print(f"\n{name}")
        print(f"Precision@{top_k}: {precision:.3f}")
        print(f"nDCG@{top_k}: {ndcg:.3f}")

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            grade = relevance.get(
                chunk_id,
                "Unjudged",
            )

            print(f"  {rank}. {chunk_id} | Relevance: {grade}")

    # --------------------------------------------------
    # 6. Load cross-encoder reranker
    # --------------------------------------------------
    print("\nLoading cross-encoder reranker...")

    reranker = CrossEncoderReranker()

    # --------------------------------------------------
    # 7. Retrieve top 15 hybrid candidates
    # --------------------------------------------------
    candidate_count = min(15, len(chunks))

    print(f"Retrieving top {candidate_count} hybrid candidates...")

    hybrid_candidates = hybrid.search(
        query,
        top_k=candidate_count,
    )

    candidate_passages = [result.text for result in hybrid_candidates]

    # --------------------------------------------------
    # 8. Rerank ALL candidate passages
    # --------------------------------------------------
    print("Reranking candidate passages...")

    all_reranked_results = reranker.rerank(
        query,
        candidate_passages,
        top_k=candidate_count,
    )

    # Keep only top 5 for evaluation
    reranked_results = all_reranked_results[:top_k]

    # --------------------------------------------------
    # 9. Inspect ALL cross-encoder scores
    # --------------------------------------------------
    print("\nCROSS-ENCODER SCORE INSPECTION")
    print("=" * 65)

    for rank, result in enumerate(
        all_reranked_results,
        start=1,
    ):
        original_index = hybrid_candidates[result.index].index

        chunk = chunks[original_index]

        grade = relevance.get(
            chunk.chunk_id,
            "Unjudged",
        )

        print(f"{rank:2}. {chunk.chunk_id} | Score: {result.score:.4f} | Relevance: {grade}")

    # --------------------------------------------------
    # 10. Map top 5 reranked results to PDF chunks
    # --------------------------------------------------
    reranked_ids = [
        chunks[hybrid_candidates[result.index].index].chunk_id for result in reranked_results
    ]

    # --------------------------------------------------
    # 11. Evaluate top 5 reranked results
    # --------------------------------------------------
    reranked_precision = precision_at_k(
        reranked_ids,
        relevance,
        k=top_k,
    )

    reranked_ndcg = ndcg_at_k(
        reranked_ids,
        relevance,
        k=top_k,
    )

    print("\nHybrid RRF + Cross-Encoder")
    print(f"Precision@{top_k}: {reranked_precision:.3f}")
    print(f"nDCG@{top_k}: {reranked_ndcg:.3f}")

    for rank, chunk_id in enumerate(
        reranked_ids,
        start=1,
    ):
        grade = relevance.get(
            chunk_id,
            "Unjudged",
        )

        print(f"  {rank}. {chunk_id} | Relevance: {grade}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
