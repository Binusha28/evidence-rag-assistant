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

BENCHMARK_DIR = Path("data/processed")
PDF_PATH = Path("data/raw/research.pdf")
TOP_K = 5
RERANK_CANDIDATES = 15

RELEVANCE_FILES = {
    "Q001": "retrieval_relevance.json",
    "Q002": "retrieval_relevance_q002.json",
    "Q003": "retrieval_relevance_q003.json",
    "Q004": "retrieval_relevance_q004.json",
    "Q005": "retrieval_relevance_q005.json",
}


def load_benchmark() -> list[dict]:
    """Load and validate benchmark relevance judgments."""
    benchmark = []

    for query_id, filename in RELEVANCE_FILES.items():
        file_path = BENCHMARK_DIR / filename

        with file_path.open(encoding="utf-8") as file:
            data = json.load(file)

        query = data.get("query")
        judgments = data.get("judgments")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{filename}: missing query")

        if not isinstance(judgments, dict) or not judgments:
            raise ValueError(f"{filename}: missing relevance judgments")

        if any(type(grade) is not int or grade not in (0, 1, 2) for grade in judgments.values()):
            raise ValueError(f"{filename}: invalid relevance grade")

        benchmark.append(
            {
                "query_id": query_id,
                "query": query,
                "judgments": judgments,
            }
        )

    return benchmark


def rerank_results(
    query: str,
    candidates: list,
    chunk_ids: list[str],
    reranker: CrossEncoderReranker,
) -> list[str]:
    """Rerank retrieved candidates and return their original chunk IDs."""
    passages = [candidate.text for candidate in candidates]

    reranked = reranker.rerank(
        query,
        passages,
        top_k=TOP_K,
    )

    return [chunk_ids[candidates[result.index].index] for result in reranked]


def main() -> None:
    benchmark = load_benchmark()

    print("\n=== FIVE-QUESTION RETRIEVAL BENCHMARK ===")
    print(f"Questions: {len(benchmark)}")
    print(
        "Relevance judgments:",
        sum(len(item["judgments"]) for item in benchmark),
    )

    print("\nLoading and filtering PDF chunks...")

    chunks = ingest_pdf(PDF_PATH)
    chunks = filter_reference_chunks(chunks)
    chunks = filter_low_information_chunks(chunks)

    passages = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    print(f"Indexed chunks: {len(chunks)}")
    print("\nInitialising retrieval models...")

    embedding_model = EmbeddingModel()
    reranker = CrossEncoderReranker()

    semantic = SemanticRetriever(embedding_model)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(
        SemanticRetriever(embedding_model),
        BM25Retriever(),
    )

    semantic.index(passages)
    bm25.index(passages)
    hybrid.index(passages)

    method_names = [
        "Semantic",
        "BM25",
        "Hybrid RRF",
        "Semantic + CE",
        "Hybrid + CE",
    ]

    totals = {name: {"precision": 0.0, "ndcg": 0.0} for name in method_names}

    for item in benchmark:
        query = item["query"]
        relevance = item["judgments"]

        print(f"\n{'=' * 70}")
        print(f"{item['query_id']}: {query}")
        print(f"{'=' * 70}")

        semantic_candidates = semantic.search(query, top_k=RERANK_CANDIDATES)
        bm25_results = bm25.search(query, top_k=TOP_K)
        hybrid_candidates = hybrid.search(query, top_k=RERANK_CANDIDATES)

        results_by_method = {
            "Semantic": [chunk_ids[result.index] for result in semantic_candidates[:TOP_K]],
            "BM25": [chunk_ids[result.index] for result in bm25_results],
            "Hybrid RRF": [chunk_ids[result.index] for result in hybrid_candidates[:TOP_K]],
            "Semantic + CE": rerank_results(
                query,
                semantic_candidates,
                chunk_ids,
                reranker,
            ),
            "Hybrid + CE": rerank_results(
                query,
                hybrid_candidates,
                chunk_ids,
                reranker,
            ),
        }

        for name, retrieved_ids in results_by_method.items():
            precision = precision_at_k(retrieved_ids, relevance, k=TOP_K)
            ndcg = ndcg_at_k(retrieved_ids, relevance, k=TOP_K)

            totals[name]["precision"] += precision
            totals[name]["ndcg"] += ndcg

            print(f"{name:<16} Precision@5={precision:.3f}  nDCG@5={ndcg:.3f}")

    print(f"\n{'=' * 70}")
    print("AVERAGE RESULTS ACROSS FIVE QUESTIONS")
    print(f"{'=' * 70}")

    question_count = len(benchmark)

    for name, scores in totals.items():
        avg_precision = scores["precision"] / question_count
        avg_ndcg = scores["ndcg"] / question_count

        print(f"{name:<16} Precision@5={avg_precision:.3f}  nDCG@5={avg_ndcg:.3f}")

    print(
        "\nNote: Unjudged passages are treated as non-relevant. "
        "These are preliminary results from incomplete relevance judgments."
    )


if __name__ == "__main__":
    main()
