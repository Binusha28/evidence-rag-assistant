"""Evaluate candidate-pool sizes for multi-query BM25 retrieval."""

import json
from pathlib import Path

from evidence_rag.evaluation.judgment_metrics import (
    judgment_coverage_at_k,
)
from evidence_rag.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
)
from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.quality_filter import (
    filter_low_information_chunks,
)
from evidence_rag.ingestion.section_filter import (
    filter_reference_chunks,
)
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.multi_query_search import (
    MultiQueryBM25Retriever,
)

DATA_DIR = Path("data/processed")
TOP_K = 5
CANDIDATE_COUNTS = (5, 10, 15, 20)

RELEVANCE_FILES = {
    "Q001": "retrieval_relevance.json",
    "Q002": "retrieval_relevance_q002.json",
    "Q003": "retrieval_relevance_q003.json",
    "Q004": "retrieval_relevance_q004.json",
    "Q005": "retrieval_relevance_q005.json",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    questions = load_json(DATA_DIR / "retrieval_questions.json")["questions"]

    variations = load_json(DATA_DIR / "retrieval_query_variations.json")

    chunks = ingest_pdf("data/raw/research.pdf")
    chunks = filter_reference_chunks(chunks)
    chunks = filter_low_information_chunks(chunks)

    if not chunks:
        raise ValueError("No searchable chunks found")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    bm25 = BM25Retriever()
    bm25.index([chunk.text for chunk in chunks])

    multi_query = MultiQueryBM25Retriever(bm25)

    print("\n=== MULTI-QUERY CANDIDATE EXPERIMENT ===")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Top K: {TOP_K}")

    print("\nCandidates | Precision@5 | NDCG@5 | MRR@5 | Coverage@5")
    print("-" * 65)

    for candidate_count in CANDIDATE_COUNTS:
        totals = {
            "precision": 0.0,
            "ndcg": 0.0,
            "rr": 0.0,
            "coverage": 0.0,
        }

        for question in questions:
            query_id = question["query_id"]

            relevance = load_json(DATA_DIR / RELEVANCE_FILES[query_id])["judgments"]

            query_variations = variations[query_id]

            results = multi_query.search(
                queries=query_variations,
                top_k=TOP_K,
                candidates_per_query=candidate_count,
            )

            retrieved_ids = [chunk_ids[result.index] for result in results]

            totals["precision"] += precision_at_k(retrieved_ids, relevance, k=TOP_K)
            totals["ndcg"] += ndcg_at_k(retrieved_ids, relevance, k=TOP_K)
            totals["rr"] += reciprocal_rank_at_k(retrieved_ids, relevance, k=TOP_K)
            totals["coverage"] += judgment_coverage_at_k(retrieved_ids, relevance, k=TOP_K)

        count = len(questions)

        print(
            f"{candidate_count:^10} | "
            f"{totals['precision'] / count:^11.3f} | "
            f"{totals['ndcg'] / count:^7.3f} | "
            f"{totals['rr'] / count:^5.3f} | "
            f"{totals['coverage'] / count:^10.3f}"
        )

    print("\nNote: Results with incomplete judgment coverage must be interpreted cautiously.")


if __name__ == "__main__":
    main()
