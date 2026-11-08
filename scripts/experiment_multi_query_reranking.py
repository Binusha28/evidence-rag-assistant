"""Compare BM25, Multi-Query RRF, and CrossEncoder reranking."""

import json
from pathlib import Path

from evidence_rag.evaluation.judgment_metrics import judgment_coverage_at_k
from evidence_rag.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
)
from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.multi_query_search import MultiQueryBM25Retriever
from evidence_rag.retrieval.reranker import CrossEncoderReranker

DATA_DIR = Path("data/processed")
PDF_PATH = Path("data/raw/research.pdf")

TOP_K = 5
CANDIDATES_PER_QUERY = 5
LARGE_POOL_SIZE = 15

RELEVANCE_FILES = {
    "Q001": "retrieval_relevance.json",
    "Q002": "retrieval_relevance_q002.json",
    "Q003": "retrieval_relevance_q003.json",
    "Q004": "retrieval_relevance_q004.json",
    "Q005": "retrieval_relevance_q005.json",
}

METHODS = (
    "BM25",
    "Multi-Query RRF",
    "RRF + CE (same 5)",
    "RRF + CE (pool 15)",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def rerank_ids(
    query: str,
    candidates: list,
    chunk_ids: list[str],
    reranker: CrossEncoderReranker,
) -> list[str]:
    """Map CrossEncoder output back to original chunk IDs."""
    reranked = reranker.rerank(
        query=query,
        passages=[candidate.text for candidate in candidates],
        top_k=TOP_K,
    )

    return [chunk_ids[candidates[result.index].index] for result in reranked]


def main() -> None:
    questions = load_json(DATA_DIR / "retrieval_questions.json")["questions"]

    variations = load_json(DATA_DIR / "retrieval_query_variations.json")

    chunks = ingest_pdf(PDF_PATH)
    chunks = filter_reference_chunks(chunks)
    chunks = filter_low_information_chunks(chunks)

    if not chunks:
        raise ValueError("No searchable chunks found")

    passages = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    bm25 = BM25Retriever()
    bm25.index(passages)

    multi_query = MultiQueryBM25Retriever(bm25)

    print("\nLoading CrossEncoder reranker...")
    reranker = CrossEncoderReranker()

    totals = {
        name: {
            "precision": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
            "coverage": 0.0,
        }
        for name in METHODS
    }

    unjudged = set()

    print("\n=== MULTI-QUERY CROSSENCODER EXPERIMENT ===")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")
    print(f"Candidates per query: {CANDIDATES_PER_QUERY}")
    print(f"Expanded fused pool: {LARGE_POOL_SIZE}")

    for question in questions:
        query_id = question["query_id"]
        query = question["query"]

        judgments = load_json(DATA_DIR / RELEVANCE_FILES[query_id])["judgments"]

        query_variations = variations[query_id]

        bm25_results = bm25.search(
            query,
            top_k=TOP_K,
        )

        fused_five = multi_query.search(
            queries=query_variations,
            top_k=TOP_K,
            candidates_per_query=CANDIDATES_PER_QUERY,
        )

        fused_large = multi_query.search(
            queries=query_variations,
            top_k=LARGE_POOL_SIZE,
            candidates_per_query=CANDIDATES_PER_QUERY,
        )

        results_by_method = {
            "BM25": [chunk_ids[result.index] for result in bm25_results],
            "Multi-Query RRF": [chunk_ids[result.index] for result in fused_five],
            "RRF + CE (same 5)": rerank_ids(query, fused_five, chunk_ids, reranker),
            "RRF + CE (pool 15)": rerank_ids(query, fused_large, chunk_ids, reranker),
        }

        print(f"\n{query_id}: {query}")

        for name, retrieved_ids in results_by_method.items():
            precision = precision_at_k(retrieved_ids, judgments, k=TOP_K)
            ndcg = ndcg_at_k(retrieved_ids, judgments, k=TOP_K)
            mrr = reciprocal_rank_at_k(retrieved_ids, judgments, k=TOP_K)
            coverage = judgment_coverage_at_k(retrieved_ids, judgments, k=TOP_K)

            totals[name]["precision"] += precision
            totals[name]["ndcg"] += ndcg
            totals[name]["mrr"] += mrr
            totals[name]["coverage"] += coverage

            print(
                f"  {name:<22} "
                f"P@5={precision:.3f} "
                f"nDCG@5={ndcg:.3f} "
                f"RR@5={mrr:.3f} "
                f"Coverage={coverage:.3f}"
            )

            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id not in judgments:
                    unjudged.add((query_id, chunk_id))
                    print(f"    UNJUDGED rank={rank}: {chunk_id}")

    count = len(questions)

    print("\n=== MACRO-AVERAGED RESULTS ===")
    print("Method                   P@5    nDCG@5  MRR@5   Coverage")
    print("-" * 65)

    for name in METHODS:
        scores = totals[name]

        print(
            f"{name:<24} "
            f"{scores['precision'] / count:.3f}  "
            f"{scores['ndcg'] / count:.3f}    "
            f"{scores['mrr'] / count:.3f}   "
            f"{scores['coverage'] / count:.3f}"
        )

    print("\n=== UNJUDGED PASSAGES ===")

    if not unjudged:
        print("None in the evaluated Top-5 results.")
    else:
        lookup = {chunk.chunk_id: chunk for chunk in chunks}

        for query_id, chunk_id in sorted(unjudged):
            chunk = lookup[chunk_id]

            print("\n" + "=" * 65)
            print(f"Question: {query_id}")
            print(f"Chunk: {chunk_id}")
            print(f"Page: {chunk.page_number}")
            print(f"Text:\n{chunk.text}")

    print("\nNote: Unjudged passages are treated as non-relevant by the current metrics.")


if __name__ == "__main__":
    main()
