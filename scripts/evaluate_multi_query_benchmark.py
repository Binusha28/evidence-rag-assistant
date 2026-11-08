"""Compare BM25 and multi-query BM25 + RRF retrieval."""

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
PDF_PATH = Path("data/raw/research.pdf")
TOP_K = 5

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


def calculate_metrics(
    retrieved_ids: list[str],
    relevance: dict[str, int],
) -> dict[str, float]:
    return {
        "precision": precision_at_k(retrieved_ids, relevance, k=TOP_K),
        "ndcg": ndcg_at_k(retrieved_ids, relevance, k=TOP_K),
        "reciprocal_rank": reciprocal_rank_at_k(retrieved_ids, relevance, k=TOP_K),
        "coverage": judgment_coverage_at_k(retrieved_ids, relevance, k=TOP_K),
    }


def main() -> None:
    questions_data = load_json(DATA_DIR / "retrieval_questions.json")
    variations = load_json(DATA_DIR / "retrieval_query_variations.json")

    original_chunks = ingest_pdf(PDF_PATH)

    chunks = filter_reference_chunks(original_chunks)
    chunks = filter_low_information_chunks(chunks)

    original_chunk_ids = {chunk.chunk_id for chunk in original_chunks}

    if not chunks:
        raise ValueError("No searchable chunks found")

    passages = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs detected")

    bm25 = BM25Retriever()
    bm25.index(passages)

    multi_query = MultiQueryBM25Retriever(bm25)

    methods = ("BM25", "Multi-Query RRF")
    metric_names = (
        "precision",
        "ndcg",
        "reciprocal_rank",
        "coverage",
    )

    totals = {method: {metric: 0.0 for metric in metric_names} for method in methods}

    print("\n=== MULTI-QUERY RETRIEVAL BENCHMARK ===")
    print(f"Indexed chunks: {len(chunks)}")

    questions = questions_data["questions"]

    for question in questions:
        query_id = question["query_id"]
        query = question["query"]

        relevance_file = RELEVANCE_FILES[query_id]
        relevance_data = load_json(DATA_DIR / relevance_file)
        relevance = relevance_data["judgments"]

        query_variations = variations[query_id]

        if not query_variations or query_variations[0] != query:
            raise ValueError(f"{query_id}: first variation must match the benchmark query")

        unknown_ids = set(relevance) - original_chunk_ids

        if unknown_ids:
            raise ValueError(
                f"{query_id}: judgments reference chunks "
                f"not found in the original document: "
                f"{sorted(unknown_ids)}"
            )

        filtered_ids = set(relevance) - set(chunk_ids)

        relevant_filtered_ids = {chunk_id for chunk_id in filtered_ids if relevance[chunk_id] >= 1}

        if relevant_filtered_ids:
            raise ValueError(
                f"{query_id}: relevant evidence was removed "
                f"by filtering: {sorted(relevant_filtered_ids)}"
            )

        if filtered_ids:
            print(
                f"  Note: {len(filtered_ids)} judged non-relevant chunk(s) excluded by filtering."
            )

        bm25_results = bm25.search(
            query=query,
            top_k=TOP_K,
        )

        multi_results = multi_query.search(
            queries=query_variations,
            top_k=TOP_K,
            candidates_per_query=15,
        )

        results_by_method = {
            "BM25": [chunk_ids[result.index] for result in bm25_results],
            "Multi-Query RRF": [chunk_ids[result.index] for result in multi_results],
        }

        print("\nUnjudged Multi-Query candidates:")

        unjudged_count = 0

        for rank, result in enumerate(multi_results, start=1):
            chunk = chunks[result.index]

            if chunk.chunk_id not in relevance:
                unjudged_count += 1

                print(f"  Rank {rank}: {chunk.chunk_id} | Page {chunk.page_number}")
                print(f"  Text: {' '.join(chunk.text.split())[:350]}")

        if unjudged_count == 0:
            print("  None")

        print(f"\n{query_id}: {query}")

        for method, retrieved_ids in results_by_method.items():
            scores = calculate_metrics(retrieved_ids, relevance)

            for metric, value in scores.items():
                totals[method][metric] += value

            print(
                f"{method:<17} "
                f"P@5={scores['precision']:.3f} "
                f"nDCG@5={scores['ndcg']:.3f} "
                f"RR@5={scores['reciprocal_rank']:.3f} "
                f"Coverage@5={scores['coverage']:.3f}"
            )

    print("\n=== MACRO-AVERAGED RESULTS ===")

    question_count = len(questions)

    for method in methods:
        scores = totals[method]

        print(
            f"{method:<17} "
            f"P@5={scores['precision'] / question_count:.3f} "
            f"nDCG@5={scores['ndcg'] / question_count:.3f} "
            f"MRR@5={scores['reciprocal_rank'] / question_count:.3f} "
            f"Coverage@5={scores['coverage'] / question_count:.3f}"
        )

    print(
        "\nNote: Relevance judgments are incomplete. "
        "Unjudged chunks count as non-relevant for "
        "Precision@5 and nDCG@5. "
        "MRR@5 uses known relevant chunks only."
    )


if __name__ == "__main__":
    main()
