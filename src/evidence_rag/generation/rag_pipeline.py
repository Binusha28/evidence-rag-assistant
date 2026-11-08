"""End-to-end retrieval-augmented generation pipeline."""

from dataclasses import dataclass
from pathlib import Path

from evidence_rag.generation.answer_generator import (
    EvidenceAnswerGenerator,
    GeneratedAnswer,
)
from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.retry_answer import (
    RetryAnswerGenerator,
    RetryAnswerResult,
)
from evidence_rag.ingestion.pipeline import DocumentChunk, ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.multi_query_search import MultiQueryBM25Retriever
from evidence_rag.retrieval.reranker import CrossEncoderReranker


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for retrieval and document chunking."""

    top_k: int = 3
    candidate_k: int = 10
    bm25_candidates_per_query: int | None = None
    chunk_size: int = 100
    overlap: int = 20

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")

        if self.bm25_candidates_per_query is not None and self.bm25_candidates_per_query <= 0:
            raise ValueError("bm25_candidates_per_query must be positive")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("Invalid chunk overlap")


class ResearchRAGPipeline:
    """Research assistant with evidence-grounded retrieval."""

    def __init__(
        self,
        llm_client: OllamaLLMClient,
        config: RAGConfig | None = None,
        reranker: CrossEncoderReranker | None = None,
        retry_generator: RetryAnswerGenerator | None = None,
    ) -> None:
        self.config = config or RAGConfig()

        self.retriever = BM25Retriever()
        self.reranker = reranker

        self.answer_generator = EvidenceAnswerGenerator(llm_client)
        self.retry_generator = retry_generator

        self.chunks: list[DocumentChunk] = []

    def load_document(
        self,
        pdf_path: str | Path,
    ) -> int:
        """Load, filter, chunk, and index a single PDF."""

        chunks = ingest_pdf(
            pdf_path,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
        )

        chunks = filter_reference_chunks(chunks)
        chunks = filter_low_information_chunks(chunks)

        if not chunks:
            raise ValueError("No usable passages found in the PDF")

        self.retriever.index([chunk.text for chunk in chunks])

        self.chunks = chunks

        return len(chunks)

    def load_documents(
        self,
        pdf_paths: list[str | Path] | tuple[str | Path, ...],
    ) -> int:
        """Load and index multiple PDFs into one collection."""

        if not pdf_paths:
            raise ValueError("At least one PDF path is required")

        all_chunks: list[DocumentChunk] = []
        seen_chunk_ids: set[str] = set()

        for pdf_path in pdf_paths:
            chunks = ingest_pdf(
                pdf_path,
                chunk_size=self.config.chunk_size,
                overlap=self.config.overlap,
            )

            chunks = filter_reference_chunks(chunks)
            chunks = filter_low_information_chunks(chunks)

            for chunk in chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    raise ValueError(
                        "Duplicate chunk identifier detected: "
                        f"{chunk.chunk_id}. "
                        "Check for duplicate PDF paths "
                        "or source names."
                    )

                seen_chunk_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)

        if not all_chunks:
            raise ValueError("No usable passages found in the supplied PDFs")

        self.retriever.index([chunk.text for chunk in all_chunks])

        self.chunks = all_chunks

        return len(all_chunks)

    def retrieve_evidence(
        self,
        question: str,
    ) -> list[Evidence]:
        """Retrieve evidence using single-query BM25."""

        if not self.chunks:
            raise RuntimeError("Load a PDF before asking questions")

        if not question.strip():
            raise ValueError("Question must not be empty")

        candidate_k = self.config.candidate_k if self.reranker is not None else self.config.top_k

        candidates = self.retriever.search(
            question,
            top_k=candidate_k,
        )

        if self.reranker is not None:
            ranked = self.reranker.rerank(
                query=question,
                passages=[item.text for item in candidates],
                top_k=self.config.top_k,
            )

            selections = [(candidates[item.index], item.score) for item in ranked]
        else:
            selections = [(item, item.score) for item in candidates]

        evidence = [
            Evidence(
                evidence_id=f"E{position}",
                chunk_id=self.chunks[result.index].chunk_id,
                source=self.chunks[result.index].source,
                page_number=self.chunks[result.index].page_number,
                text=self.chunks[result.index].text,
                retrieval_score=score,
            )
            for position, (result, score) in enumerate(
                selections,
                start=1,
            )
        ]

        return evidence

    def retrieve_multi_query_evidence(
        self,
        queries: list[str] | tuple[str, ...],
    ) -> list[Evidence]:
        """Retrieve evidence using multi-query BM25 and RRF."""

        if not self.chunks:
            raise RuntimeError("Load a PDF before asking questions")

        if not queries or any(not query.strip() for query in queries):
            raise ValueError("Queries must not be empty")

        multi_retriever = MultiQueryBM25Retriever(
            retriever=self.retriever,
        )

        pool_k = (
            max(self.config.top_k, self.config.candidate_k)
            if self.reranker is not None
            else self.config.top_k
        )

        results = multi_retriever.search(
            queries=queries,
            top_k=pool_k,
            candidates_per_query=(
                self.config.bm25_candidates_per_query
                if self.config.bm25_candidates_per_query is not None
                else self.config.candidate_k
            ),
        )

        if self.reranker is not None and results:
            ranked = self.reranker.rerank(
                query=" ".join(queries),
                passages=[item.text for item in results],
                top_k=self.config.top_k,
            )

            selections = [(results[item.index], item.score) for item in ranked]
        else:
            selections = [(item, item.score) for item in results]

        evidence = [
            Evidence(
                evidence_id=f"E{position}",
                chunk_id=self.chunks[result.index].chunk_id,
                source=self.chunks[result.index].source,
                page_number=self.chunks[result.index].page_number,
                text=self.chunks[result.index].text,
                retrieval_score=score,
            )
            for position, (result, score) in enumerate(
                selections,
                start=1,
            )
        ]

        return evidence

    def ask(
        self,
        question: str,
    ) -> GeneratedAnswer:
        """Generate an answer using single-query evidence."""

        evidence = self.retrieve_evidence(question)

        return self.answer_generator.generate(
            question=question,
            evidence=evidence,
        )

    def ask_verified(
        self,
        question: str,
        evidence: list[Evidence] | None = None,
    ) -> RetryAnswerResult:
        """Generate a verified answer using a consistent evidence set."""

        if self.retry_generator is None:
            raise RuntimeError(
                "Verified answering is not configured. "
                "Provide a RetryAnswerGenerator "
                "when creating the pipeline."
            )

        if not question.strip():
            raise ValueError("Question must not be empty")

        if evidence is None:
            evidence = self.retrieve_evidence(question)

        return self.retry_generator.generate(
            question=question,
            evidence=evidence,
        )

    def ask_multi_query_verified(
        self,
        question: str,
        queries: list[str] | tuple[str, ...],
    ) -> RetryAnswerResult:
        """Generate a verified answer using multi-query evidence."""

        if self.retry_generator is None:
            raise RuntimeError(
                "Verified answering is not configured. "
                "Provide a RetryAnswerGenerator "
                "when creating the pipeline."
            )

        if not question.strip():
            raise ValueError("Question must not be empty")

        evidence = self.retrieve_multi_query_evidence(
            queries=queries,
        )

        return self.retry_generator.generate(
            question=question,
            evidence=evidence,
        )
