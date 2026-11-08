"""End-to-end retrieval-augmented generation pipeline."""

from dataclasses import dataclass
from pathlib import Path

from evidence_rag.generation.answer_generator import (
    EvidenceAnswerGenerator,
    GeneratedAnswer,
)
from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.ingestion.pipeline import DocumentChunk, ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks
from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.reranker import CrossEncoderReranker


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for retrieval and document chunking."""

    top_k: int = 3
    candidate_k: int = 10
    chunk_size: int = 100
    overlap: int = 20

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("Invalid chunk overlap")


class ResearchRAGPipeline:
    """Retrieve and generate evidence-grounded answers from a research PDF."""

    def __init__(
        self,
        llm_client: OllamaLLMClient,
        config: RAGConfig | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.retriever = BM25Retriever()
        self.reranker = reranker
        self.answer_generator = EvidenceAnswerGenerator(llm_client)
        self.chunks: list[DocumentChunk] = []

    def load_document(self, pdf_path: str | Path) -> int:
        """Extract, filter, and index passages from a PDF."""

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

    def retrieve_evidence(self, question: str) -> list[Evidence]:
        """Retrieve relevant passages and optionally rerank them."""

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

    def ask(self, question: str) -> GeneratedAnswer:
        """Retrieve evidence and generate a citation-linked answer."""

        evidence = self.retrieve_evidence(question)

        return self.answer_generator.generate(
            question=question,
            evidence=evidence,
        )
