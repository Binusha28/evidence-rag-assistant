# Evidence-Based RAG Research Assistant

An evidence-grounded research assistant that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG), local large language models, semantic claim verification, and citation validation.

The project focuses on making AI-generated research answers more traceable, verifiable, and reliable.

## Key Features

- **Multi-PDF ingestion:** Extracts and processes text from uploaded research documents.
- **Evidence retrieval:** Uses BM25 search to identify relevant document passages.
- **Cross-encoder reranking:** Reorders retrieved passages to improve evidence selection.
- **Local LLM generation:** Uses Ollama with Qwen2.5 for document-grounded answers.
- **Citation validation:** Checks whether generated answers reference valid evidence identifiers.
- **Semantic verification:** Uses natural language inference (NLI) to assess whether cited passages support generated claims.
- **Retry and recovery:** Retries unsuccessful generations and supports partially verified answers when appropriate.
- **Safe fallback:** Avoids presenting an unverified answer as evidence-grounded.
- **Streamlit interface:** Provides document upload, question answering, verification status, and expandable evidence.
- **Automated testing:** Includes unit and integration tests for retrieval, generation, and verification components.

## System Architecture

```text
PDF Documents
     |
     v
PDF Text Extraction
     |
     v
Text Chunking and Quality Filtering
     |
     v
BM25 Evidence Retrieval
     |
     v
Cross-Encoder Reranking
     |
     v
Local LLM (Ollama + Qwen2.5)
     |
     v
Citation and Semantic Claim Verification
     |
     +---- Verified Answer
     |
     +---- Retry / Partial Recovery
     |
     +---- Safe Fallback
     |
     v
Streamlit Research Interface
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Application interface | Streamlit |
| Local LLM runtime | Ollama |
| Language model | Qwen2.5 |
| PDF processing | PyMuPDF |
| Keyword retrieval | BM25 |
| Passage reranking | Cross-Encoder |
| Claim verification | Natural Language Inference (NLI) |
| Dependency management | uv |
| Automated testing | pytest |
| Code quality | Ruff |

## Project Structure

- `app.py` — Streamlit application.
- `src/evidence_rag/ingestion/` — PDF loading, chunking, and filtering.
- `src/evidence_rag/retrieval/` — Search and reranking.
- `src/evidence_rag/generation/` — LLM generation and answer recovery.
- `src/evidence_rag/verification/` — Citation and semantic verification.
- `src/evidence_rag/evaluation/` — Evaluation utilities and metrics.
- `scripts/` — Experiments, diagnostics, and benchmarks.
- `tests/` — Automated tests.

## Verification Philosophy

The assistant distinguishes between answers that pass automated verification, answers that are only partially verified, and answers for which sufficient support cannot be established.

Automated verification reduces unsupported claims but does not guarantee scientific or factual correctness. Users should review the original cited evidence before relying on generated answers.

## Development Status

Active development. Core document ingestion, retrieval, reranking, local answer generation, verification, and Streamlit functionality have been implemented.

## Author

Paulin Binusha Sahayaranjan

GitHub: [Binusha28](https://github.com/Binusha28)
