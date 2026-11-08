"""Streamlit interface for the Evidence-Based RAG Research Assistant."""

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from evidence_rag.generation.answer_generator import EvidenceAnswerGenerator
from evidence_rag.generation.llm_client import OllamaLLMClient
from evidence_rag.generation.rag_pipeline import RAGConfig, ResearchRAGPipeline
from evidence_rag.generation.retry_answer import RetryAnswerGenerator
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.retrieval.reranker import CrossEncoderReranker
from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier

st.set_page_config(
    page_title="Evidence-Based Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Evidence-Based RAG Research Assistant")
st.caption(
    "Ask questions about research papers and receive evidence-grounded answers with verification."
)


@st.cache_resource(show_spinner="Loading verification model...")
def create_verifier():
    return SemanticClaimVerifier(device="cpu")


def create_pipeline():
    llm = OllamaLLMClient()
    generator = EvidenceAnswerGenerator(llm)
    verified = VerifiedAnswerGenerator(
        answer_generator=generator,
        verifier=create_verifier(),
    )
    retry = RetryAnswerGenerator(
        verified_generator=verified,
        max_attempts=2,
        allow_partial_answers=True,
    )
    return ResearchRAGPipeline(
        llm_client=llm,
        config=RAGConfig(
            top_k=5,
            candidate_k=15,
            bm25_candidates_per_query=15,
        ),
        reranker=CrossEncoderReranker(device="cpu"),
        retry_generator=retry,
    )


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "document_key" not in st.session_state:
    st.session_state.document_key = None

if "document_directory" not in st.session_state:
    st.session_state.document_directory = None


with st.sidebar:
    st.header("Research documents")
    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.info("Answers are generated locally using Ollama.")


if uploaded_files:
    documents = [(file.name, file.getvalue()) for file in uploaded_files]
    document_key = hashlib.sha256(
        repr([(name, hashlib.sha256(data).hexdigest()) for name, data in documents]).encode()
    ).hexdigest()

    if document_key != st.session_state.document_key:
        with st.spinner("Indexing research documents..."):
            try:
                directory = tempfile.TemporaryDirectory()
                paths = []

                for index, (name, content) in enumerate(documents):
                    safe_name = Path(name).name
                    path = Path(directory.name) / f"{index}_{safe_name}"
                    path.write_bytes(content)
                    paths.append(path)

                pipeline = create_pipeline()
                count = pipeline.load_documents(paths)

            except Exception as error:
                directory.cleanup()
                st.error(f"Document processing failed: {error}")
            else:
                previous = st.session_state.document_directory
                st.session_state.pipeline = pipeline
                st.session_state.document_key = document_key
                st.session_state.document_directory = directory

                if previous is not None:
                    previous.cleanup()

                st.success(f"Indexed {len(paths)} PDF(s) into {count} passages.")
else:
    st.session_state.pipeline = None
    st.session_state.document_key = None
    previous = st.session_state.document_directory
    if previous is not None:
        previous.cleanup()
        st.session_state.document_directory = None


question = st.text_input(
    "Ask a research question",
    placeholder="What are the main findings of these papers?",
)

if st.button("Generate verified answer", type="primary"):
    if st.session_state.pipeline is None:
        st.warning("Please upload at least one PDF first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        pipeline = st.session_state.pipeline

        with st.spinner("Retrieving evidence and verifying the answer..."):
            try:
                evidence = pipeline.retrieve_evidence(question)
                result = pipeline.ask_verified(question, evidence=evidence)
            except Exception as error:
                st.error(f"Answer generation failed: {error}")
            else:
                status = result.response.status

                if status == "passed_checks":
                    st.success("Complete answer passed automated verification checks.")
                    st.subheader("Evidence-grounded answer")
                elif status == "partially_verified":
                    st.warning(
                        "Partial answer: only independently verified sentences were retained."
                    )
                    st.subheader("Partially verified answer")
                else:
                    st.error(
                        "Insufficient verification: no answer passed the required evidence checks."
                    )
                    st.subheader("Verification result")

                st.markdown(result.response.text)

                st.subheader("Verification details")
                st.write("Status:", status)
                st.write("Generation attempts:", result.attempts)

                if result.response.reasons:
                    for reason in result.response.reasons:
                        st.write("-", reason)

                st.caption(
                    "Automated evidence verification is not a guarantee "
                    "of factual or scientific correctness. "
                    "Review the cited source passages before relying "
                    "on any research finding."
                )

                st.subheader("Retrieved evidence")
                for item in evidence:
                    with st.expander(
                        f"{item.citation} — {Path(item.source).name}, page {item.page_number}"
                    ):
                        st.write(item.text)
