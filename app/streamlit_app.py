"""
DocuMind AI — Document Intelligence Assistant
Author: Makwana Priyanka
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.state import init_session_state
from app.styles import card, footer, inject_css, metric_strip, page_header, section
from src.config.settings import settings
from src.ingestion.loader import (
    DocumentIngestionError,
    DocumentTooLargeError,
    UnsupportedFileTypeError,
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI | Intelligence Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_css()

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------


def render_evidence(evidence):
    if not evidence:
        return

    with st.expander(f"View sources & evidence ({len(evidence)})"):
        for e in evidence:
            location = (
                f"Page {e.page_number}"
                if e.page_number
                else (e.section or "")
            )

            st.markdown(
                f"**[{e.citation_index}] {e.document_name}**"
                + (f" · {location}" if location else "")
            )

            st.caption(f"Relevance score: {e.score:.3f}")

            st.write(
                e.text[:600] + ("..." if len(e.text) > 600 else "")
            )
            st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="brand">Docu<span>Mind</span> AI</div>
        <div class="brand-sub">Your document workspace</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<span class="pill">Workspace ready</span>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown("### Knowledge Base")

    uploaded_files = st.file_uploader(
        "Add your documents",
        type=["pdf", "docx", "txt", "csv", "md", "html", "htm"],
        accept_multiple_files=True,
        help=f"Maximum {settings.max_file_size_mb} MB per file.",
    )

    with st.expander("Advanced indexing settings"):

        col1, col2 = st.columns(2)

        with col1:
            adv_chunk_size = st.number_input(
                "Chunk size",
                min_value=200,
                max_value=3000,
                value=settings.chunk_size,
                step=50,
            )

        with col2:
            adv_chunk_overlap = st.number_input(
                "Overlap",
                min_value=0,
                max_value=500,
                value=settings.chunk_overlap,
                step=10,
            )

        chunk_strategy = st.selectbox(
            "Chunking strategy",
            ["recursive", "token", "semantic"],
            index=0,
        )

    if uploaded_files and st.button(
        "Index Documents",
        use_container_width=True,
    ):

        progress = st.progress(0.0, text="Preparing documents...")
        errors = []

        for i, file in enumerate(uploaded_files):

            try:
                result = st.session_state.kb.ingest_file(
                    file.getvalue(),
                    file.name,
                    chunk_size=adv_chunk_size,
                    chunk_overlap=adv_chunk_overlap,
                    strategy=chunk_strategy,
                )

                if not result.success:
                    errors.append(f"{file.name}: {result.error}")

            except (
                DocumentIngestionError,
                DocumentTooLargeError,
                UnsupportedFileTypeError,
            ) as exc:
                errors.append(f"{file.name}: {exc}")

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Processing {file.name}",
            )

        st.session_state.kb.persist()
        progress.empty()

        for error in errors:
            st.error(error)

        if errors:
            st.warning("Some documents could not be indexed.")
        else:
            st.success("All documents indexed successfully!")

        st.rerun()

    stats = st.session_state.kb.stats()

    col1, col2 = st.columns(2)

    col1.metric("Documents", stats["num_documents"])
    col2.metric("Chunks", stats["num_chunks"])

    if st.button("Clear Knowledge Base", use_container_width=True):

        st.session_state.kb.clear()
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()

    st.markdown("### Retrieval Controls")

    st.session_state.top_k = st.slider(
        "Top-K chunks",
        1,
        15,
        st.session_state.top_k,
    )

    st.session_state.alpha = st.slider(
        "Dense / Sparse balance",
        0.0,
        1.0,
        float(st.session_state.alpha),
        0.05,
        help="1 = semantic search, 0 = keyword search.",
    )

    st.session_state.enable_reranking = st.toggle(
        "Reranking",
        value=st.session_state.enable_reranking,
    )

    st.session_state.enable_query_rewriting = st.toggle(
        "Query rewriting",
        value=st.session_state.enable_query_rewriting,
    )

    st.divider()

    st.markdown("### AI Generation")

    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        float(st.session_state.temperature),
        0.05,
    )

    st.session_state.max_tokens = st.slider(
        "Maximum output tokens",
        128,
        4096,
        int(st.session_state.max_tokens),
        128,
    )

    st.divider()

    with st.expander("System Information"):

        st.caption(f"Provider: {settings.llm_provider.upper()}")
        st.caption(f"Model: {settings.llm_model}")
        st.caption(f"Embeddings: {settings.embedding_model}")
        st.caption(f"Vector DB: {settings.vector_db.upper()}")
        st.caption(f"Reranker: {settings.reranker_model}")

    st.divider()

    st.caption("Designed & Developed by")
    st.markdown("**Makwana Priyanka**")
    st.caption("B.E. Information Technology")
    st.caption("Data Science · Machine Learning · GenAI")

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------

stats = st.session_state.kb.stats()
num_documents = stats["num_documents"]
num_chunks = stats["num_chunks"]

page_header(
    "Document workspace",
    "Ask questions. Find clear answers.",
    "Upload documents, search their contents, and review the sources behind each answer.",
    pill="Ready for your documents",
)

# Dashboard metrics

section("Workspace overview")

metric_strip(
    [
        ("Documents", num_documents),
        ("Indexed passages", num_chunks),
        ("Conversations", len(st.session_state.chat_history)),
        ("Provider", settings.llm_provider.upper()),
    ],
    compact=True,
)

st.write("")

if st.session_state.llm_error:
    st.warning(
        f"LLM configuration issue: {st.session_state.llm_error}. "
        "Check your .env API_KEY and provider settings."
    )

# ---------------------------------------------------------
# EMPTY KNOWLEDGE BASE WELCOME SCREEN
# ---------------------------------------------------------

if num_chunks == 0:

    section("Start with your documents")

    st.info(
        "Your knowledge base is currently empty. "
        "Upload a PDF, DOCX, TXT or another supported document "
        "using the sidebar to begin."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "Upload documents",
            "Bring your PDFs, Word files, CSVs and other documents into one workspace.",
        )

    with c2:
        card(
            "Search across files",
            "Find relevant passages with keyword and meaning-based search.",
        )

    with c3:
        card(
            "Review the sources",
            "See the passages used to support each response.",
        )

# ---------------------------------------------------------
# CHAT WORKSPACE
# ---------------------------------------------------------

else:

    section("Ask about your documents")

    st.caption(
        "Ask a question about the files in your workspace. "
        "Relevant passages are reviewed before an answer is written."
    )

    # Welcome suggestions when no conversation exists

    if not st.session_state.chat_history:

        st.write("")

        card(
            "Your documents are ready.",
            "Ask a question below to find information across the files you have indexed.",
            accent=True,
        )

        st.write("")
        st.markdown("**Try asking something like:**")

        suggestions = [
            "What is the title of my internship project?",
            "Summarize the main objectives of this document.",
            "What technologies are mentioned?",
        ]

        cols = st.columns(3)

        for i, suggestion in enumerate(suggestions):

            with cols[i]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_query = suggestion
                    st.rerun()

    # Existing conversation history

    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg.get("evidence"):
                render_evidence(msg["evidence"])

            if "confidence" in msg:
                st.caption(
                    f"Evidence confidence: {msg['confidence']:.2f}"
                )

    # User input

    typed_query = st.chat_input(
        "Ask anything about your documents..."
    )

    pending_query = st.session_state.pop("pending_query", None)
    query = typed_query or pending_query

    if query:

        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            if st.session_state.pipeline is None:

                error_msg = (
                    f"AI configuration issue: {st.session_state.llm_error}"
                )

                st.error(error_msg)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )

            else:

                with st.spinner(
                    "Retrieving evidence and generating your answer..."
                ):

                    try:

                        response = st.session_state.pipeline.answer(
                            query,
                            memory=st.session_state.memory,
                            top_k=st.session_state.top_k,
                            alpha=st.session_state.alpha,
                            enable_query_rewriting=(
                                st.session_state.enable_query_rewriting
                            ),
                            enable_reranking=(
                                st.session_state.enable_reranking
                            ),
                        )

                        st.markdown(response.answer)

                        render_evidence(response.evidence)

                        st.caption(
                            f"Evidence confidence: {response.confidence:.2f}"
                        )

                        st.session_state.memory.add_user(query)
                        st.session_state.memory.add_assistant(
                            response.answer
                        )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": response.answer,
                                "evidence": response.evidence,
                                "confidence": response.confidence,
                            }
                        )

                    except Exception as exc:

                        error_msg = (
                            f"Something went wrong while answering: {exc}"
                        )

                        st.error(error_msg)

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": error_msg,
                            }
                        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

footer(
    "DocuMind AI · Document assistant",
    "Search, review, and work from your source material.",
)