
"""
DocuMind AI — Premium Document Intelligence Assistant
Author: Makwana Priyanka
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.state import init_session_state
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

# ---------------------------------------------------------
# PREMIUM DARK UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080b16;
        --panel: #111629;
        --panel2: #171d34;
        --purple: #9b7bff;
        --blue: #5b9dff;
        --text: #f3f3ff;
        --muted: #9ca6c2;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(100,70,210,.16), transparent 35%),
            radial-gradient(ellipse at 90% 20%, rgba(40,110,220,.10), transparent 35%),
            #080b16;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8,11,22,.75);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10152a 0%, #0b1020 100%);
        border-right: 1px solid rgba(155,123,255,.18);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .brand span {
        color: #a78bfa;
    }

    .brand-sub {
        color: #929db9;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 36px 34px;
        border-radius: 23px;
        background: linear-gradient(115deg,
            rgba(85,65,170,.34),
            rgba(28,54,111,.25),
            rgba(17,22,41,.88));
        border: 1px solid rgba(167,139,250,.28);
        margin: 8px 0 25px 0;
        box-shadow: 0 15px 50px rgba(0,0,0,.22);
    }

    .hero h1 {
        font-size: clamp(28px, 4vw, 43px);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1.8px;
        color: white;
        margin: 12px 0;
    }

    .hero p {
        color: #c0c7e0;
        font-size: 14px;
        line-height: 1.8;
        max-width: 650px;
    }

    .eyebrow {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 30px;
        background: rgba(167,139,250,.13);
        color: #c4b5fd;
        border: 1px solid rgba(167,139,250,.28);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 700;
        color: #f3f3ff;
        margin: 18px 0 12px 0;
    }

    .metric-card {
        background: linear-gradient(145deg, #171e35, #111629);
        border: 1px solid rgba(151,163,210,.15);
        padding: 19px;
        border-radius: 17px;
        min-height: 115px;
    }

    .metric-label {
        font-size: 12px;
        color: #9ca8c5;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 29px;
        color: #ffffff;
        font-weight: 800;
    }

    .metric-icon {
        font-size: 19px;
        float: right;
    }

    .feature-card {
        background: rgba(20,27,49,.86);
        border: 1px solid rgba(155,123,255,.17);
        border-radius: 17px;
        padding: 20px;
        min-height: 145px;
    }

    .feature-icon {
        font-size: 25px;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 14px;
        font-weight: 700;
        color: #eeeaff;
        margin-bottom: 7px;
    }

    .feature-description {
        font-size: 12px;
        color: #a5aec7;
        line-height: 1.6;
    }

    .suggestion {
        background: rgba(20,27,49,.85);
        border: 1px solid rgba(155,123,255,.2);
        border-radius: 13px;
        padding: 13px;
        color: #d9d7ed;
        font-size: 12px;
        min-height: 66px;
    }

    .footer {
        text-align: center;
        color: #727e9b;
        font-size: 11px;
        padding: 28px 0 10px 0;
        letter-spacing: .4px;
    }

    .status-pill {
        display: inline-block;
        background: rgba(34,197,94,.10);
        border: 1px solid rgba(34,197,94,.25);
        color: #86efac;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(20,27,49,.65);
        border: 1px solid rgba(155,123,255,.12);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(155,123,255,.3);
        background: linear-gradient(110deg, #30245c, #202b50);
        color: white;
        font-weight: 600;
        transition: .2s ease;
    }

    .stButton > button:hover {
        border-color: #a78bfa;
        color: white;
        background: linear-gradient(110deg, #493681, #293b70);
        transform: translateY(-1px);
    }

    .stTextInput input,
    .stNumberInput input {
        background: #171d34;
        color: white;
        border-radius: 10px;
    }

    [data-testid="stChatInput"] {
        border-color: rgba(155,123,255,.3);
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #60a5fa);
    }

    hr {
        border-color: rgba(155,163,210,.15);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def metric_card(icon, label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon, title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence(evidence):
    if not evidence:
        return

    with st.expander(f"📎 View Sources & Evidence ({len(evidence)})"):
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
        <div class="brand">🧠 Docu<span>Mind</span> AI</div>
        <div class="brand-sub">Document Intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<span class="status-pill">● Workspace Active</span>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Add your documents",
        type=["pdf", "docx", "txt", "csv", "md", "html", "htm"],
        accept_multiple_files=True,
        help=f"Maximum {settings.max_file_size_mb} MB per file.",
    )

    with st.expander("⚙️ Advanced indexing settings"):

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
        "⬆️ Index Documents",
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

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):

        st.session_state.kb.clear()
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()

    st.markdown("### 🔍 Retrieval Controls")

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

    st.markdown("### ✨ AI Generation")

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

    with st.expander("🖥️ System Information"):

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

st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">✦ NEXT-GENERATION DOCUMENT AI</span>
        <h1>Knowledge, beyond<br>the documents.</h1>
        <p>
            Your intelligent document workspace. Upload your knowledge,
            ask natural-language questions and discover evidence-backed
            insights with Retrieval-Augmented Generation.
        </p>
        <span class="status-pill">● AI WORKSPACE READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dashboard metrics

st.markdown('<div class="section-heading">Your Workspace Overview</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card("📚", "Documents Indexed", num_documents)

with m2:
    metric_card("🧩", "Knowledge Chunks", num_chunks)

with m3:
    metric_card("💬", "Chat Conversations", len(st.session_state.chat_history))

with m4:
    metric_card("⚡", "AI Provider", settings.llm_provider.upper())

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

    st.markdown(
        '<div class="section-heading">Get Started with DocuMind</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Your knowledge base is currently empty. "
        "Upload a PDF, DOCX, TXT or another supported document "
        "using the sidebar to begin."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        feature_card(
            "📂",
            "Upload Documents",
            "Bring your PDFs, Word files, CSVs and other documents into one workspace.",
        )

    with c2:
        feature_card(
            "🔎",
            "Smart Retrieval",
            "Find relevant information using semantic and keyword-based search.",
        )

    with c3:
        feature_card(
            "✨",
            "Grounded AI Answers",
            "Ask questions and receive evidence-backed answers with citations.",
        )

    st.write("")

    st.markdown(
        '<div class="footer">Your documents. Your knowledge. One intelligent workspace.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# CHAT WORKSPACE
# ---------------------------------------------------------

else:

    st.markdown(
        '<div class="section-heading">💬 AI Conversation Workspace</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Ask questions about your indexed documents. "
        "DocuMind AI retrieves relevant evidence before generating an answer."
    )

    # Welcome suggestions when no conversation exists

    if not st.session_state.chat_history:

        st.write("")

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">✦</div>
                <div class="feature-title">Hello! I'm DocuMind AI.</div>
                <div class="feature-description">
                    I've got your documents ready. What would you like to discover today?
                    Ask me anything based on your uploaded knowledge.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
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

st.markdown(
    """
    <div class="footer">
        DOCUMIND AI · INTELLIGENT DOCUMENT ASSISTANT<br>
        Crafted with precision, powered by Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True,
)
