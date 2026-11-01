
"""Document Explorer: inspect indexed documents, chunks, and metadata; search the knowledge base."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.state import init_session_state  # noqa: E402


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Explorer | DocuMind AI",
    page_icon="🗂️",
    layout="wide",
)

init_session_state()


# ---------------------------------------------------------
# PREMIUM UI STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(99, 75, 190, 0.13), transparent 35%),
            radial-gradient(ellipse at 90% 10%, rgba(40, 120, 190, 0.10), transparent 35%),
            #0b1020;
        color: #e8eaf5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #f4f4ff !important;
        letter-spacing: -0.5px;
    }

    p, label, .stCaption {
        color: #aeb6cc;
    }

    .page-hero {
        background: linear-gradient(120deg, #191b39 0%, #20234a 55%, #182d48 100%);
        border: 1px solid rgba(160, 150, 255, 0.20);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
    }

    .hero-tag {
        color: #b9a8ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-description {
        color: #c1c8dc;
        font-size: 14px;
        margin: 0;
        line-height: 1.7;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 700;
        color: #f1f2ff;
        margin-top: 25px;
        margin-bottom: 14px;
    }

    .stat-card {
        background: linear-gradient(145deg, #171e34, #1c2340);
        border: 1px solid rgba(150, 160, 220, 0.16);
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 108px;
    }

    .stat-label {
        color: #9da9c3;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stat-value {
        color: #ffffff;
        font-size: 27px;
        font-weight: 800;
        margin-top: 9px;
    }

    .document-card {
        background: linear-gradient(145deg, #171e34, #1b2340);
        border: 1px solid rgba(150, 160, 220, 0.17);
        border-radius: 14px;
        padding: 17px 20px;
        margin: 12px 0;
    }

    .document-name {
        color: #f1f2ff;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 7px;
        word-break: break-word;
    }

    .document-meta {
        color: #aab5ce;
        font-size: 12px;
    }

    .search-panel {
        background: linear-gradient(145deg, #171e34, #1b2340);
        border: 1px solid rgba(150, 160, 220, 0.17);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0 20px 0;
    }

    .result-card {
        background: linear-gradient(145deg, #171e34, #1b2340);
        border: 1px solid rgba(150, 160, 220, 0.17);
        border-radius: 14px;
        padding: 18px 20px;
        margin: 12px 0;
    }

    .result-title {
        color: #c8baff;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-meta {
        color: #96a5c3;
        font-size: 12px;
        margin-bottom: 12px;
    }

    .result-text {
        color: #d0d6e7;
        font-size: 13px;
        line-height: 1.8;
    }

    div[data-testid="stExpander"] {
        background: rgba(23, 30, 52, 0.85);
        border: 1px solid rgba(150, 160, 220, 0.17);
        border-radius: 14px;
        margin-bottom: 12px;
    }

    div[data-testid="stTextInput"] input {
        background: #151c31;
        color: #ffffff;
        border: 1px solid #343d5d;
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(150, 160, 220, 0.15);
        border-radius: 12px;
        overflow: hidden;
    }

    .empty-state {
        background: #171e34;
        border: 1px dashed #465071;
        border-radius: 16px;
        padding: 35px;
        text-align: center;
        color: #aeb6cc;
        margin-top: 20px;
    }

    .empty-icon {
        font-size: 35px;
        margin-bottom: 12px;
    }

    hr {
        border-color: rgba(150, 160, 220, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.markdown(
    """
    <div class="page-hero">
        <div class="hero-tag">DOCUMIND AI · KNOWLEDGE MANAGEMENT</div>
        <div class="hero-title">🗂️ Document Explorer</div>
        <p class="hero-description">
            Explore your indexed documents, inspect individual chunks,
            understand document metadata, and search your knowledge base.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------------------------------

kb = st.session_state.kb
chunks = kb.all_chunks()

if not chunks:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <h3>No documents indexed yet</h3>
            <p>
                Your knowledge base is currently empty.
                Upload and index documents from the main chat page's sidebar
                to start exploring them here.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------
# GROUP CHUNKS BY DOCUMENT
# ---------------------------------------------------------

by_doc: dict[str, list] = {}

for chunk in chunks:
    by_doc.setdefault(chunk.document_name, []).append(chunk)


total_documents = len(by_doc)
total_chunks = len(chunks)

document_types = {
    chunk.document_type.upper()
    for chunk in chunks
    if chunk.document_type
}

total_pages = len({
    (chunk.document_name, chunk.page_number)
    for chunk in chunks
    if chunk.page_number is not None
})


# ---------------------------------------------------------
# KNOWLEDGE BASE OVERVIEW
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">📊 Knowledge Base Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Documents</div>
            <div class="stat-value">{total_documents}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Total Chunks</div>
            <div class="stat-value">{total_chunks}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Total Pages</div>
            <div class="stat-value">{total_pages}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">File Types</div>
            <div class="stat-value">{len(document_types)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# INDEXED DOCUMENTS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">📚 Indexed Documents</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Your knowledge base contains {total_documents} indexed document(s). "
    "Expand a document to inspect its chunks and metadata."
)

for doc_name, doc_chunks in by_doc.items():

    doc_type = doc_chunks[0].document_type or "Unknown"

    pages = sorted({
        chunk.page_number
        for chunk in doc_chunks
        if chunk.page_number is not None
    })

    document_id = doc_chunks[0].document_id

    expander_title = (
        f"📄 {doc_name}  ·  {doc_type.upper()}  ·  "
        f"{len(doc_chunks)} chunks"
    )

    if pages:
        expander_title += f"  ·  {len(pages)} pages"

    with st.expander(expander_title):

        st.markdown(
            f"""
            <div class="document-card">
                <div class="document-name">📄 {doc_name}</div>
                <div class="document-meta">
                    Type: {doc_type.upper()} &nbsp; | &nbsp;
                    Chunks: {len(doc_chunks)} &nbsp; | &nbsp;
                    Pages: {len(pages) if pages else 'Not available'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"Document ID: {document_id}  ·  Index status: ✅ Indexed"
        )

        table = pd.DataFrame(
            [
                {
                    "Chunk ID": chunk.chunk_id,
                    "Chunk Index": chunk.chunk_index,
                    "Page": chunk.page_number,
                    "Section": chunk.section,
                    "Characters": len(chunk.text),
                    "Preview": (
                        chunk.text[:120] + "..."
                        if len(chunk.text) > 120
                        else chunk.text
                    ),
                }
                for chunk in doc_chunks
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            height=min(400, 40 + 35 * len(table)),
            hide_index=True,
        )


# ---------------------------------------------------------
# KNOWLEDGE BASE SEARCH
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">🔎 Search Your Knowledge Base</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="search-panel">
        <div class="document-name">Find information inside your documents</div>
        <div class="document-meta">
            Search uses BM25 lexical retrieval to find matching text chunks.
            Enter a keyword, phrase, or question.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

search_query = st.text_input(
    "Search indexed content",
    placeholder="e.g. What was the objective of my internship project?",
    key="document_explorer_search",
)

if search_query.strip():

    with st.spinner("Searching your indexed documents..."):

        hits = kb.bm25.search(search_query.strip(), top_k=10)

    if not hits:

        st.warning("No matching chunks found. Try different keywords.")

    else:

        st.success(f"Found {len(hits)} matching chunk(s).")

        for index, (chunk, score) in enumerate(hits, start=1):

            if chunk.page_number is not None:
                location = f"Page {chunk.page_number}"
            else:
                location = chunk.section or "Location not available"

            preview = (
                chunk.text[:400] + "..."
                if len(chunk.text) > 400
                else chunk.text
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        {index}. 📄 {chunk.document_name}
                    </div>
                    <div class="result-meta">
                        📍 {location}
                        &nbsp; | &nbsp;
                        🔢 Chunk {chunk.chunk_index}
                        &nbsp; | &nbsp;
                        🎯 BM25 Score: {score:.3f}
                    </div>
                    <div class="result-text">
                        {preview}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <h3>Ready to search</h3>
            <p>
                Enter a search query above to find relevant chunks
                from your indexed documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px 0;">
        <span style="color:#8290ad; font-size:12px;">
            DocuMind AI · Intelligent Document Exploration
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
