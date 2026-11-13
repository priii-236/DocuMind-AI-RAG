"""
Document Explorer: inspect indexed documents, chunks, and metadata; search the knowledge base.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.state import init_session_state  # noqa: E402
from app.styles import (  # noqa: E402
    card,
    empty_state,
    footer,
    inject_css,
    page_header,
    section,
    fact_list,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Explorer | DocuMind AI",
    page_icon="🗂️",
    layout="wide",
)

init_session_state()
inject_css()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

page_header(
    "Workspace files",
    "Document Explorer",
    "Browse indexed files, inspect their extracted text, and search for a phrase or topic.",
)


# ---------------------------------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------------------------------

kb = st.session_state.kb
chunks = kb.all_chunks()

if not chunks:
    empty_state(
        "No documents indexed yet",
        "Your knowledge base is currently empty. "
        "Upload and index documents from the main chat page's sidebar "
        "to start exploring them here.",
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

section("Library overview")

fact_list(
    [
        ("Documents", total_documents),
        ("Extracted passages", total_chunks),
        ("Pages with text", total_pages),
        ("File formats", len(document_types)),
    ]
)


# ---------------------------------------------------------
# INDEXED DOCUMENTS
# ---------------------------------------------------------

section("Files in this workspace")

st.caption(
    f"{total_documents} document(s) are indexed. "
    "Open a file to inspect its extracted sections and metadata."
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
        f"{doc_name}  ·  {doc_type.upper()}  ·  "
        f"{len(doc_chunks)} chunks"
    )

    if pages:
        expander_title += f"  ·  {len(pages)} pages"

    with st.expander(expander_title):

        card(
            doc_name,
            meta=(
                f"Type: {doc_type.upper()}  ·  "
                f"Chunks: {len(doc_chunks)}  ·  "
                f"Pages: {len(pages) if pages else 'Not available'}"
            ),
        )

        st.caption(
            f"Document ID: {document_id}  ·  Index status: Indexed"
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

section("Search the library")

card(
    "Find a passage",
    meta=(
        "Search by keyword, phrase, or question to find matching passages."
    ),
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

            card(
                f"{index}. {chunk.document_name}",
                body=preview,
                meta=(
                    f"{location}  ·  Chunk {chunk.chunk_index}  ·  "
                    f"BM25 score: {score:.3f}"
                ),
                title_accent=True,
            )

else:

    empty_state(
        "Ready to search",
        "Enter a search query above to find relevant chunks "
        "from your indexed documents.",
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

footer("DocuMind AI · Document library")