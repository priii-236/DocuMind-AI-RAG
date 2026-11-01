
"""Retrieval Debugger: step through the full advanced RAG pipeline for a single query."""

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
    page_title="Retrieval Debugger | DocuMind AI",
    page_icon="🛠️",
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
        line-height: 1.7;
        margin: 0;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 700;
        color: #f1f2ff;
        margin-top: 25px;
        margin-bottom: 14px;
    }

    .info-card {
        background: linear-gradient(145deg, #171e34, #1c2340);
        border: 1px solid rgba(150, 160, 220, 0.16);
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 105px;
    }

    .info-label {
        color: #9da9c3;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-value {
        color: #ffffff;
        font-size: 25px;
        font-weight: 800;
        margin-top: 9px;
    }

    .query-card {
        background: linear-gradient(145deg, #191d3b, #20274a);
        border: 1px solid rgba(160, 150, 255, 0.20);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }

    .query-label {
        color: #a99aff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .query-text {
        color: #f3f1ff;
        font-size: 16px;
        font-weight: 600;
        line-height: 1.6;
        word-break: break-word;
    }

    .stage-card {
        background: linear-gradient(145deg, #171e34, #1b2340);
        border: 1px solid rgba(150, 160, 220, 0.17);
        border-radius: 14px;
        padding: 18px 20px;
        margin: 12px 0;
    }

    .stage-title {
        color: #d0c5ff;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .stage-description {
        color: #aab5ce;
        font-size: 12px;
        line-height: 1.6;
    }

    .answer-card {
        background: linear-gradient(145deg, #191d3b, #202747);
        border: 1px solid rgba(160, 150, 255, 0.22);
        border-radius: 17px;
        padding: 24px;
        margin-top: 15px;
    }

    .answer-label {
        color: #b9a8ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .answer-text {
        color: #e4e7f5;
        font-size: 14px;
        line-height: 1.9;
        white-space: pre-wrap;
        word-break: break-word;
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

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7357dc, #4d75d9);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 25px;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #8468ec, #5c86ec);
        box-shadow: 0 5px 18px rgba(100, 90, 220, 0.25);
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
        <div class="hero-tag">DOCUMIND AI · PIPELINE INSPECTION</div>
        <div class="hero-title">🛠️ Retrieval Debugger</div>
        <p class="hero-description">
            Inspect every stage of your advanced RAG pipeline.
            Understand how queries are transformed, how documents are retrieved,
            how results are reranked, and what context reaches the LLM.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# KNOWLEDGE BASE CHECK
# ---------------------------------------------------------

kb_stats = st.session_state.kb.stats()

if kb_stats["num_chunks"] == 0:

    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:35px;">📂</div>
            <h3>Knowledge Base is Empty</h3>
            <p>
                Upload and index documents from the main chat page
                before running the retrieval debugger.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ---------------------------------------------------------
# PIPELINE OVERVIEW
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">⚙️ Pipeline Configuration</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Indexed Chunks</div>
            <div class="info-value">
                """ + str(kb_stats["num_chunks"]) + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    rewriting_status = (
        "Enabled"
        if st.session_state.enable_query_rewriting
        else "Disabled"
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Query Rewriting</div>
            <div class="info-value" style="font-size:21px;">
                {rewriting_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    reranking_status = (
        "Enabled"
        if st.session_state.enable_reranking
        else "Disabled"
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Reranking</div>
            <div class="info-value" style="font-size:21px;">
                {reranking_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# QUERY INPUT
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">🔍 Test Your RAG Pipeline</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Enter a question to inspect how the retrieval system processes it."
)

query = st.text_input(
    "Debug Query",
    placeholder="e.g. What is the objective of my internship project?",
    key="retrieval_debugger_query",
)

run = st.button(
    "🚀 Run Pipeline",
    type="primary",
    use_container_width=False,
)


# ---------------------------------------------------------
# DEBUG TABLE HELPER
# ---------------------------------------------------------

def _debug_table(rows: list[dict], stage_name: str) -> None:

    if not rows:
        st.caption(f"No results available for {stage_name}.")
        return

    table = pd.DataFrame(rows)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# PIPELINE EXECUTION
# ---------------------------------------------------------

if run and not query.strip():

    st.warning("Please enter a query before running the pipeline.")

elif run and query.strip():

    if st.session_state.pipeline is None:

        st.error(
            f"LLM is not configured: {st.session_state.llm_error}. "
            "Please check your LLM settings. Retrieval stages require "
            "a successful pipeline response to display the trace."
        )

    else:

        try:

            with st.spinner(
                "🔄 Running retrieval, ranking, and answer generation..."
            ):

                response = st.session_state.pipeline.answer(
                    query.strip(),
                    memory=None,
                    top_k=st.session_state.top_k,
                    alpha=st.session_state.alpha,
                    enable_query_rewriting=(
                        st.session_state.enable_query_rewriting
                    ),
                    enable_reranking=(
                        st.session_state.enable_reranking
                    ),
                )

            trace = response.trace if response else None

            if trace is None:

                st.warning(
                    "The pipeline completed, but no debug trace was returned."
                )

            else:

                # -----------------------------------------
                # ORIGINAL QUERY
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">1️⃣ Original Query</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="query-card">
                        <div class="query-label">User Input</div>
                        <div class="query-text">{trace.original_query}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # -----------------------------------------
                # REWRITTEN QUERIES
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">2️⃣ Rewritten / Expanded Queries</div>',
                    unsafe_allow_html=True,
                )

                if trace.rewritten_queries:

                    for index, rewritten_query in enumerate(
                        trace.rewritten_queries,
                        start=1,
                    ):

                        st.markdown(
                            f"""
                            <div class="stage-card">
                                <div class="stage-title">
                                    Query Variation {index}
                                </div>
                                <div class="stage-description">
                                    {rewritten_query}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:

                    st.caption(
                        "No rewritten queries generated. "
                        "Original query was used."
                    )


                # -----------------------------------------
                # DENSE RETRIEVAL
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">3️⃣ Dense Retrieval Results</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Semantic search using vector embeddings."
                )

                _debug_table(
                    trace.dense_results,
                    "Dense Retrieval",
                )


                # -----------------------------------------
                # SPARSE RETRIEVAL
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">4️⃣ Sparse (BM25) Results</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Keyword-based retrieval using BM25 scoring."
                )

                _debug_table(
                    trace.sparse_results,
                    "Sparse Retrieval",
                )


                # -----------------------------------------
                # MERGED RESULTS
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">5️⃣ Merged / Fused Results</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Combined results from dense and sparse retrieval."
                )

                _debug_table(
                    trace.merged_results,
                    "Merged Results",
                )


                # -----------------------------------------
                # RERANKED RESULTS
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">6️⃣ Reranked Results</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Final ranking after relevance-based reranking."
                )

                _debug_table(
                    trace.reranked_results,
                    "Reranked Results",
                )


                # -----------------------------------------
                # FINAL CONTEXT
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">7️⃣ Final Context Sent to LLM</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "These are the final context chunks used to generate the answer."
                )

                if trace.final_context_chunks:

                    for index, chunk in enumerate(
                        trace.final_context_chunks,
                        start=1,
                    ):

                        with st.expander(
                            f"📄 Context Chunk {index}",
                            expanded=(index == 1),
                        ):

                            st.json(chunk)

                else:

                    st.info("No final context chunks available.")


                # -----------------------------------------
                # GENERATED ANSWER
                # -----------------------------------------

                st.divider()

                st.markdown(
                    '<div class="section-heading">✨ Generated Answer</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-label">AI Generated Response</div>
                        <div class="answer-text"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(trace.answer)


                # -----------------------------------------
                # PERFORMANCE METRICS
                # -----------------------------------------

                st.markdown(
                    '<div class="section-heading">📈 Pipeline Performance</div>',
                    unsafe_allow_html=True,
                )

                metric1, metric2, metric3 = st.columns(3)

                with metric1:

                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="info-label">Confidence</div>
                            <div class="info-value">
                                {trace.confidence:.3f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with metric2:

                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="info-label">Latency</div>
                            <div class="info-value">
                                {trace.latency_seconds:.2f}s
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with metric3:

                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="info-label">Pipeline Mode</div>
                            <div class="info-value" style="font-size:19px;">
                                {trace.pipeline_mode}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                # -----------------------------------------
                # DEBUG SUMMARY
                # -----------------------------------------

                st.divider()

                st.markdown(
                    '<div class="section-heading">📋 Debug Summary</div>',
                    unsafe_allow_html=True,
                )

                summary = pd.DataFrame(
                    [
                        {
                            "Pipeline Stage": "Dense Retrieval",
                            "Retrieved Results": len(trace.dense_results),
                        },
                        {
                            "Pipeline Stage": "Sparse Retrieval",
                            "Retrieved Results": len(trace.sparse_results),
                        },
                        {
                            "Pipeline Stage": "Merged Results",
                            "Retrieved Results": len(trace.merged_results),
                        },
                        {
                            "Pipeline Stage": "Reranked Results",
                            "Retrieved Results": len(trace.reranked_results),
                        },
                        {
                            "Pipeline Stage": "Final Context",
                            "Retrieved Results": len(
                                trace.final_context_chunks
                            ),
                        },
                    ]
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:

            st.error(
                f"Pipeline execution failed: {exc}"
            )

            st.exception(exc)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px 0;">
        <span style="color:#8290ad; font-size:12px;">
            DocuMind AI · Advanced RAG Pipeline Debugging
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
