"""Retrieval Debugger: step through the full advanced RAG pipeline for a single query."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.state import init_session_state  # noqa: E402

from app.styles import (
    card,
    empty_state,
    footer,
    inject_css,
    page_header,
    section,
    fact_list,
    metric_strip,
)



st.set_page_config(
    page_title="Retrieval Debugger | DocuMind AI",
    page_icon="🛠️",
    layout="wide",
)

init_session_state()
inject_css()


page_header(
    "Search diagnostics",
    "Retrieval Debugger",
    "Run a test question and inspect how results are selected before an answer is generated.",
    pill="Developer view",
)

kb_stats = st.session_state.kb.stats()

if kb_stats["num_chunks"] == 0:
    empty_state(
        "Knowledge base is empty",
        "Upload and index documents from the main chat page before running "
        "the retrieval debugger.",
    )
    st.stop()


section("Current search settings")

fact_list(
    [
        ("Indexed passages", kb_stats["num_chunks"]),
        (
            "Query rewriting",
            "Enabled" if st.session_state.enable_query_rewriting else "Disabled",
        ),
        (
            "Reranking",
            "Enabled" if st.session_state.enable_reranking else "Disabled",
        ),
    ]
)


section("Run a test search")
st.caption("Enter a question to see each step used to find supporting passages.")

query = st.text_input(
    "Debug query",
    placeholder="e.g. What is the objective of my internship project?",
    key="retrieval_debugger_query",
)

run = st.button("Run pipeline", type="primary")


def _debug_table(rows: list[dict], stage_name: str) -> None:
    if not rows:
        st.caption(f"No results available for {stage_name}.")
        return

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


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
            with st.spinner("Running retrieval, ranking, and answer generation..."):
                response = st.session_state.pipeline.answer(
                    query.strip(),
                    memory=None,
                    top_k=st.session_state.top_k,
                    alpha=st.session_state.alpha,
                    enable_query_rewriting=(
                        st.session_state.enable_query_rewriting
                    ),
                    enable_reranking=st.session_state.enable_reranking,
                )

            trace = response.trace if response else None

            if trace is None:
                st.warning(
                    "The pipeline completed, but no debug trace was returned."
                )
            else:
                section("1. Question received")
                card(
                    title="User input",
                    body=trace.original_query,
                    accent=True,
                )

                section("2. Rewritten / expanded queries")
                if trace.rewritten_queries:
                    for index, rewritten_query in enumerate(
                        trace.rewritten_queries,
                        start=1,
                    ):
                        card(
                            title=f"Search wording {index}",
                            body=rewritten_query,
                        )
                else:
                    st.caption(
                        "No rewritten queries generated. Original query was used."
                    )

                section("3. Meaning-based matches")
                st.caption("Matches found by comparing the meaning of the question with each passage.")
                _debug_table(trace.dense_results, "Dense retrieval")

                section("4. Keyword matches")
                st.caption("Matches found by comparing the question's words with each passage.")
                _debug_table(trace.sparse_results, "Sparse retrieval")

                section("5. Combined results")
                st.caption("Meaning-based and keyword matches brought into one list.")
                _debug_table(trace.merged_results, "Merged results")

                section("6. Final ranking")
                st.caption("The combined matches ordered by estimated relevance.")
                _debug_table(trace.reranked_results, "Reranked results")

                section("7. Passages used for the answer")
                st.caption(
                    "These are the final context chunks used to generate the answer."
                )

                if trace.final_context_chunks:
                    for index, chunk in enumerate(
                        trace.final_context_chunks,
                        start=1,
                    ):
                        with st.expander(
                            f"Context chunk {index}",
                            expanded=index == 1,
                        ):
                            st.json(chunk)
                else:
                    st.info("No final context chunks available.")

                section("Answer")
                card(label="Response")
                st.markdown(trace.answer)

                section("Pipeline performance")
                metric_strip(
                    [
                        ("Confidence", f"{trace.confidence:.3f}"),
                        ("Response time", f"{trace.latency_seconds:.2f}s"),
                        ("Search mode", trace.pipeline_mode),
                    ],
                    compact=True,
                )

                section("Debug summary")
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
            st.error(f"Pipeline execution failed: {exc}")
            st.exception(exc)


footer(
    "DocuMind AI · Search diagnostics",
    "Designed & Developed by Makwana Priyanka",
)