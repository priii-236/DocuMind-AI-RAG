"""
DocuMind AI — Analytics Dashboard
Query performance, retrieval insights and RAG experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import init_session_state
from app.styles import card, footer, inject_css, metric_strip, page_header, section
from src.config.settings import settings
from src.observability.logger import trace_store

st.set_page_config(
    page_title="Analytics | DocuMind AI",
    page_icon="📊",
    layout="wide",
)

init_session_state()
inject_css()

# ---------------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------------

_GRID = "rgba(255,255,255,.07)"


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#D2D6E0",
            size=13,
        ),
        title_font=dict(size=15, color="#E8E6E1"),
        colorway=["#B8A27A", "#8EA1A8", "#C8C8C4", "#78B695"],
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#D2D6E0", size=13),
        ),
        xaxis=dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            linecolor=_GRID,
        ),
        yaxis=dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            linecolor=_GRID,
        ),
    )
    return fig


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

page_header(
    "Workspace activity",
    "Analytics Dashboard",
    "Review how the workspace is being used, how quickly it responds, "
    "and how often answers include supporting evidence.",
)

traces = trace_store.recent(500)
kb_stats = st.session_state.kb.stats()

successful = [t for t in traces if not t.error]

avg_latency = (
    sum(t.latency_seconds for t in successful) / len(successful)
    if successful else 0.0
)

avg_confidence = (
    sum(t.confidence for t in successful) / len(successful)
    if successful else 0.0
)

grounded = [
    t for t in successful
    if t.confidence >= settings.min_confidence_threshold
]

success_rate = (
    len(grounded) / len(successful) * 100
    if successful else 0
)

# ---------------------------------------------------------
# OVERVIEW METRICS
# ---------------------------------------------------------

section("Performance overview")

metric_strip(
    [
        ("Documents", kb_stats["num_documents"]),
        ("Indexed passages", kb_stats["num_chunks"]),
        ("Questions", len(traces)),
        ("Avg. response", f"{avg_latency:.2f}s"),
        ("Avg. confidence", f"{avg_confidence:.2f}"),
        ("Supported answers", f"{success_rate:.1f}%"),
    ]
)

# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------

section("Search and response activity")

if not traces:

    st.info(
        "No analytics data available yet. "
        "Start asking questions in the main chat to populate your dashboard."
    )

else:

    df = pd.DataFrame(
        [
            {
                "trace_id": t.trace_id,
                "query": t.original_query,
                "mode": t.pipeline_mode,
                "latency": t.latency_seconds,
                "confidence": t.confidence,
                "num_sources": len(t.final_context_chunks),
                "timestamp": pd.to_datetime(
                    t.timestamp, unit="s"
                ),
                "status": "Failed" if t.error else "Successful",
            }
            for t in traces
        ]
    )

    left, right = st.columns(2)

    with left:

        section("Query volume over time")

        volume = (
            df.set_index("timestamp")
            .resample("1min")
            .size()
            .reset_index(name="queries")
        )

        fig = px.bar(
            volume,
            x="timestamp",
            y="queries",
            title="Queries per minute",
        )

        fig.update_traces(
            marker_color="#B8A27A",
            marker_line_width=0,
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        section("Response latency")

        fig = px.histogram(
            df,
            x="latency",
            nbins=20,
            title="Response time distribution",
        )

        fig.update_traces(
            marker_color="#B8A27A",
            marker_line_width=0,
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    with right:

        section("Retrieval confidence")

        fig = px.histogram(
            df,
            x="confidence",
            nbins=20,
            title="Confidence score distribution",
        )

        fig.update_traces(
            marker_color="#B8A27A",
            marker_line_width=0,
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        section("Top retrieved documents")

        document_counts = {}

        for trace in traces:
            for chunk in trace.final_context_chunks:
                name = chunk.get("document_name", "Unknown")
                document_counts[name] = (
                    document_counts.get(name, 0) + 1
                )

        if document_counts:

            doc_df = pd.DataFrame(
                sorted(
                    document_counts.items(),
                    key=lambda x: -x[1],
                ),
                columns=["Document", "Times Retrieved"],
            )

            fig = px.bar(
                doc_df,
                x="Document",
                y="Times Retrieved",
                title="Most referenced documents",
            )

            fig.update_traces(
                marker_color="#B8A27A",
                marker_line_width=0,
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True,
            )

        else:
            st.caption("No document retrieval records yet.")

# ---------------------------------------------------------
# QUERY HISTORY
# ---------------------------------------------------------

section("Recent query activity")

if traces:

    recent_df = df.sort_values(
        "timestamp", ascending=False
    ).head(15)

    st.dataframe(
        recent_df[
            [
                "timestamp",
                "query",
                "mode",
                "latency",
                "confidence",
                "num_sources",
                "status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.caption("Query history will appear here.")

# ---------------------------------------------------------
# EXPERIMENT COMPARISON
# ---------------------------------------------------------

st.divider()

section("Retrieval comparison")

card(
    body=(
        "Compare dense retrieval with the hybrid search pipeline using Precision@K, Recall@K, MRR "
        "and NDCG@K evaluation metrics."
    ),
    label="Evaluation results",
)

exp_path = settings.experiments_dir / "results.csv"

if exp_path.exists():

    exp_df = pd.read_csv(exp_path)

    st.dataframe(
        exp_df,
        use_container_width=True,
        hide_index=True,
    )

    for exp_name in exp_df["experiment"].dropna().unique():

        subset = exp_df[
            exp_df["experiment"] == exp_name
        ]

        st.markdown(
            f"**{exp_name.replace('_', ' ').title()}**"
        )

        metric_cols = [
            col for col in [
                "precision_at_k",
                "recall_at_k",
                "mrr",
                "ndcg_at_k",
            ]
            if col in subset.columns
        ]

        if metric_cols:

            melted = subset.melt(
                id_vars=["config"],
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                melted,
                x="config",
                y="Score",
                color="Metric",
                barmode="group",
                title=f"{exp_name} comparison",
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True,
            )

else:

    st.info(
        "Experiment results are not available yet. "
        "Run `python -m experiments.run_experiments` "
        "after preparing the sample evaluation data."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

footer(
    "DocuMind AI · Workspace activity",
    "Designed & Developed by Makwana Priyanka",
)