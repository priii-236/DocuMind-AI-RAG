
"""
DocuMind AI — Premium Analytics Dashboard
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
from src.config.settings import settings
from src.observability.logger import trace_store

st.set_page_config(
    page_title="Analytics | DocuMind AI",
    page_icon="📊",
    layout="wide",
)

init_session_state()

# ---------------------------------------------------------
# PREMIUM THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(100,70,210,.13), transparent 35%),
            #080b16;
        color: #f3f3ff;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #10152a;
    }

    .page-label {
        color: #a78bfa;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .page-title {
        color: white;
        font-size: 35px;
        font-weight: 800;
        letter-spacing: -1.3px;
        margin-bottom: 5px;
    }

    .page-subtitle {
        color: #9ca8c5;
        font-size: 13px;
        margin-bottom: 28px;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a2039, #111629);
        border: 1px solid rgba(155,123,255,.19);
        padding: 19px;
        border-radius: 17px;
        min-height: 125px;
    }

    .metric-label {
        font-size: 12px;
        color: #a5aec7;
        margin-bottom: 12px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 28px;
        color: #ffffff;
        font-weight: 800;
    }

    .metric-icon {
        float: right;
        font-size: 22px;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 700;
        color: #f3f3ff;
        margin: 24px 0 14px 0;
    }

    .insight-card {
        background: rgba(22,29,52,.9);
        border: 1px solid rgba(155,123,255,.17);
        border-radius: 15px;
        padding: 17px;
        min-height: 105px;
    }

    .insight-title {
        color: #c4b5fd;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .insight-text {
        color: #c0c7e0;
        font-size: 13px;
        line-height: 1.7;
    }

    .footer {
        color: #77819e;
        font-size: 11px;
        text-align: center;
        padding: 25px 0 10px;
    }

    [data-testid="stPlotlyChart"] {
        background: rgba(19,25,46,.65);
        border: 1px solid rgba(155,123,255,.12);
        border-radius: 15px;
        padding: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPERS
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


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------------

def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color="#c4cbe0",
            size=11,
        ),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c4cbe0"),
        ),
        xaxis=dict(
            gridcolor="rgba(155,123,255,.10)",
            zerolinecolor="rgba(155,123,255,.10)",
        ),
        yaxis=dict(
            gridcolor="rgba(155,123,255,.10)",
            zerolinecolor="rgba(155,123,255,.10)",
        ),
    )
    return fig


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="page-label">DOCUMIND AI / INTELLIGENCE</div>
    <div class="page-title">Analytics Dashboard</div>
    <div class="page-subtitle">
        Monitor retrieval performance, response latency and AI quality.
        Turn your document interactions into measurable insights.
    </div>
    """,
    unsafe_allow_html=True,
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

section("✦ Performance Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("📚", "Documents Indexed", kb_stats["num_documents"])

with c2:
    metric_card("🧩", "Knowledge Chunks", kb_stats["num_chunks"])

with c3:
    metric_card("💬", "Queries Processed", len(traces))

with c4:
    metric_card("⚡", "Average Latency", f"{avg_latency:.2f}s")

st.write("")

c5, c6 = st.columns(2)

with c5:
    metric_card("🎯", "Average Confidence", f"{avg_confidence:.2f}")

with c6:
    metric_card("✅", "Retrieval Success Rate", f"{success_rate:.1f}%")

# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------

section("📈 Query & Retrieval Intelligence")

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

        section("Query Volume Over Time")

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
            marker_color="#8b5cf6",
            marker_line_width=0,
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        section("Response Latency")

        fig = px.histogram(
            df,
            x="latency",
            nbins=20,
            title="Response time distribution",
        )

        fig.update_traces(
            marker_color="#60a5fa",
            marker_line_width=0,
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

    with right:

        section("Retrieval Confidence")

        fig = px.histogram(
            df,
            x="confidence",
            nbins=20,
            title="Confidence score distribution",
        )

        fig.update_traces(
            marker_color="#a78bfa",
            marker_line_width=0,
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        section("Top Retrieved Documents")

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
                marker_color="#34d399",
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

section("📝 Recent Query Activity")

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

section("⚖️ Baseline vs Advanced RAG")

st.markdown(
    """
    <div class="insight-card">
        <div class="insight-title">EXPERIMENT INSIGHTS</div>
        <div class="insight-text">
            Compare traditional dense retrieval against the advanced
            hybrid RAG pipeline using Precision@K, Recall@K, MRR
            and NDCG@K evaluation metrics.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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

st.markdown(
    """
    <div class="footer">
        DOCUMIND AI · ANALYTICS INTELLIGENCE<br>
        Designed & Developed by Makwana Priyanka
    </div>
    """,
    unsafe_allow_html=True,
)
