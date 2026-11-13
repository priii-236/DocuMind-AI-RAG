"""Shared design system for every DocuMind AI page.

UI only. No application logic lives in this file.
Call `inject_css()` once at the top of each page (after `st.set_page_config`).
"""

from __future__ import annotations

from html import escape

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #101214;
    --surface: #17191D;
    --surface-2: #1D2025;
    --border: #292D33;
    --border-strong: #3A4048;
    --text: #E8E6E1;
    --text-2: #C8C8C4;
    --muted: #969CA5;
    --accent: #B8A27A;
    --accent-hover: #C9B58F;
    --accent-text: #D0BC94;
    --success: #78B695;
}

/* ---------- Base ---------- */

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}

/* Font is applied to text elements only, so Streamlit's icon fonts keep working */
.stApp :where(p, label, li, h1, h2, h3, h4, h5, h6, input, textarea, button, select, th, td, summary) {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1280px;
    padding-top: 2.25rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.01em;
}

[data-testid="stMarkdownContainer"] p {
    font-size: 15px;
    line-height: 1.65;
    color: var(--text-2);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--muted);
}

hr {
    border-color: var(--border) !important;
}

::placeholder {
    color: var(--muted) !important;
    opacity: 1 !important;
}

/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {
    background: #141619;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h3 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: none;
    color: var(--muted) !important;
    margin-bottom: 0.25rem;
}

[data-testid="stWidgetLabel"] p {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-2);
}

.brand {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.2;
}

.brand span {
    color: var(--accent);
}

.brand-sub {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: none;
    color: var(--muted);
    margin-top: 4px;
}

/* ---------- Buttons ---------- */

.stButton > button,
[data-testid="stBaseButton-secondary"] {
    background: var(--surface-2);
    border: 1px solid var(--border-strong);
    color: var(--text);
    border-radius: 10px;
    font-weight: 500;
    min-height: 2.75rem;
    height: auto;
    padding: 0.55rem 1rem;
    white-space: normal;
}

.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent);
    color: var(--text);
    background: var(--surface-2);
}

.stButton > button p {
    color: inherit;
    font-size: 14.5px;
    font-weight: 500;
    margin: 0;
}

.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--accent);
    border: 1px solid var(--accent);
    color: #17191D;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #17191D;
}

/* ---------- Inputs ---------- */

[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="select"] > div {
    background: var(--surface-2) !important;
    border-color: var(--border-strong) !important;
    border-radius: 10px !important;
}

[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    color: var(--text);
    font-size: 15px;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2);
    border: 1px dashed var(--border-strong);
    border-radius: 10px;
}

/* ---------- Metrics (sidebar st.metric) ---------- */

[data-testid="stMetric"] {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    font-size: 13px;
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
}

/* ---------- Expander ---------- */

[data-testid="stExpander"] {
    background: transparent;
    border: none;
}

[data-testid="stExpander"] details {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
}

[data-testid="stExpander"] summary p {
    font-size: 14.5px;
    font-weight: 500;
    color: var(--text);
}

/* ---------- Chat ---------- */

[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
}

[data-testid="stChatInput"] {
    background: var(--surface-2);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
}

[data-testid="stChatInput"] textarea {
    font-size: 15px;
    color: var(--text);
}

[data-testid="stBottom"] > div {
    background: var(--bg);
}

/* ---------- Alerts, tables, charts ---------- */

[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 14.5px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stPlotlyChart"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
}

/* ---------- Custom components ---------- */

.page-header {
    margin: 0 0 1.75rem 0;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.page-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: none;
    color: var(--accent);
    margin-bottom: 10px;
}

.page-title {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: var(--text);
    margin: 0 0 10px 0;
}

.page-subtitle {
    font-size: 16px;
    line-height: 1.65;
    color: var(--muted);
    max-width: 720px;
    margin: 0;
}

.page-header .pill {
    margin-top: 18px;
}

.section-heading {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
    margin: 1.75rem 0 0.8rem 0;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0;
    margin: 0.25rem 0 1.5rem;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}

.metric-item {
    padding: 14px 18px 15px 0;
    margin-right: 18px;
    border-right: 1px solid var(--border);
}

.metric-item:last-child {
    border-right: 0;
}

.metric-label {
    font-size: 12.5px;
    color: var(--muted);
    margin-bottom: 5px;
}

.metric-value {
    font-size: 22px;
    font-weight: 650;
    line-height: 1.25;
    color: var(--text);
    word-break: break-word;
}

.metric-value.small {
    font-size: 16px;
}

.fact-list {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0 28px;
    margin: 0.25rem 0 1.75rem;
    border-top: 1px solid var(--border);
}

.fact-item {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 13px 0;
    border-bottom: 1px solid var(--border);
}

.fact-label {
    color: var(--muted);
    font-size: 13px;
}

.fact-value {
    color: var(--text);
    font-size: 15px;
    font-weight: 600;
    text-align: right;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    margin: 0 0 0.55rem 0;
}

.card.accent {
    border-left: 2px solid var(--accent);
}

.card-label {
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: none;
    color: var(--accent);
    margin-bottom: 8px;
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    line-height: 1.45;
    color: var(--text);
    word-break: break-word;
}

.card-title.accent {
    color: var(--accent);
}

.card-meta {
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--muted);
    margin-top: 6px;
}

.card-body {
    font-size: 15px;
    line-height: 1.7;
    color: var(--text-2);
    margin-top: 10px;
    word-break: break-word;
}

.card-title + .card-body {
    margin-top: 6px;
}

.empty-state {
    background: var(--surface);
    border: 1px dashed var(--border-strong);
    border-radius: 8px;
    padding: 32px 24px;
    text-align: center;
    margin-top: 1.5rem;
}

.empty-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 8px;
}

.empty-text {
    font-size: 15px;
    line-height: 1.6;
    color: var(--muted);
    max-width: 480px;
    margin: 0 auto;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0;
    background: transparent;
    border: 0;
    color: #9AC9AD;
    font-size: 13px;
    font-weight: 500;
}

.pill::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
    padding: 2.5rem 0 0.5rem 0;
}
"""


def inject_css() -> None:
    """Inject the shared stylesheet. Call once per page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def page_header(label: str, title: str, subtitle: str, pill: str | None = None) -> None:
    pill_html = f'<div class="pill">{escape(pill)}</div>' if pill else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-label">{escape(label)}</div>
            <div class="page-title">{escape(title)}</div>
            <p class="page-subtitle">{escape(subtitle)}</p>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(
        f'<div class="section-heading">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, object]], compact: bool = False) -> None:
    entries = []
    for label, value in items:
        value_class = " small" if compact else ""
        entries.append(
            f'<div class="metric-item"><div class="metric-label">'
            f'{escape(str(label))}</div><div class="metric-value{value_class}">'
            f'{escape(str(value))}</div></div>'
        )
    st.markdown(
        f'<div class="metric-strip">{"".join(entries)}</div>',
        unsafe_allow_html=True,
    )


def fact_list(items: list[tuple[str, object]]) -> None:
    entries = []
    for label, value in items:
        entries.append(
            f'<div class="fact-item"><span class="fact-label">'
            f'{escape(str(label))}</span><span class="fact-value">'
            f'{escape(str(value))}</span></div>'
        )
    st.markdown(
        f'<div class="fact-list">{"".join(entries)}</div>',
        unsafe_allow_html=True,
    )


def card(
    title: str | None = None,
    body: str | None = None,
    meta: str | None = None,
    label: str | None = None,
    accent: bool = False,
    title_accent: bool = False,
) -> None:
    """Generic content card. All text is HTML-escaped."""
    parts = []
    if label:
        parts.append(f'<div class="card-label">{escape(label)}</div>')
    if title:
        cls = "card-title accent" if title_accent else "card-title"
        parts.append(f'<div class="{cls}">{escape(title)}</div>')
    if meta:
        parts.append(f'<div class="card-meta">{escape(meta)}</div>')
    if body:
        parts.append(f'<div class="card-body">{escape(body)}</div>')
    cls = "card accent" if accent else "card"
    st.markdown(
        f'<div class="{cls}">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-title">{escape(title)}</div>
            <div class="empty-text">{escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer(*lines: str) -> None:
    body = "<br>".join(escape(line) for line in lines)
    st.markdown(f'<div class="footer">{body}</div>', unsafe_allow_html=True)