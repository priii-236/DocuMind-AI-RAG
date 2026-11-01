"""Shared Streamlit session-state initialization used by every page."""
from __future__ import annotations

import streamlit as st

from src.config.settings import settings
from src.generation.llm_client import LLMClient, LLMConfigurationError
from src.generation.rag_pipeline import RAGPipeline
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.knowledge_base import KnowledgeBase


def init_session_state() -> None:
    if "kb" not in st.session_state:
        with st.spinner("Loading embedding model and knowledge base..."):
            st.session_state.kb = KnowledgeBase()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[{"role", "content", "evidence"?}]

    if "llm_error" not in st.session_state:
        st.session_state.llm_error = None

    if "pipeline" not in st.session_state:
        try:
            llm = LLMClient()
            st.session_state.pipeline = RAGPipeline(st.session_state.kb, llm_client=llm)
            st.session_state.llm_error = None
        except LLMConfigurationError as exc:
            st.session_state.pipeline = None
            st.session_state.llm_error = str(exc)

    # Retrieval / generation settings (sidebar-controlled)
    st.session_state.setdefault("top_k", settings.retrieval_top_k)
    st.session_state.setdefault("alpha", settings.hybrid_alpha)
    st.session_state.setdefault("enable_reranking", settings.enable_reranking)
    st.session_state.setdefault("enable_query_rewriting", settings.enable_query_rewriting)
    st.session_state.setdefault("temperature", settings.llm_temperature)
    st.session_state.setdefault("max_tokens", settings.llm_max_tokens)
