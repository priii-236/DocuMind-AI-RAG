
"""
Central configuration for DocuMind AI.
Configuration can be loaded from local .env or Streamlit Cloud Secrets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


def _get(name: str, default=None):
    """Read environment variables first, then Streamlit Cloud Secrets."""
    value = os.getenv(name)

    if value is not None:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def _bool(name: str, default: bool) -> bool:
    val = _get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    val = _get(name)
    return int(val) if val not in (None, "") else default


def _float(name: str, default: float) -> float:
    val = _get(name)
    return float(val) if val not in (None, "") else default


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = _get("LLM_PROVIDER", "anthropic")
    llm_model: str = _get("LLM_MODEL", "claude-sonnet-4-5")
    api_key: str = _get("API_KEY", "")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.2)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 1024)

    # --- Embeddings ---
    embedding_model: str = _get(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 32)

    # --- Vector store ---
    vector_db: str = _get("VECTOR_DB", "faiss")
    qdrant_url: str = _get("QDRANT_URL", "")
    qdrant_api_key: str = _get("QDRANT_API_KEY", "")
    qdrant_collection: str = _get("QDRANT_COLLECTION", "documind")

    # --- Chunking ---
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120)
    chunking_strategy: str = _get("CHUNKING_STRATEGY", "recursive")

    # --- Retrieval ---
    retrieval_top_k: int = _int("RETRIEVAL_TOP_K", 8)
    candidate_pool_size: int = _int("CANDIDATE_POOL_SIZE", 25)
    hybrid_alpha: float = _float("HYBRID_ALPHA", 0.5)
    enable_query_rewriting: bool = _bool("ENABLE_QUERY_REWRITING", True)
    enable_reranking: bool = _bool("ENABLE_RERANKING", True)

    # --- Reranker ---
    reranker_model: str = _get(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # --- Hallucination control ---
    min_confidence_threshold: float = _float(
        "MIN_CONFIDENCE_THRESHOLD", 0.28
    )

    # --- File handling ---
    max_file_size_mb: int = _int("MAX_FILE_SIZE_MB", 25)
    allowed_extensions: tuple = (
        ".pdf", ".docx", ".txt", ".csv", ".md", ".html", ".htm"
    )

    # --- Paths ---
    vector_store_dir: Path = BASE_DIR / "vector_store"
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    experiments_dir: Path = BASE_DIR / "experiments"
    evaluation_dir: Path = BASE_DIR / "evaluation"
    logs_dir: Path = BASE_DIR / "logs"

    author: str = "Makwana Priyanka"
    app_name: str = "DocuMind AI"
    tagline: str = "Ask your documents. Get grounded answers with evidence."


settings = Settings()

for _dir in (
    settings.vector_store_dir,
    settings.data_raw_dir,
    settings.data_processed_dir,
    settings.experiments_dir,
    settings.evaluation_dir,
    settings.logs_dir,
):
    _dir.mkdir(parents=True, exist_ok=True)
