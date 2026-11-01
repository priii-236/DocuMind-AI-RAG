"""
Central configuration for DocuMind AI.
Every value is overridable via environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    api_key: str = os.getenv("API_KEY", "")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.2)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 1024)

    # --- Embeddings ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 32)

    # --- Vector store ---
    vector_db: str = os.getenv("VECTOR_DB", "faiss")  # "faiss" or "qdrant"
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documind")

    # --- Chunking ---
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120)
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "recursive")  # recursive | token | semantic

    # --- Retrieval ---
    retrieval_top_k: int = _int("RETRIEVAL_TOP_K", 8)
    candidate_pool_size: int = _int("CANDIDATE_POOL_SIZE", 25)
    hybrid_alpha: float = _float("HYBRID_ALPHA", 0.5)  # weight on dense score, (1-alpha) on sparse
    enable_query_rewriting: bool = _bool("ENABLE_QUERY_REWRITING", True)
    enable_reranking: bool = _bool("ENABLE_RERANKING", True)

    # --- Reranker ---
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Hallucination control ---
    min_confidence_threshold: float = _float("MIN_CONFIDENCE_THRESHOLD", 0.28)

    # --- File handling ---
    max_file_size_mb: int = _int("MAX_FILE_SIZE_MB", 25)
    allowed_extensions: tuple = (".pdf", ".docx", ".txt", ".csv", ".md", ".html", ".htm")

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
