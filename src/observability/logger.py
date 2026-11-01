"""
Structured logging + lightweight query trace collection for DocuMind AI.

Never logs API keys or credentials. Traces are kept in-memory (and optionally
appended to a JSONL file) so the Streamlit Analytics + Retrieval Debugger
pages can read them back.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config.settings import settings

_SENSITIVE_KEYS = {"api_key", "apikey", "token", "authorization", "secret"}


def _redact(d: dict) -> dict:
    return {k: ("***REDACTED***" if k.lower() in _SENSITIVE_KEYS else v) for k, v in d.items()}


def get_logger(name: str = "documind") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        try:
            file_handler = logging.FileHandler(settings.logs_dir / "documind.log")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            pass  # read-only filesystem (e.g. some cloud deployments) - console logging still works
    return logger


logger = get_logger()


@dataclass
class QueryTrace:
    """Full observability record for a single question answered by the pipeline."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    original_query: str = ""
    rewritten_queries: list[str] = field(default_factory=list)
    dense_results: list[dict] = field(default_factory=list)
    sparse_results: list[dict] = field(default_factory=list)
    merged_results: list[dict] = field(default_factory=list)
    reranked_results: list[dict] = field(default_factory=list)
    final_context_chunks: list[dict] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    pipeline_mode: str = "advanced"  # "baseline" | "advanced"
    latency_seconds: float = 0.0
    token_usage: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class TraceStore:
    """In-memory + append-only JSONL store of query traces for the current session."""

    def __init__(self) -> None:
        self.traces: list[QueryTrace] = []
        self._log_path = settings.logs_dir / "traces.jsonl"

    def add(self, trace: QueryTrace) -> None:
        self.traces.append(trace)
        logger.info(
            "query trace | id=%s | mode=%s | latency=%.2fs | confidence=%.2f",
            trace.trace_id,
            trace.pipeline_mode,
            trace.latency_seconds,
            trace.confidence,
        )
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_redact(trace.to_dict())) + "\n")
        except OSError:
            pass

    def recent(self, n: int = 50) -> list[QueryTrace]:
        return self.traces[-n:]

    def clear(self) -> None:
        self.traces = []


trace_store = TraceStore()
