"""
RerankerService: second-stage cross-encoder reranking over the hybrid
retrieval candidate pool. Swappable so a different reranker (e.g. a hosted
API-based one) can be dropped in without touching pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from src.config.settings import settings
from src.observability.logger import logger
from src.retrieval.hybrid_retriever import RetrievalCandidate


class BaseRerankerService(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        ...


class CrossEncoderRerankerService(BaseRerankerService):
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        logger.info("Loading reranker model '%s' ...", self.model_name)
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        raw_scores = self._model.predict(pairs)

        # Rescale to 0..1 for easy comparison against fused_score / confidence thresholds.
        lo, hi = float(min(raw_scores)), float(max(raw_scores))
        span = hi - lo if hi - lo > 1e-9 else 1.0

        reranked = []
        for candidate, raw in zip(candidates, raw_scores):
            candidate.fused_score = (float(raw) - lo) / span  # overwrite with reranker score for downstream use
            reranked.append(candidate)

        reranked.sort(key=lambda c: c.fused_score, reverse=True)
        return reranked[:top_k]


class NoOpRerankerService(BaseRerankerService):
    """Used when reranking is disabled - simply truncates the already-fused-score-sorted list."""

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        return candidates[:top_k]


@lru_cache(maxsize=1)
def get_reranker_service(enabled: bool = True) -> BaseRerankerService:
    if not enabled:
        return NoOpRerankerService()
    try:
        return CrossEncoderRerankerService()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker model unavailable (%s); falling back to no-op reranker.", exc)
        return NoOpRerankerService()
