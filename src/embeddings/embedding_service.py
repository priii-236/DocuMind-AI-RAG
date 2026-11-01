"""
EmbeddingService: a swappable abstraction over the embedding provider.

Default implementation uses a local sentence-transformers model so the
project runs with zero API cost. The interface is deliberately narrow so a
different provider (OpenAI, Cohere, Voyage, etc.) can be dropped in later
without touching any retrieval code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from src.config.settings import settings
from src.observability.logger import logger


class BaseEmbeddingService(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Local, free, CPU-friendly embedding model (default provider)."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        logger.info("Loading embedding model '%s' ...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim


@lru_cache(maxsize=1)
def get_embedding_service() -> BaseEmbeddingService:
    """Cached singleton so the (relatively slow) model load only happens once per process."""
    return SentenceTransformerEmbeddingService()
