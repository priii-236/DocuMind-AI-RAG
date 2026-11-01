"""
VectorStore abstraction.

    VectorStore (ABC)
     |-- FAISSVectorStore   (local, zero-dependency default)
     +-- QdrantVectorStore  (optional, production-style)

Both implementations expose the same add() / search() / persist() / load()
interface so the rest of the pipeline never needs to know which backend is
active.
"""
from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from src.ingestion.document import Chunk
from src.observability.logger import logger


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def persist(self) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        ...


class FAISSVectorStore(VectorStore):
    """Local, dependency-light default. Uses flat inner-product search (embeddings are pre-normalized)."""

    def __init__(self, dimension: int, persist_dir: Path):
        import faiss

        self._faiss = faiss
        self.dimension = dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        assert embeddings.shape[0] == len(chunks), "embeddings/chunks length mismatch"
        self.index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)
        logger.info("FAISSVectorStore: added %d chunks (total=%d)", len(chunks), len(self._chunks))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.count())
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self._chunks = []

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def persist(self) -> None:
        self._faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))
        with open(self.persist_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("FAISSVectorStore: persisted %d chunks to %s", len(self._chunks), self.persist_dir)

    def load(self) -> None:
        index_path = self.persist_dir / "index.faiss"
        chunks_path = self.persist_dir / "chunks.pkl"
        if index_path.exists() and chunks_path.exists():
            self.index = self._faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info("FAISSVectorStore: loaded %d chunks", len(self._chunks))


class QdrantVectorStore(VectorStore):
    """Production-style backend. Requires QDRANT_URL (+ optional QDRANT_API_KEY) to be configured."""

    def __init__(self, dimension: int, url: str, api_key: str = "", collection: str = "documind"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dimension = dimension
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key or None)

        existing = [c.name for c in self.client.get_collections().collections]
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        self._local_chunk_cache: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload = chunk.to_metadata_dict()
            payload["text"] = chunk.text
            points.append(PointStruct(id=chunk.chunk_id, vector=emb.tolist(), payload=payload))
            self._local_chunk_cache[chunk.chunk_id] = chunk
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        hits = self.client.search(
            collection_name=self.collection, query_vector=query_embedding.tolist(), limit=top_k
        )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = Chunk(
                chunk_id=str(hit.id),
                text=payload.get("text", ""),
                document_id=payload.get("document_id", ""),
                document_name=payload.get("document_name", ""),
                document_type=payload.get("document_type", ""),
                source=payload.get("source", ""),
                page_number=payload.get("page_number"),
                section=payload.get("section"),
                chunk_index=payload.get("chunk_index", 0),
            )
            results.append((chunk, float(hit.score)))
        return results

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0

    def clear(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self.client.delete_collection(collection_name=self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )
        self._local_chunk_cache = {}

    def all_chunks(self) -> list[Chunk]:
        return list(self._local_chunk_cache.values())

    def persist(self) -> None:
        pass  # Qdrant persists server-side automatically.

    def load(self) -> None:
        pass  # Nothing to do; data already lives server-side.


def get_vector_store(dimension: int) -> VectorStore:
    """Factory: returns FAISS or Qdrant based on settings.vector_db, with automatic FAISS fallback."""
    from src.config.settings import settings

    if settings.vector_db == "qdrant" and settings.qdrant_url:
        try:
            return QdrantVectorStore(
                dimension=dimension,
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection=settings.qdrant_collection,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant unavailable (%s) - falling back to local FAISS.", exc)

    store = FAISSVectorStore(dimension=dimension, persist_dir=settings.vector_store_dir)
    store.load()
    return store
