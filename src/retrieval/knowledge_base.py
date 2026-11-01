"""
KnowledgeBase: orchestrates ingestion -> chunking -> embedding -> indexing,
and owns the live VectorStore + BM25Retriever instances for the session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.chunking.chunker import chunk_documents
from src.config.settings import settings
from src.embeddings.embedding_service import BaseEmbeddingService, get_embedding_service
from src.ingestion.document import Chunk
from src.ingestion.loader import load_document
from src.observability.logger import logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore, get_vector_store


@dataclass
class IngestResult:
    filename: str
    success: bool
    num_chunks: int = 0
    error: str | None = None


class KnowledgeBase:
    def __init__(self, embedding_service: BaseEmbeddingService | None = None):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store: VectorStore = get_vector_store(self.embedding_service.dimension)
        self.bm25 = BM25Retriever()
        self._rebuild_bm25()
        self.document_registry: dict[str, str] = {}  # document_id -> filename

    def _rebuild_bm25(self) -> None:
        self.bm25.build(self.vector_store.all_chunks())

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str | None = None,
    ) -> IngestResult:
        try:
            raw_docs = load_document(file_bytes, filename, max_size_mb=settings.max_file_size_mb)
            chunks: list[Chunk] = chunk_documents(
                raw_docs,
                chunk_size=chunk_size or settings.chunk_size,
                chunk_overlap=chunk_overlap or settings.chunk_overlap,
                strategy=strategy or settings.chunking_strategy,
            )
            if not chunks:
                return IngestResult(filename=filename, success=False, error="No content could be chunked from this file.")

            embeddings = self.embedding_service.embed_texts([c.text for c in chunks])
            self.vector_store.add(chunks, embeddings)
            self._rebuild_bm25()
            self.document_registry[chunks[0].document_id] = filename

            logger.info("Indexed '%s' -> %d chunks", filename, len(chunks))
            return IngestResult(filename=filename, success=True, num_chunks=len(chunks))
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion failed for '%s': %s", filename, exc)
            return IngestResult(filename=filename, success=False, error=str(exc))

    def ingest_files(self, files: list[tuple[bytes, str]], **kwargs) -> list[IngestResult]:
        results = [self.ingest_file(data, name, **kwargs) for data, name in files]
        self.persist()
        return results

    def persist(self) -> None:
        self.vector_store.persist()

    def clear(self) -> None:
        self.vector_store.clear()
        self._rebuild_bm25()
        self.document_registry = {}
        self.persist()

    def stats(self) -> dict:
        chunks = self.vector_store.all_chunks()
        docs = {c.document_id for c in chunks}
        return {
            "num_documents": len(docs),
            "num_chunks": len(chunks),
            "document_names": sorted({c.document_name for c in chunks}),
        }

    def all_chunks(self) -> list[Chunk]:
        return self.vector_store.all_chunks()
