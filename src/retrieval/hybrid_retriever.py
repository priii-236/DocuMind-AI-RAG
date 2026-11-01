"""
HybridRetriever: combines dense (embedding) and sparse (BM25) retrieval with
a configurable weighted-fusion score:

    final_score = alpha * normalized_dense_score + (1 - alpha) * normalized_sparse_score

Supports multi-query retrieval (one call per expanded query, results merged
and de-duplicated by chunk_id, keeping the best score per chunk).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.embeddings.embedding_service import BaseEmbeddingService
from src.ingestion.document import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore


@dataclass
class RetrievalCandidate:
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0

    def to_debug_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_name": self.chunk.document_name,
            "page_number": self.chunk.page_number,
            "section": self.chunk.section,
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "fused_score": round(self.fused_score, 4),
            "text_preview": self.chunk.text[:220],
        }


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever, embedding_service: BaseEmbeddingService):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedding_service = embedding_service

    def retrieve_single(self, query: str, top_k: int, alpha: float) -> tuple[list[RetrievalCandidate], list[dict], list[dict]]:
        """Runs one query through dense + sparse retrieval and fuses the results.

        Returns (fused_candidates, dense_debug, sparse_debug).
        """
        query_embedding = self.embedding_service.embed_query(query)
        dense_hits = self.vector_store.search(query_embedding, top_k=top_k)
        sparse_hits = self.bm25.search(query, top_k=top_k)

        dense_by_id = {c.chunk_id: (c, s) for c, s in dense_hits}
        sparse_by_id = {c.chunk_id: (c, s) for c, s in sparse_hits}

        dense_norm = _minmax_normalize({cid: s for cid, (_, s) in dense_by_id.items()})
        sparse_norm = _minmax_normalize({cid: s for cid, (_, s) in sparse_by_id.items()})

        all_ids = set(dense_by_id) | set(sparse_by_id)
        candidates: list[RetrievalCandidate] = []
        for cid in all_ids:
            chunk = (dense_by_id.get(cid) or sparse_by_id.get(cid))[0]
            d = dense_norm.get(cid, 0.0)
            s = sparse_norm.get(cid, 0.0)
            fused = alpha * d + (1 - alpha) * s
            candidates.append(RetrievalCandidate(chunk=chunk, dense_score=d, sparse_score=s, fused_score=fused))

        candidates.sort(key=lambda c: c.fused_score, reverse=True)

        dense_debug = [{"chunk_id": c.chunk_id, "document_name": c.document_name, "score": round(s, 4)} for c, s in dense_hits]
        sparse_debug = [{"chunk_id": c.chunk_id, "document_name": c.document_name, "score": round(s, 4)} for c, s in sparse_hits]

        return candidates, dense_debug, sparse_debug

    def retrieve_multi(self, queries: list[str], top_k: int, alpha: float) -> tuple[list[RetrievalCandidate], list[dict], list[dict]]:
        """Runs multiple (expanded) queries, merges + de-duplicates by chunk_id keeping the best score."""
        best: dict[str, RetrievalCandidate] = {}
        all_dense_debug: list[dict] = []
        all_sparse_debug: list[dict] = []

        for q in queries:
            candidates, dense_debug, sparse_debug = self.retrieve_single(q, top_k=top_k, alpha=alpha)
            all_dense_debug.extend(dense_debug)
            all_sparse_debug.extend(sparse_debug)
            for c in candidates:
                existing = best.get(c.chunk.chunk_id)
                if existing is None or c.fused_score > existing.fused_score:
                    best[c.chunk.chunk_id] = c

        merged = sorted(best.values(), key=lambda c: c.fused_score, reverse=True)
        return merged, all_dense_debug, all_sparse_debug
