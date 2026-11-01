"""Sparse lexical retrieval using BM25 (rank_bm25)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.ingestion.document import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if not self.is_ready():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        query_terms = set(tokenize(query))
        # On very small corpora, BM25's IDF term can go to zero/negative for common words,
        # so we don't filter on score > 0 - instead we only keep chunks that share at least
        # one token with the query, then rank by score.
        candidate_idx = [
            i for i in range(len(scores)) if query_terms & set(tokenize(self._chunks[i].text))
        ]
        ranked_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in ranked_idx]

    def count(self) -> int:
        return len(self._chunks)
