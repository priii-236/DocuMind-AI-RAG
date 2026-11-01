"""
Configurable chunking strategies: recursive character, token-aware, and a
lightweight semantic strategy (splits on embedding-similarity breakpoints).

All strategies return a flat list[Chunk] with full provenance metadata carried
over from the parent RawDocument.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from src.ingestion.document import Chunk, RawDocument, make_id

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[str]:
    """Classic LangChain-style recursive character splitter, implemented from scratch."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > chunk_size and remaining_seps:
                chunks.extend(_recursive_split(part, chunk_size, overlap, remaining_seps))
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    # Apply overlap by carrying the tail of each chunk into the next.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped

    return [c.strip() for c in chunks if c.strip()]


def recursive_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    return _recursive_split(text, chunk_size, overlap, _SEPARATORS)


def token_aware_chunk_text(text: str, chunk_size: int, overlap: int, tokens_per_char: float = 0.25) -> list[str]:
    """
    Approximates token-based chunking without pulling in a heavy tokenizer:
    chunk_size is interpreted as tokens, converted to an approximate character
    budget (~4 chars/token for English), then recursively split.
    """
    approx_char_budget = int(chunk_size / max(tokens_per_char, 0.01))
    approx_char_overlap = int(overlap / max(tokens_per_char, 0.01))
    return recursive_chunk_text(text, approx_char_budget, approx_char_overlap)


def semantic_chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    embed_fn: Optional[Callable[[list[str]], "list[list[float]]"]] = None,
    similarity_drop_threshold: float = 0.35,
) -> list[str]:
    """
    Lightweight semantic chunking: split into sentences, embed each sentence,
    and cut a new chunk whenever cosine similarity to the running chunk drops
    sharply (a topic boundary) OR the size budget is exceeded.

    Falls back to recursive chunking if no embed_fn is supplied (e.g. to keep
    unit tests fast / avoid loading the embedding model unnecessarily).
    """
    if embed_fn is None:
        return recursive_chunk_text(text, chunk_size, overlap)

    import numpy as np

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = np.array(embed_fn(sentences))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = embeddings / norms

    chunks: list[str] = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = float(np.dot(unit[i], unit[i - 1]))
        sent_len = len(sentences[i])
        if sim < (1 - similarity_drop_threshold) or current_len + sent_len > chunk_size:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_len = sent_len
        else:
            current.append(sentences[i])
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


_STRATEGIES = {
    "recursive": recursive_chunk_text,
    "token": token_aware_chunk_text,
}


def chunk_document(
    raw_doc: RawDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    strategy: str = "recursive",
    embed_fn: Optional[Callable] = None,
) -> list[Chunk]:
    """Split one RawDocument into retrieval-ready Chunk objects."""
    if strategy == "semantic":
        pieces = semantic_chunk_text(raw_doc.text, chunk_size, chunk_overlap, embed_fn=embed_fn)
    else:
        splitter = _STRATEGIES.get(strategy, recursive_chunk_text)
        pieces = splitter(raw_doc.text, chunk_size, chunk_overlap)

    meta = raw_doc.metadata
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        chunk_id = make_id(meta.document_id, str(meta.page_number), str(meta.section), str(idx), piece[:40])
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=piece,
                document_id=meta.document_id,
                document_name=meta.document_name,
                document_type=meta.document_type,
                source=meta.source,
                page_number=meta.page_number,
                section=meta.section,
                chunk_index=idx,
            )
        )
    return chunks


def chunk_documents(
    raw_docs: list[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    strategy: str = "recursive",
    embed_fn: Optional[Callable] = None,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in raw_docs:
        all_chunks.extend(
            chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy, embed_fn=embed_fn)
        )
    return all_chunks
