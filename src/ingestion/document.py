"""Core data models shared across the ingestion -> chunking -> retrieval pipeline."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional


def make_id(*parts: str) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass
class DocumentMetadata:
    document_id: str
    document_name: str
    document_type: str
    source: str
    created_at: float = field(default_factory=time.time)
    page_number: Optional[int] = None
    section: Optional[str] = None


@dataclass
class RawDocument:
    """A single extracted unit of text before chunking (e.g. one PDF page, or a whole .txt file)."""

    text: str
    metadata: DocumentMetadata


@dataclass
class Chunk:
    """A retrieval-ready unit of text with full provenance metadata."""

    chunk_id: str
    text: str
    document_id: str
    document_name: str
    document_type: str
    source: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    created_at: float = field(default_factory=time.time)

    def to_metadata_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "source": self.source,
            "page_number": self.page_number,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }
