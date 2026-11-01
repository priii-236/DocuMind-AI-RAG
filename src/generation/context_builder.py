"""
ContextBuilder: turns a ranked list of retrieval candidates into the final,
compressed evidence block sent to the LLM.

Responsibilities:
  - drop near-duplicate chunks (high text overlap)
  - drop chunks below a minimal relevance floor
  - cap total context length (character budget, cheap stand-in for tokens)
  - format each surviving chunk with a citation tag [n] the LLM must reuse
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.hybrid_retriever import RetrievalCandidate


@dataclass
class EvidenceChunk:
    citation_index: int
    chunk_id: str
    document_name: str
    page_number: int | None
    section: str | None
    text: str
    score: float


def _jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class ContextBuilder:
    def __init__(self, max_context_chars: int = 6000, dedup_threshold: float = 0.85, min_relevance: float = 0.0):
        self.max_context_chars = max_context_chars
        self.dedup_threshold = dedup_threshold
        self.min_relevance = min_relevance

    def build(self, candidates: list[RetrievalCandidate]) -> list[EvidenceChunk]:
        selected: list[EvidenceChunk] = []
        total_chars = 0
        seen_texts: list[str] = []

        for c in candidates:
            if c.fused_score < self.min_relevance:
                continue
            if any(_jaccard_similarity(c.chunk.text, seen) >= self.dedup_threshold for seen in seen_texts):
                continue
            if total_chars + len(c.chunk.text) > self.max_context_chars and selected:
                break

            selected.append(
                EvidenceChunk(
                    citation_index=len(selected) + 1,
                    chunk_id=c.chunk.chunk_id,
                    document_name=c.chunk.document_name,
                    page_number=c.chunk.page_number,
                    section=c.chunk.section,
                    text=c.chunk.text,
                    score=c.fused_score,
                )
            )
            seen_texts.append(c.chunk.text)
            total_chars += len(c.chunk.text)

        return selected

    @staticmethod
    def format_for_prompt(evidence: list[EvidenceChunk]) -> str:
        blocks = []
        for e in evidence:
            loc = f"Page {e.page_number}" if e.page_number else (e.section or "")
            header = f"[{e.citation_index}] {e.document_name}" + (f" — {loc}" if loc else "")
            blocks.append(f"{header}\n{e.text}")
        return "\n\n".join(blocks)
