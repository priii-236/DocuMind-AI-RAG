"""
Query understanding: contextual (conversational) rewriting + multi-query expansion.

Uses the configured LLM for both. Falls back to the original query
unmodified if the LLM call fails, so a flaky rewrite step never breaks
retrieval.
"""
from __future__ import annotations

import json

from src.generation.llm_client import LLMClient
from src.observability.logger import logger

_CONTEXTUAL_REWRITE_PROMPT = """You rewrite a user's latest chat message into a fully standalone search \
query, using the conversation history to resolve pronouns and implicit references \
(e.g. "it", "that", "the second one").

Conversation history:
{history}

Latest user message: "{query}"

Return ONLY the standalone, self-contained search query. No preamble, no quotes."""

_EXPANSION_PROMPT = """Generate {n} different search queries that would help retrieve documents \
relevant to answering the user's question. Vary phrasing and angle (definition, cause, \
method, comparison) so together they cover the question thoroughly.

Question: "{query}"

Return ONLY a JSON array of {n} strings, nothing else."""


def rewrite_with_history(query: str, history: list[dict], llm: LLMClient) -> str:
    """Resolve conversational references ('it', 'that') into a standalone query."""
    if not history:
        return query
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])
    try:
        result = llm.complete(
            _CONTEXTUAL_REWRITE_PROMPT.format(history=history_text, query=query),
            max_tokens=100,
            temperature=0.0,
        )
        cleaned = result.strip().strip('"')
        return cleaned if cleaned else query
    except Exception as exc:  # noqa: BLE001
        logger.warning("Contextual query rewrite failed (%s); using original query.", exc)
        return query


def expand_query(query: str, llm: LLMClient, n: int = 3) -> list[str]:
    """Multi-query expansion: generate n alternate phrasings/angles of the same question."""
    try:
        result = llm.complete(_EXPANSION_PROMPT.format(query=query, n=n), max_tokens=250, temperature=0.4)
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        queries = json.loads(cleaned)
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if query not in queries:
            queries.insert(0, query)
        return queries[: n + 1]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query expansion failed (%s); using original query only.", exc)
        return [query]
