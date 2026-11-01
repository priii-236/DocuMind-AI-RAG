"""
RAGPipeline: orchestrates the full advanced flow, and exposes a baseline
mode for the comparison experiments.

    Advanced:
        query -> contextual rewrite -> query expansion -> hybrid retrieval
              -> merge/dedup -> rerank -> context compression -> grounded LLM answer

    Baseline:
        query -> dense retrieval only -> top-k -> LLM answer  (no rewriting,
        no hybrid fusion, no reranking, no compression)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.config.settings import settings
from src.generation.context_builder import ContextBuilder, EvidenceChunk
from src.generation.llm_client import LLMClient
from src.memory.conversation_memory import ConversationMemory
from src.observability.logger import QueryTrace, trace_store
from src.reranking.reranker import get_reranker_service
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalCandidate
from src.retrieval.knowledge_base import KnowledgeBase
from src.retrieval.query_rewriter import expand_query, rewrite_with_history

_NOT_FOUND_MESSAGE = (
    "I couldn't find enough evidence in the uploaded knowledge base to answer this confidently. "
    "Try rephrasing the question, or upload a document that covers this topic."
)

_SYSTEM_PROMPT = """You are DocuMind AI, a grounded document question-answering assistant.

Rules you must follow strictly:
- Answer ONLY using the numbered evidence passages provided below. Do not use outside knowledge for factual claims.
- Every factual claim must be traceable to one or more of the evidence passages. Reference them inline using their bracket number, e.g. [1], [2].
- If the evidence does not contain the answer, say so plainly instead of guessing.
- Never invent a citation number that isn't in the evidence list.
- Prefer concise, well-structured answers (use bullet points or short paragraphs, tables when comparing things).
- Distinguish clearly between what the evidence states and any general reasoning you add on top of it."""

_USER_PROMPT_TEMPLATE = """Evidence passages:
{context}

Question: {question}

Answer the question using only the evidence above. Cite passage numbers inline."""


@dataclass
class RAGResponse:
    answer: str
    evidence: list[EvidenceChunk]
    confidence: float
    trace: QueryTrace
    is_grounded: bool


class RAGPipeline:
    def __init__(self, knowledge_base: KnowledgeBase, llm_client: LLMClient | None = None):
        self.kb = knowledge_base
        self.llm = llm_client or LLMClient()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.kb.vector_store, bm25=self.kb.bm25, embedding_service=self.kb.embedding_service
        )
        self.context_builder = ContextBuilder(min_relevance=0.0)

    # ------------------------------------------------------------------ #
    # Advanced pipeline
    # ------------------------------------------------------------------ #
    def answer(
        self,
        query: str,
        memory: ConversationMemory | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        candidate_pool: int | None = None,
        enable_query_rewriting: bool | None = None,
        enable_reranking: bool | None = None,
    ) -> RAGResponse:
        start = time.time()
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        alpha = alpha if alpha is not None else settings.hybrid_alpha
        candidate_pool = candidate_pool if candidate_pool is not None else settings.candidate_pool_size
        do_rewrite = settings.enable_query_rewriting if enable_query_rewriting is None else enable_query_rewriting
        do_rerank = settings.enable_reranking if enable_reranking is None else enable_reranking

        trace = QueryTrace(original_query=query, pipeline_mode="advanced")

        try:
            if self.kb.vector_store.count() == 0:
                trace.answer = "Your knowledge base is empty. Please upload at least one document first."
                trace.confidence = 0.0
                trace.latency_seconds = time.time() - start
                trace_store.add(trace)
                return RAGResponse(trace.answer, [], 0.0, trace, is_grounded=False)

            # 1. Contextual rewrite (resolve "it", "that", etc.)
            standalone_query = query
            if do_rewrite and memory and memory.turns:
                standalone_query = rewrite_with_history(query, memory.as_dicts(), self.llm)

            # 2. Multi-query expansion
            queries = [standalone_query]
            if do_rewrite:
                queries = expand_query(standalone_query, self.llm, n=3)
            trace.rewritten_queries = queries

            # 3. Hybrid retrieval (dense + sparse, multi-query merge)
            candidates, dense_debug, sparse_debug = self.hybrid_retriever.retrieve_multi(
                queries, top_k=candidate_pool, alpha=alpha
            )
            trace.dense_results = dense_debug
            trace.sparse_results = sparse_debug
            trace.merged_results = [c.to_debug_dict() for c in candidates[:candidate_pool]]

            # 4. Reranking
            if do_rerank and candidates:
                reranker = get_reranker_service(enabled=True)
                candidates = reranker.rerank(standalone_query, candidates[:candidate_pool], top_k=top_k)
            else:
                candidates = candidates[:top_k]
            trace.reranked_results = [c.to_debug_dict() for c in candidates]

            # 5. Context compression
            evidence = self.context_builder.build(candidates)
            trace.final_context_chunks = [
                {"citation_index": e.citation_index, "document_name": e.document_name, "page_number": e.page_number, "score": round(e.score, 4)}
                for e in evidence
            ]

            confidence = candidates[0].fused_score if candidates else 0.0
            is_grounded = bool(evidence) and confidence >= settings.min_confidence_threshold

            if not is_grounded:
                trace.answer = _NOT_FOUND_MESSAGE
                trace.confidence = confidence
                trace.latency_seconds = time.time() - start
                trace_store.add(trace)
                return RAGResponse(_NOT_FOUND_MESSAGE, evidence, confidence, trace, is_grounded=False)

            # 6. Grounded generation
            context_text = self.context_builder.format_for_prompt(evidence)
            prompt = _USER_PROMPT_TEMPLATE.format(context=context_text, question=query)
            print("DEBUG: Gemini API call started")
            llm_start = time.perf_counter()

            answer_text = self.llm.complete(
                prompt, system=_SYSTEM_PROMPT
            )

            print(
                f"DEBUG: Gemini API took "
                f"{time.perf_counter() - llm_start:.2f} seconds"
            )

            trace.answer = answer_text
            trace.confidence = confidence
            trace.latency_seconds = time.time() - start
            trace_store.add(trace)

            return RAGResponse(answer_text, evidence, confidence, trace, is_grounded=True)

        except Exception as exc:  # noqa: BLE001
            trace.error = str(exc)
            trace.latency_seconds = time.time() - start
            trace_store.add(trace)
            raise

    # ------------------------------------------------------------------ #
    # Baseline pipeline (dense-only, no rewrite/rerank/compression) - used
    # purely for the experiments/comparison report.
    # ------------------------------------------------------------------ #
    def answer_baseline(self, query: str, top_k: int | None = None) -> RAGResponse:
        start = time.time()
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        trace = QueryTrace(original_query=query, pipeline_mode="baseline")

        if self.kb.vector_store.count() == 0:
            trace.answer = "Knowledge base is empty."
            trace_store.add(trace)
            return RAGResponse(trace.answer, [], 0.0, trace, is_grounded=False)

        query_embedding = self.kb.embedding_service.embed_query(query)
        dense_hits = self.kb.vector_store.search(query_embedding, top_k=top_k)
        candidates = [RetrievalCandidate(chunk=c, dense_score=s, fused_score=s) for c, s in dense_hits]
        trace.dense_results = [{"chunk_id": c.chunk.chunk_id, "document_name": c.chunk.document_name, "score": round(c.fused_score, 4)} for c in candidates]

        evidence = self.context_builder.build(candidates)
        trace.final_context_chunks = [{"citation_index": e.citation_index, "document_name": e.document_name} for e in evidence]

        confidence = candidates[0].fused_score if candidates else 0.0
        is_grounded = bool(evidence) and confidence >= settings.min_confidence_threshold

        if not is_grounded:
            trace.answer = _NOT_FOUND_MESSAGE
            trace.confidence = confidence
            trace.latency_seconds = time.time() - start
            trace_store.add(trace)
            return RAGResponse(_NOT_FOUND_MESSAGE, evidence, confidence, trace, is_grounded=False)

        context_text = self.context_builder.format_for_prompt(evidence)
        prompt = _USER_PROMPT_TEMPLATE.format(context=context_text, question=query)
        answer_text = self.llm.complete(prompt, system=_SYSTEM_PROMPT)

        trace.answer = answer_text
        trace.confidence = confidence
        trace.latency_seconds = time.time() - start
        trace_store.add(trace)
        return RAGResponse(answer_text, evidence, confidence, trace, is_grounded=True)
