"""
Evaluates retrieval quality (baseline dense-only vs advanced hybrid+rerank)
against evaluation/evaluation_dataset.csv.

Usage:
    python -m src.evaluation.evaluate_retrieval
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config.settings import settings
from src.evaluation.metrics import aggregate_retrieval_metrics, evaluate_single_retrieval
from src.generation.rag_pipeline import RAGPipeline
from src.observability.logger import logger
from src.retrieval.knowledge_base import KnowledgeBase


def load_eval_dataset(path: Path | None = None) -> list[dict]:
    path = path or (settings.evaluation_dir / "evaluation_dataset.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _relevant_ids_for_row(row: dict, kb: KnowledgeBase) -> set[str]:
    """
    The dataset stores a human-readable `relevant_document` (filename) and
    `relevant_chunk` (a short text fragment expected to appear in the
    correct chunk). We resolve that to actual chunk_ids present in the
    live knowledge base so this works regardless of how chunking split things.
    """
    target_doc = row.get("relevant_document", "").strip()
    fragment = row.get("relevant_chunk", "").strip().lower()
    ids = set()
    for chunk in kb.all_chunks():
        if target_doc and target_doc.lower() not in chunk.document_name.lower():
            continue
        if fragment and fragment in chunk.text.lower():
            ids.add(chunk.chunk_id)
    return ids


def run_retrieval_evaluation(kb: KnowledgeBase, pipeline: RAGPipeline, k: int = 5) -> dict:
    dataset = load_eval_dataset()
    baseline_metrics, advanced_metrics = [], []

    for row in dataset:
        question = row["question"]
        relevant_ids = _relevant_ids_for_row(row, kb)
        if not relevant_ids:
            continue  # skip rows we can't ground against the current KB

        # Baseline: dense-only top-k
        query_embedding = kb.embedding_service.embed_query(question)
        dense_hits = kb.vector_store.search(query_embedding, top_k=k)
        baseline_ids = [c.chunk_id for c, _ in dense_hits]
        baseline_metrics.append(evaluate_single_retrieval(baseline_ids, relevant_ids, k=k))

        # Advanced: hybrid + rerank (reuse pipeline's retriever directly, no LLM call needed)
        candidates, _, _ = pipeline.hybrid_retriever.retrieve_single(question, top_k=settings.candidate_pool_size, alpha=settings.hybrid_alpha)
        from src.reranking.reranker import get_reranker_service

        reranker = get_reranker_service(enabled=True)
        reranked = reranker.rerank(question, candidates, top_k=k)
        advanced_ids = [c.chunk.chunk_id for c in reranked]
        advanced_metrics.append(evaluate_single_retrieval(advanced_ids, relevant_ids, k=k))

    result = {
        "baseline": aggregate_retrieval_metrics(baseline_metrics),
        "advanced": aggregate_retrieval_metrics(advanced_metrics),
        "num_queries_evaluated": len(baseline_metrics),
        "k": k,
    }
    logger.info("Retrieval evaluation complete: %s", json.dumps(result))
    return result


if __name__ == "__main__":
    kb = KnowledgeBase()
    pipeline = RAGPipeline(kb)
    if kb.vector_store.count() == 0:
        print("Knowledge base is empty. Run scripts/load_sample_data.py first (see data/README.md).")
    else:
        results = run_retrieval_evaluation(kb, pipeline)
        print(json.dumps(results, indent=2))
