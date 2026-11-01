"""
Experiment framework: sweeps chunk size, retrieval method, reranking on/off,
query rewriting on/off, and top-k, logging retrieval metrics + latency for
each configuration to experiments/results.csv.

Usage:
    python -m experiments.run_experiments
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402
from src.evaluation.evaluate_retrieval import _relevant_ids_for_row, load_eval_dataset  # noqa: E402
from src.evaluation.metrics import aggregate_retrieval_metrics, evaluate_single_retrieval  # noqa: E402
from src.generation.rag_pipeline import RAGPipeline  # noqa: E402
from src.reranking.reranker import get_reranker_service  # noqa: E402
from src.retrieval.knowledge_base import KnowledgeBase  # noqa: E402

RESULTS_PATH = settings.experiments_dir / "results.csv"

FIELDNAMES = [
    "experiment", "config", "retrieval_method", "alpha", "top_k", "reranking",
    "query_rewriting", "chunk_size", "chunk_overlap", "embedding_model",
    "precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k",
    "avg_latency_seconds",
]


def _evaluate_config(kb: KnowledgeBase, pipeline: RAGPipeline, method: str, alpha: float, top_k: int, reranking: bool) -> dict:
    dataset = load_eval_dataset()
    metrics_list = []
    latencies = []

    reranker = get_reranker_service(enabled=reranking)

    for row in dataset:
        question = row["question"]
        relevant_ids = _relevant_ids_for_row(row, kb)
        if not relevant_ids:
            continue

        t0 = time.time()
        if method == "dense":
            emb = kb.embedding_service.embed_query(question)
            hits = kb.vector_store.search(emb, top_k=top_k)
            ids = [c.chunk_id for c, _ in hits]
        elif method == "sparse":
            hits = kb.bm25.search(question, top_k=top_k)
            ids = [c.chunk_id for c, _ in hits]
        else:  # hybrid
            candidates, _, _ = pipeline.hybrid_retriever.retrieve_single(question, top_k=settings.candidate_pool_size, alpha=alpha)
            if reranking:
                candidates = reranker.rerank(question, candidates, top_k=top_k)
            else:
                candidates = candidates[:top_k]
            ids = [c.chunk.chunk_id for c in candidates]
        latencies.append(time.time() - t0)
        metrics_list.append(evaluate_single_retrieval(ids, relevant_ids, k=top_k))

    agg = aggregate_retrieval_metrics(metrics_list)
    avg_latency = round(sum(latencies) / len(latencies), 4) if latencies else 0.0
    return {**agg, "avg_latency_seconds": avg_latency}


def run_all_experiments() -> list[dict]:
    kb = KnowledgeBase()
    if kb.vector_store.count() == 0:
        raise RuntimeError("Knowledge base is empty. Load sample data first (see data/README.md).")
    pipeline = RAGPipeline(kb)

    rows = []

    # Experiment 2: Dense vs BM25 vs Hybrid
    for method in ("dense", "sparse", "hybrid"):
        metrics = _evaluate_config(kb, pipeline, method=method, alpha=settings.hybrid_alpha, top_k=5, reranking=False)
        rows.append({
            "experiment": "retrieval_method", "config": method, "retrieval_method": method,
            "alpha": settings.hybrid_alpha, "top_k": 5, "reranking": False, "query_rewriting": False,
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "precision_at_k": metrics.get("Precision@5"), "recall_at_k": metrics.get("Recall@5"),
            "hit_rate_at_k": metrics.get("HitRate@5"), "mrr": metrics.get("MRR"), "ndcg_at_k": metrics.get("NDCG@5"),
            "avg_latency_seconds": metrics.get("avg_latency_seconds"),
        })

    # Experiment 3: reranking off vs on (hybrid)
    for reranking in (False, True):
        metrics = _evaluate_config(kb, pipeline, method="hybrid", alpha=settings.hybrid_alpha, top_k=5, reranking=reranking)
        rows.append({
            "experiment": "reranking", "config": f"reranking={reranking}", "retrieval_method": "hybrid",
            "alpha": settings.hybrid_alpha, "top_k": 5, "reranking": reranking, "query_rewriting": False,
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "precision_at_k": metrics.get("Precision@5"), "recall_at_k": metrics.get("Recall@5"),
            "hit_rate_at_k": metrics.get("HitRate@5"), "mrr": metrics.get("MRR"), "ndcg_at_k": metrics.get("NDCG@5"),
            "avg_latency_seconds": metrics.get("avg_latency_seconds"),
        })

    # Experiment 5: top-k sweep (hybrid + rerank)
    for top_k in (3, 5, 8, 10):
        metrics = _evaluate_config(kb, pipeline, method="hybrid", alpha=settings.hybrid_alpha, top_k=top_k, reranking=True)
        rows.append({
            "experiment": "top_k_sweep", "config": f"top_k={top_k}", "retrieval_method": "hybrid",
            "alpha": settings.hybrid_alpha, "top_k": top_k, "reranking": True, "query_rewriting": False,
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "precision_at_k": metrics.get(f"Precision@{top_k}"), "recall_at_k": metrics.get(f"Recall@{top_k}"),
            "hit_rate_at_k": metrics.get(f"HitRate@{top_k}"), "mrr": metrics.get("MRR"), "ndcg_at_k": metrics.get(f"NDCG@{top_k}"),
            "avg_latency_seconds": metrics.get("avg_latency_seconds"),
        })

    # Experiment 1: alpha sweep (dense/sparse weighting) at fixed top_k
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        metrics = _evaluate_config(kb, pipeline, method="hybrid", alpha=alpha, top_k=5, reranking=False)
        rows.append({
            "experiment": "alpha_sweep", "config": f"alpha={alpha}", "retrieval_method": "hybrid",
            "alpha": alpha, "top_k": 5, "reranking": False, "query_rewriting": False,
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "precision_at_k": metrics.get("Precision@5"), "recall_at_k": metrics.get("Recall@5"),
            "hit_rate_at_k": metrics.get("HitRate@5"), "mrr": metrics.get("MRR"), "ndcg_at_k": metrics.get("NDCG@5"),
            "avg_latency_seconds": metrics.get("avg_latency_seconds"),
        })

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    results = run_all_experiments()
    print(f"Wrote {len(results)} experiment rows to {RESULTS_PATH}")
