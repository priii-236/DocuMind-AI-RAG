"""
Evaluates generation quality (baseline vs advanced pipeline) using the
LLM-as-judge metrics defined in metrics.py.

Usage:
    python -m src.evaluation.evaluate_generation
"""
from __future__ import annotations

import json
import time

from src.config.settings import settings
from src.evaluation.evaluate_retrieval import load_eval_dataset
from src.evaluation.metrics import aggregate_generation_metrics, judge_generation
from src.generation.llm_client import LLMClient
from src.generation.rag_pipeline import RAGPipeline
from src.observability.logger import logger
from src.retrieval.knowledge_base import KnowledgeBase


def run_generation_evaluation(kb: KnowledgeBase, pipeline: RAGPipeline, sample_size: int = 15) -> dict:
    dataset = load_eval_dataset()[:sample_size]
    judge = LLMClient()

    baseline_scores, advanced_scores = [], []
    baseline_latencies, advanced_latencies = [], []

    for row in dataset:
        question = row["question"]
        ground_truth = row.get("ground_truth_answer", "")

        t0 = time.time()
        baseline_resp = pipeline.answer_baseline(question)
        baseline_latencies.append(time.time() - t0)
        baseline_evidence_text = "\n".join(e.text for e in baseline_resp.evidence)
        baseline_scores.append(judge_generation(question, ground_truth, baseline_evidence_text, baseline_resp.answer, judge))

        t0 = time.time()
        advanced_resp = pipeline.answer(question)
        advanced_latencies.append(time.time() - t0)
        advanced_evidence_text = "\n".join(e.text for e in advanced_resp.evidence)
        advanced_scores.append(judge_generation(question, ground_truth, advanced_evidence_text, advanced_resp.answer, judge))

    result = {
        "baseline": {
            **aggregate_generation_metrics(baseline_scores),
            "avg_latency_seconds": round(sum(baseline_latencies) / len(baseline_latencies), 3) if baseline_latencies else 0,
        },
        "advanced": {
            **aggregate_generation_metrics(advanced_scores),
            "avg_latency_seconds": round(sum(advanced_latencies) / len(advanced_latencies), 3) if advanced_latencies else 0,
        },
        "num_queries_evaluated": len(dataset),
    }
    logger.info("Generation evaluation complete: %s", json.dumps(result))
    return result


if __name__ == "__main__":
    kb = KnowledgeBase()
    pipeline = RAGPipeline(kb)
    if kb.vector_store.count() == 0:
        print("Knowledge base is empty. Run scripts/load_sample_data.py first (see data/README.md).")
    else:
        results = run_generation_evaluation(kb, pipeline)
        print(json.dumps(results, indent=2))
