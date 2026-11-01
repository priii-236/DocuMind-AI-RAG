"""
Retrieval and generation metrics.

Retrieval: Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K
Generation: Faithfulness, Answer Relevancy, Context Precision, Context Recall
  (computed with an LLM-as-judge pattern - a lighter-weight, more
  deployment-friendly stand-in for the full RAGAS library, which pulls in a
  heavy dependency tree that is fragile on Streamlit Community Cloud. The
  scoring rubric mirrors RAGAS's definitions.)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from src.generation.llm_client import LLMClient
from src.observability.logger import logger


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for r in top_k if r in relevant_ids) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if any(r in relevant_ids for r in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    dcg = sum((1.0 if rid in relevant_ids else 0.0) / math.log2(i + 2) for i, rid in enumerate(top_k))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int

    def to_dict(self) -> dict:
        return {
            f"Precision@{self.k}": round(self.precision_at_k, 4),
            f"Recall@{self.k}": round(self.recall_at_k, 4),
            f"HitRate@{self.k}": round(self.hit_rate_at_k, 4),
            "MRR": round(self.mrr, 4),
            f"NDCG@{self.k}": round(self.ndcg_at_k, 4),
        }


def evaluate_single_retrieval(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        hit_rate_at_k=hit_rate_at_k(retrieved_ids, relevant_ids, k),
        mrr=reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
        k=k,
    )


def aggregate_retrieval_metrics(per_query: list[RetrievalMetrics]) -> dict:
    if not per_query:
        return {}
    k = per_query[0].k
    return {
        f"Precision@{k}": round(sum(m.precision_at_k for m in per_query) / len(per_query), 4),
        f"Recall@{k}": round(sum(m.recall_at_k for m in per_query) / len(per_query), 4),
        f"HitRate@{k}": round(sum(m.hit_rate_at_k for m in per_query) / len(per_query), 4),
        "MRR": round(sum(m.mrr for m in per_query) / len(per_query), 4),
        f"NDCG@{k}": round(sum(m.ndcg_at_k for m in per_query) / len(per_query), 4),
    }


# --------------------------------------------------------------------------- #
# Generation metrics (LLM-as-judge, RAGAS-style definitions)
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """You are an impartial evaluator scoring a RAG system's answer against the \
retrieved evidence and the question. Score each metric from 0.0 to 1.0.

Definitions:
- faithfulness: fraction of claims in the answer that are actually supported by the evidence (1.0 = fully grounded, no hallucination)
- answer_relevancy: how directly the answer addresses the question asked (1.0 = fully on-topic and complete)
- context_precision: fraction of the retrieved evidence that is actually relevant to the question (1.0 = no irrelevant evidence retrieved)
- context_recall: how much of the information needed to answer (per the ground truth) is present in the retrieved evidence (1.0 = evidence fully covers the ground truth)

Question: {question}
Ground truth answer (may be partial): {ground_truth}
Retrieved evidence:
{evidence}
Generated answer: {answer}

Return ONLY a JSON object like:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}}"""


@dataclass
class GenerationMetrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def to_dict(self) -> dict:
        return {
            "Faithfulness": round(self.faithfulness, 4),
            "Answer Relevancy": round(self.answer_relevancy, 4),
            "Context Precision": round(self.context_precision, 4),
            "Context Recall": round(self.context_recall, 4),
        }


def judge_generation(question: str, ground_truth: str, evidence: str, answer: str, llm: LLMClient) -> GenerationMetrics:
    try:
        raw = llm.complete(
            _JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, evidence=evidence, answer=answer),
            max_tokens=200,
            temperature=0.0,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        scores = json.loads(cleaned)
        return GenerationMetrics(
            faithfulness=float(scores.get("faithfulness", 0.0)),
            answer_relevancy=float(scores.get("answer_relevancy", 0.0)),
            context_precision=float(scores.get("context_precision", 0.0)),
            context_recall=float(scores.get("context_recall", 0.0)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM-judge scoring failed (%s); returning zeros for this sample.", exc)
        return GenerationMetrics(0.0, 0.0, 0.0, 0.0)


def aggregate_generation_metrics(per_query: list[GenerationMetrics]) -> dict:
    if not per_query:
        return {}
    n = len(per_query)
    return {
        "Faithfulness": round(sum(m.faithfulness for m in per_query) / n, 4),
        "Answer Relevancy": round(sum(m.answer_relevancy for m in per_query) / n, 4),
        "Context Precision": round(sum(m.context_precision for m in per_query) / n, 4),
        "Context Recall": round(sum(m.context_recall for m in per_query) / n, 4),
    }
