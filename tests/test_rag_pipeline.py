import hashlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.generation.rag_pipeline import RAGPipeline, _NOT_FOUND_MESSAGE
from src.retrieval.knowledge_base import KnowledgeBase


class _FakeEmbeddingService:
    """
    Deterministic, dependency-free stand-in for the real sentence-transformers
    model, so unit tests don't require downloading a model / network access.
    Uses a simple hashed bag-of-words vector so texts sharing words end up
    with non-trivial cosine similarity - good enough to exercise the
    retrieval/ranking logic without a real embedding model.
    """

    dimension = 64

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self._embed_one(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_one(query)


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    # Isolate this test's vector store from the real app's persisted data
    # (which uses a different embedding dimension) by pointing FAISS at a
    # throwaway directory for the duration of these tests.
    from src.config.settings import settings

    settings.vector_store_dir = tmp_path_factory.mktemp("test_vector_store")

    kb = KnowledgeBase(embedding_service=_FakeEmbeddingService())
    kb.ingest_file(
        b"Random Forest is an ensemble learning method combining many decision trees "
        b"to reduce overfitting and improve accuracy compared to a single decision tree.",
        "ml_notes.txt",
    )
    return kb


def _fake_llm(fixed_answer: str = "Random Forest combines many decision trees [1]."):
    llm = MagicMock()
    llm.complete.return_value = fixed_answer
    return llm


def test_pipeline_returns_not_found_message_on_empty_kb(tmp_path):
    from src.config.settings import settings

    settings.vector_store_dir = tmp_path
    empty_kb = KnowledgeBase(embedding_service=_FakeEmbeddingService())
    pipeline = RAGPipeline(empty_kb, llm_client=_fake_llm())
    response = pipeline.answer("What is a Random Forest?")
    assert not response.is_grounded
    assert "empty" in response.answer.lower()


def test_pipeline_grounded_answer_includes_citations(kb):
    pipeline = RAGPipeline(kb, llm_client=_fake_llm())
    response = pipeline.answer("What is a Random Forest?", enable_query_rewriting=False, enable_reranking=False)
    assert response.is_grounded
    assert len(response.evidence) >= 1
    assert response.evidence[0].citation_index == 1
    assert response.evidence[0].document_name == "ml_notes.txt"


def test_pipeline_hallucination_control_for_irrelevant_query(kb):
    pipeline = RAGPipeline(kb, llm_client=_fake_llm())
    # Force an unrealistically high confidence bar so even a real hit is rejected,
    # simulating the "insufficient evidence" path.
    from src.config import settings as settings_module

    original_threshold = settings_module.settings.min_confidence_threshold
    settings_module.settings.min_confidence_threshold = 1.1
    try:
        response = pipeline.answer("Completely unrelated quantum physics question", enable_query_rewriting=False, enable_reranking=False)
        assert not response.is_grounded
        assert response.answer == _NOT_FOUND_MESSAGE
    finally:
        settings_module.settings.min_confidence_threshold = original_threshold


def test_baseline_pipeline_runs(kb):
    pipeline = RAGPipeline(kb, llm_client=_fake_llm())
    response = pipeline.answer_baseline("What is a Random Forest?")
    assert response.trace.pipeline_mode == "baseline"
