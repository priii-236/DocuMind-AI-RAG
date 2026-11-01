from src.evaluation.metrics import (
    evaluate_single_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.generation.context_builder import ContextBuilder
from src.ingestion.document import Chunk
from src.retrieval.bm25_retriever import BM25Retriever, tokenize
from src.retrieval.hybrid_retriever import RetrievalCandidate, _minmax_normalize


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, document_id="d1", document_name="doc.txt",
        document_type="txt", source="doc.txt",
    )


def test_precision_recall_hit_rate():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5
    assert recall_at_k(retrieved, relevant, k=5) == 2 / 3
    assert hit_rate_at_k(retrieved, relevant, k=5) == 1.0
    assert hit_rate_at_k(["x", "y"], relevant, k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    retrieved = ["a", "b"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=2) == 1.0


def test_evaluate_single_retrieval_bundle():
    metrics = evaluate_single_retrieval(["a", "b", "c"], {"a"}, k=3)
    assert metrics.precision_at_k > 0
    assert metrics.mrr == 1.0


def test_bm25_tokenize_lowercases_and_strips_punctuation():
    tokens = tokenize("Hello, World! 123")
    assert tokens == ["hello", "world", "123"]


def test_bm25_retriever_finds_relevant_chunk():
    retriever = BM25Retriever()
    chunks = [
        _make_chunk("1", "Random Forest is an ensemble of decision trees."),
        _make_chunk("2", "SQL SELECT statements retrieve data from tables."),
    ]
    retriever.build(chunks)
    results = retriever.search("decision trees ensemble", top_k=2)
    assert results
    assert results[0][0].chunk_id == "1"


def test_minmax_normalize_handles_equal_scores():
    normalized = _minmax_normalize({"a": 5.0, "b": 5.0})
    assert normalized == {"a": 1.0, "b": 1.0}


def test_minmax_normalize_empty():
    assert _minmax_normalize({}) == {}


def test_context_builder_deduplicates_near_identical_chunks():
    c1 = RetrievalCandidate(chunk=_make_chunk("1", "Random Forest combines many decision trees for accuracy."), fused_score=0.9)
    c2 = RetrievalCandidate(chunk=_make_chunk("2", "Random Forest combines many decision trees for better accuracy."), fused_score=0.8)
    c3 = RetrievalCandidate(chunk=_make_chunk("3", "SQL JOIN combines rows from two tables."), fused_score=0.7)

    builder = ContextBuilder(max_context_chars=5000, dedup_threshold=0.7)
    evidence = builder.build([c1, c2, c3])

    ids = {e.chunk_id for e in evidence}
    assert "1" in ids
    assert "2" not in ids  # deduplicated against chunk 1
    assert "3" in ids


def test_context_builder_respects_char_budget():
    long_text = "word " * 2000
    candidates = [RetrievalCandidate(chunk=_make_chunk(str(i), long_text), fused_score=1.0 - i * 0.01) for i in range(5)]
    builder = ContextBuilder(max_context_chars=3000)
    evidence = builder.build(candidates)
    assert len(evidence) >= 1
    assert sum(len(e.text) for e in evidence) <= 3000 + len(long_text)  # allows the first chunk over budget
