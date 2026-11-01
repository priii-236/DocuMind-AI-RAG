from src.chunking.chunker import chunk_document, recursive_chunk_text
from src.ingestion.document import DocumentMetadata, RawDocument


def test_recursive_chunk_text_respects_size_budget():
    text = "Sentence one. " * 200
    chunks = recursive_chunk_text(text, chunk_size=200, overlap=20)
    assert all(len(c) <= 260 for c in chunks)  # allow slack for overlap
    assert len(chunks) > 1


def test_recursive_chunk_text_short_text_single_chunk():
    text = "Short text."
    chunks = recursive_chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == ["Short text."]


def test_chunk_document_produces_chunk_ids_and_metadata():
    meta = DocumentMetadata(document_id="doc1", document_name="test.txt", document_type="txt", source="test.txt")
    raw = RawDocument(text="Paragraph one is here. " * 100, metadata=meta)
    chunks = chunk_document(raw, chunk_size=300, chunk_overlap=30, strategy="recursive")

    assert len(chunks) > 0
    assert all(c.document_id == "doc1" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # all unique


def test_chunk_document_overlap_creates_shared_text():
    meta = DocumentMetadata(document_id="doc2", document_name="test.txt", document_type="txt", source="test.txt")
    raw = RawDocument(text="Word " * 500, metadata=meta)
    chunks = chunk_document(raw, chunk_size=200, chunk_overlap=50, strategy="recursive")
    assert len(chunks) > 1
