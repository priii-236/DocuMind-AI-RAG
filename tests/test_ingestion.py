import pytest

from src.ingestion.loader import (
    DocumentIngestionError,
    DocumentTooLargeError,
    UnsupportedFileTypeError,
    load_document,
    normalize_text,
    sanitize_filename,
)


def test_normalize_text_collapses_whitespace():
    raw = "Hello   world\n\n\n\nThis  is\tfine."
    result = normalize_text(raw)
    assert "   " not in result
    assert "\n\n\n" not in result


def test_sanitize_filename_strips_path_and_unsafe_chars():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my report (final)!.pdf") == "my_report__final__.pdf"


def test_load_txt_document():
    docs = load_document(b"Machine learning is a field of AI.", "notes.txt")
    assert len(docs) == 1
    assert "Machine learning" in docs[0].text
    assert docs[0].metadata.document_type == "txt"


def test_load_markdown_document_splits_by_heading():
    content = b"# Intro\nSome intro text.\n## Details\nMore details here that are long enough."
    docs = load_document(content, "doc.md")
    assert len(docs) >= 1
    assert all(d.metadata.document_type == "md" for d in docs)


def test_load_csv_document():
    content = b"name,value\na,1\nb,2\n"
    docs = load_document(content, "data.csv")
    assert len(docs) == 1
    assert "name" in docs[0].text


def test_unsupported_file_type_raises():
    with pytest.raises(UnsupportedFileTypeError):
        load_document(b"binary", "malware.exe")


def test_empty_txt_raises_ingestion_error():
    with pytest.raises(DocumentIngestionError):
        load_document(b"   \n\n  ", "empty.txt")


def test_oversized_file_raises():
    big_content = b"a" * (2 * 1024 * 1024)
    with pytest.raises(DocumentTooLargeError):
        load_document(big_content, "big.txt", max_size_mb=1)
