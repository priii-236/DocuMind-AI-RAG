"""
Document ingestion layer.

Converts uploaded files (PDF, DOCX, TXT, CSV, Markdown, HTML) into a list of
RawDocument objects with normalized text and full provenance metadata.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Optional

from src.ingestion.document import DocumentMetadata, RawDocument, make_id
from src.observability.logger import logger

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md", ".html", ".htm"}


class UnsupportedFileTypeError(Exception):
    pass


class DocumentTooLargeError(Exception):
    pass


class DocumentIngestionError(Exception):
    pass


def sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters from a user-supplied filename."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    return name[:180] if len(name) > 180 else name


def normalize_text(text: str) -> str:
    """Collapse excess whitespace while preserving paragraph breaks and headings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")
    return ext.lstrip(".")


def load_pdf(file_bytes: bytes, filename: str) -> list[RawDocument]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    doc_id = make_id(filename, str(len(file_bytes)))
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)
        if not text:
            continue
        docs.append(
            RawDocument(
                text=text,
                metadata=DocumentMetadata(
                    document_id=doc_id,
                    document_name=filename,
                    document_type="pdf",
                    source=filename,
                    page_number=page_num,
                ),
            )
        )
    if not docs:
        raise DocumentIngestionError(f"No extractable text found in '{filename}' (may be a scanned/image PDF).")
    return docs


def load_docx(file_bytes: bytes, filename: str) -> list[RawDocument]:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(file_bytes))
    doc_id = make_id(filename, str(len(file_bytes)))

    current_section: Optional[str] = None
    buffer: list[str] = []
    docs: list[RawDocument] = []
    chunk_counter = 0

    def flush():
        nonlocal buffer, chunk_counter
        text = normalize_text("\n".join(buffer))
        if text:
            chunk_counter += 1
            docs.append(
                RawDocument(
                    text=text,
                    metadata=DocumentMetadata(
                        document_id=doc_id,
                        document_name=filename,
                        document_type="docx",
                        source=filename,
                        section=current_section,
                    ),
                )
            )
        buffer = []

    for para in document.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            flush()
            current_section = para.text.strip() or current_section
            buffer.append(f"## {para.text.strip()}")
        else:
            if para.text.strip():
                buffer.append(para.text)
        # keep sections from growing unbounded
        if sum(len(b) for b in buffer) > 3000:
            flush()
    flush()

    if not docs:
        raise DocumentIngestionError(f"No extractable text found in '{filename}'.")
    return docs


def load_txt(file_bytes: bytes, filename: str) -> list[RawDocument]:
    text = normalize_text(file_bytes.decode("utf-8", errors="ignore"))
    if not text:
        raise DocumentIngestionError(f"'{filename}' is empty.")
    doc_id = make_id(filename, str(len(file_bytes)))
    return [
        RawDocument(
            text=text,
            metadata=DocumentMetadata(
                document_id=doc_id, document_name=filename, document_type="txt", source=filename
            ),
        )
    ]


def load_markdown(file_bytes: bytes, filename: str) -> list[RawDocument]:
    raw = file_bytes.decode("utf-8", errors="ignore")
    doc_id = make_id(filename, str(len(file_bytes)))

    # Split on top-level headings so each section keeps its own metadata.
    sections = re.split(r"(?m)^(#{1,3}\s+.*)$", raw)
    docs: list[RawDocument] = []
    current_heading = None
    buf = []

    def flush():
        text = normalize_text("".join(buf))
        if text:
            docs.append(
                RawDocument(
                    text=text,
                    metadata=DocumentMetadata(
                        document_id=doc_id,
                        document_name=filename,
                        document_type="md",
                        source=filename,
                        section=current_heading,
                    ),
                )
            )

    for part in sections:
        if re.match(r"^#{1,3}\s+", part):
            flush()
            buf = [part + "\n"]
            current_heading = part.lstrip("#").strip()
        else:
            buf.append(part)
    flush()

    if not docs:
        text = normalize_text(raw)
        if text:
            docs = [
                RawDocument(
                    text=text,
                    metadata=DocumentMetadata(
                        document_id=doc_id, document_name=filename, document_type="md", source=filename
                    ),
                )
            ]
    if not docs:
        raise DocumentIngestionError(f"'{filename}' is empty.")
    return docs


def load_html(file_bytes: bytes, filename: str) -> list[RawDocument]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(file_bytes, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = normalize_text(soup.get_text(separator="\n"))
    if not text:
        raise DocumentIngestionError(f"No extractable text found in '{filename}'.")
    doc_id = make_id(filename, str(len(file_bytes)))
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    return [
        RawDocument(
            text=text,
            metadata=DocumentMetadata(
                document_id=doc_id,
                document_name=filename,
                document_type="html",
                source=filename,
                section=title,
            ),
        )
    ]


def load_csv(file_bytes: bytes, filename: str) -> list[RawDocument]:
    text_data = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text_data))
    doc_id = make_id(filename, str(len(file_bytes)))
    docs: list[RawDocument] = []
    rows_per_chunk = 20
    buffer_rows: list[str] = []

    for i, row in enumerate(reader):
        row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        buffer_rows.append(row_text)
        if len(buffer_rows) >= rows_per_chunk:
            docs.append(
                RawDocument(
                    text=normalize_text("\n".join(buffer_rows)),
                    metadata=DocumentMetadata(
                        document_id=doc_id,
                        document_name=filename,
                        document_type="csv",
                        source=filename,
                        section=f"rows {i - len(buffer_rows) + 2}-{i + 1}",
                    ),
                )
            )
            buffer_rows = []

    if buffer_rows:
        docs.append(
            RawDocument(
                text=normalize_text("\n".join(buffer_rows)),
                metadata=DocumentMetadata(
                    document_id=doc_id, document_name=filename, document_type="csv", source=filename
                ),
            )
        )

    if not docs:
        raise DocumentIngestionError(f"'{filename}' has no rows.")
    return docs


_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "txt": load_txt,
    "md": load_markdown,
    "html": load_html,
    "htm": load_html,
    "csv": load_csv,
}


def load_document(file_bytes: bytes, filename: str, max_size_mb: int = 25) -> list[RawDocument]:
    """Single entry point: validates, dispatches to the right loader, returns RawDocuments."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise DocumentTooLargeError(f"'{filename}' is {size_mb:.1f}MB, exceeds the {max_size_mb}MB limit.")

    safe_name = sanitize_filename(filename)
    file_type = _detect_type(safe_name)
    loader = _LOADERS[file_type]

    try:
        docs = loader(file_bytes, safe_name)
    except (UnsupportedFileTypeError, DocumentTooLargeError, DocumentIngestionError):
        raise
    except Exception as exc:  # noqa: BLE001 - convert any parser failure into a safe, user-facing error
        logger.exception("Failed to ingest %s", safe_name)
        raise DocumentIngestionError(f"Could not process '{safe_name}': the file may be corrupted or malformed.") from exc

    logger.info("Ingested '%s' -> %d raw section(s)", safe_name, len(docs))
    return docs
