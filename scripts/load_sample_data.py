"""
Ingests every file in data/raw/ into the vector store so the app, the
evaluation scripts, and the experiments can be run immediately without
manually uploading documents through the UI.

Usage:
    python scripts/load_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402
from src.retrieval.knowledge_base import KnowledgeBase  # noqa: E402


def main() -> None:
    kb = KnowledgeBase()
    raw_files = sorted(settings.data_raw_dir.glob("*"))
    raw_files = [f for f in raw_files if f.suffix.lower() in settings.allowed_extensions]

    if not raw_files:
        print(f"No sample files found in {settings.data_raw_dir}")
        return

    files_payload = [(f.read_bytes(), f.name) for f in raw_files]
    results = kb.ingest_files(files_payload)

    for r in results:
        status = "OK" if r.success else "FAILED"
        detail = f"{r.num_chunks} chunks" if r.success else r.error
        print(f"[{status}] {r.filename} - {detail}")

    stats = kb.stats()
    print(f"\nKnowledge base ready: {stats['num_documents']} documents, {stats['num_chunks']} chunks.")


if __name__ == "__main__":
    main()
