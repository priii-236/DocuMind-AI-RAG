# Data

## Your own documents

DocuMind AI is built to work with **your own knowledge base**. Upload PDF, DOCX, TXT, CSV,
Markdown, or HTML files directly from the Streamlit sidebar — nothing needs to be placed here
manually for normal use.

## Sample knowledge base (for demonstration / evaluation)

`data/raw/` ships with six original Markdown documents (Machine Learning, SQL, Statistics, Python
for Data Science, and AI/RAG concepts) written specifically for this project so the demo and the
evaluation dataset (`evaluation/evaluation_dataset.csv`) have a real, license-free knowledge base
to run against. No copyrighted books, course material, or third-party content is included.

To index the sample knowledge base:

```bash
python scripts/load_sample_data.py
```

This ingests every file in `data/raw/` into the vector store so you can immediately try the chat
UI, the Retrieval Debugger, and the evaluation/experiment scripts without uploading anything.

## data/processed/

Reserved for any cached/derived artifacts (currently unused by default FAISS mode — FAISS persists
directly to `vector_store/`).
