# Local-first PDF Question Answering

A local app for uploading PDFs, extracting text locally, OCR-ing scanned pages only when needed, indexing chunks in FAISS, and answering questions with page-level citations.

## What it does

- Upload multiple PDFs
- Extract text with PyMuPDF
- OCR only scanned pages with Tesseract
- Chunk pages while preserving document name, page number, and source type
- Create local embeddings with sentence-transformers
- Store vectors in FAISS
- Retrieve with semantic search plus optional keyword reranking
- Answer only from retrieved chunks
- Refuse unsupported questions with: `I don't know based on the uploaded documents.`
- Run entirely locally

## Repo structure

```text
.
├── app/
│   ├── __init__.py
│   ├── answer.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── eval.py
│   ├── extract.py
│   ├── ingest.py
│   ├── index.py
│   ├── main.py
│   ├── ocr.py
│   ├── retrieve.py
│   ├── storage.py
│   └── ui.py
├── data/
│   ├── eval/
│   │   └── sample_eval.json
│   ├── index/
│   ├── metadata/
│   └── uploads/
├── tests/
├── requirements.txt
└── README.md
```

## Run Locally

1. Bootstrap the local environment:

```bash
source scripts/startup.sh
```

2. Run:

```bash
streamlit run app/ui.py
```

3. Open the app on port `8501`.

## How it works

### Ingestion
Each PDF is copied into `data/uploads/`, then parsed page by page. If a page has too little text to be a real text page, it is rendered and OCR'd with Tesseract.

### Indexing
Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` and added to a local FAISS index. Chunk metadata is stored in JSONL so citations remain traceable.

### Retrieval
The app retrieves the most relevant chunks using semantic search. Keyword overlap reranking is available from the sidebar.

### Answering
The answerer is deliberately extractive. It selects the most relevant sentences from retrieved chunks and refuses to guess when evidence is weak.

## Design choices

- **No paid APIs**: everything is local and open-source.
- **Extractive answering**: safer than a free-form LLM for a first MVP because it avoids invented citations.
- **Page-level metadata**: every chunk keeps document name, page number, and source type for traceability.
- **Simple storage**: FAISS for vectors, JSONL for metadata, no database server.
- **Local-first setup**: the bootstrap script creates a virtualenv and installs Python dependencies.

## Tests

Run:

```bash
pytest
```

## Notes

The bundled evaluation files include a small demo PDF and page-level expected sources so you can test the loop immediately. Replace them with your own documents and questions for real use.
