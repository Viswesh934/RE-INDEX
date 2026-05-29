from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import shutil

from .chunking import chunk_text_by_page
from .embeddings import embed_texts
from .extract import extract_pdf_pages
from .index import LocalFaissIndex, save_chunks_jsonl
from .storage import ChunkRecord, DocumentRecord, LocalStore


@dataclass
class IngestResult:
    document: DocumentRecord
    chunks_added: int
    pages_processed: int


def ingest_pdf(
    pdf_path: str | Path,
    store: LocalStore,
    faiss_index: LocalFaissIndex,
    model_name: str,
) -> IngestResult:
    pdf_path = Path(pdf_path)
    document_id = store.new_document_id()
    uploaded_at = store.now()
    target_path = store.paths.uploads_dir / f"{document_id}_{pdf_path.name}"
    shutil.copy2(pdf_path, target_path)

    pages = extract_pdf_pages(target_path)
    chunks: list[ChunkRecord] = []
    for page in pages:
        chunks.extend(
            chunk_text_by_page(
                document_id=document_id,
                file_name=pdf_path.name,
                page_number=page.page_number,
                page_text=page.text,
                source_type=page.source_type,
                timestamp=uploaded_at,
            )
        )

    store.save_document(
        DocumentRecord(
            document_id=document_id,
            file_name=pdf_path.name,
            original_path=str(target_path),
            uploaded_at=uploaded_at,
            num_pages=len(pages),
        )
    )
    save_chunks_jsonl(store, chunks)

    if chunks:
        embeddings = embed_texts([c.chunk_text for c in chunks], model_name=model_name)
        faiss_index.add(embeddings)
    else:
        faiss_index.load()

    return IngestResult(
        document=DocumentRecord(
            document_id=document_id,
            file_name=pdf_path.name,
            original_path=str(target_path),
            uploaded_at=uploaded_at,
            num_pages=len(pages),
        ),
        chunks_added=len(chunks),
        pages_processed=len(pages),
    )


def ingest_pdfs(
    pdf_paths: Iterable[str | Path],
    store: LocalStore,
    faiss_index: LocalFaissIndex,
    model_name: str,
) -> list[IngestResult]:
    results: list[IngestResult] = []
    for pdf_path in pdf_paths:
        results.append(ingest_pdf(pdf_path, store, faiss_index, model_name))
    return results
