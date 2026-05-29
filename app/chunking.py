from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .storage import ChunkRecord


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def chunk_text_by_page(
    *,
    document_id: str,
    file_name: str,
    page_number: int,
    page_text: str,
    source_type: str,
    timestamp: float,
    chunk_size_chars: int = 900,
    overlap_chars: int = 120,
) -> list[ChunkRecord]:
    """Chunk a single page while keeping document and page metadata intact."""
    text = page_text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        proposed = current + " " + sent
        if len(proposed) <= chunk_size_chars:
            current = proposed
        else:
            chunks.append(current.strip())
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + " " + sent).strip() if tail else sent

    if current.strip():
        chunks.append(current.strip())

    records: list[ChunkRecord] = []
    for idx, chunk in enumerate(chunks):
        records.append(
            ChunkRecord(
                document_id=document_id,
                file_name=file_name,
                page_number=page_number,
                chunk_id=f"{document_id}-p{page_number:04d}-c{idx:03d}",
                chunk_text=chunk,
                source_type=source_type,
                timestamp=timestamp,
                chunk_index=idx,
            )
        )
    return records
