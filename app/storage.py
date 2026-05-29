from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import time
import uuid


@dataclass
class ChunkRecord:
    document_id: str
    file_name: str
    page_number: int
    chunk_id: str
    chunk_text: str
    source_type: str
    timestamp: float
    section_heading: str | None = None
    chunk_index: int = 0
    semantic_score: float = 0.0
    keyword_score: float = 0.0

    def citation(self) -> str:
        return f"Source: {self.file_name}, p. {self.page_number}"


@dataclass
class DocumentRecord:
    document_id: str
    file_name: str
    original_path: str
    uploaded_at: float
    num_pages: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoragePaths:
    root: Path
    uploads_dir: Path = field(init=False)
    index_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    documents_json: Path = field(init=False)
    chunks_jsonl: Path = field(init=False)
    faiss_index: Path = field(init=False)

    def __post_init__(self) -> None:
        self.uploads_dir = self.root / "data" / "uploads"
        self.index_dir = self.root / "data" / "index"
        self.metadata_dir = self.root / "data" / "metadata"
        self.documents_json = self.metadata_dir / "documents.json"
        self.chunks_jsonl = self.metadata_dir / "chunks.jsonl"
        self.faiss_index = self.index_dir / "index.faiss"


class LocalStore:
    def __init__(self, root: str | Path):
        self.paths = StoragePaths(Path(root))
        self.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.paths.index_dir.mkdir(parents=True, exist_ok=True)
        self.paths.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_document(self, doc: DocumentRecord) -> None:
        docs = self._read_json(self.paths.documents_json, [])
        docs = [d for d in docs if d["document_id"] != doc.document_id]
        docs.append(asdict(doc))
        self._write_json(self.paths.documents_json, docs)

    def load_documents(self) -> list[DocumentRecord]:
        docs = self._read_json(self.paths.documents_json, [])
        return [DocumentRecord(**d) for d in docs]

    def append_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        with self.paths.chunks_jsonl.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    def load_chunks(self) -> list[ChunkRecord]:
        if not self.paths.chunks_jsonl.exists():
            return []
        chunks: list[ChunkRecord] = []
        with self.paths.chunks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(ChunkRecord(**json.loads(line)))
        return chunks

    def new_document_id(self) -> str:
        return uuid.uuid4().hex

    def now(self) -> float:
        return time.time()
