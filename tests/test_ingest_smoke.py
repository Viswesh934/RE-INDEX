from pathlib import Path
import fitz
import numpy as np

from app.index import LocalFaissIndex
from app.ingest import ingest_pdf
from app.storage import LocalStore


def test_ingest_smoke(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Refund policy is 30 days.")
    doc.save(pdf_path)
    doc.close()

    store = LocalStore(tmp_path)
    idx = LocalFaissIndex(store)

    monkeypatch.setattr("app.ingest.embed_texts", lambda texts, model_name: np.ones((len(texts), 384), dtype="float32"))
    result = ingest_pdf(pdf_path, store, idx, model_name="dummy")
    assert result.pages_processed == 1
    assert result.chunks_added >= 1
