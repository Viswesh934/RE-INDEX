from app.chunking import chunk_text_by_page


def test_chunking_preserves_metadata():
    chunks = chunk_text_by_page(
        document_id="doc1",
        file_name="sample.pdf",
        page_number=2,
        page_text="Sentence one. Sentence two. Sentence three.",
        source_type="pdf_text",
        timestamp=123.0,
        chunk_size_chars=25,
        overlap_chars=5,
    )
    assert chunks
    assert all(c.file_name == "sample.pdf" for c in chunks)
    assert all(c.page_number == 2 for c in chunks)
    assert all(c.source_type == "pdf_text" for c in chunks)
