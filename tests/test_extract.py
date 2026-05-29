from pathlib import Path
import fitz

from app.extract import extract_pdf_pages


def test_extract_text_pdf(tmp_path):
    pdf_path = tmp_path / "text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world from PDF.")
    doc.save(pdf_path)
    doc.close()

    pages = extract_pdf_pages(pdf_path)
    assert len(pages) == 1
    assert pages[0].source_type == "pdf_text"
    assert "Hello world" in pages[0].text
