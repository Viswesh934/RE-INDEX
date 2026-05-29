from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image

from .ocr import ocr_pil_image


@dataclass
class PageExtraction:
    page_number: int
    text: str
    source_type: str  # pdf_text or ocr
    is_scanned: bool


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_scanned(text: str, min_chars: int = 12) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < min_chars:
        return True
    alpha = sum(ch.isalpha() for ch in text)
    return alpha < 3


def extract_pdf_pages(pdf_path: str | Path, ocr_lang: str = "eng") -> list[PageExtraction]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages: list[PageExtraction] = []

    for i, page in enumerate(doc, start=1):
        text = _clean_text(page.get_text("text") or "")
        scanned = _looks_scanned(text)

        if scanned:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            text = _clean_text(ocr_pil_image(image, lang=ocr_lang))
            source_type = "ocr"
        else:
            source_type = "pdf_text"

        pages.append(PageExtraction(page_number=i, text=text, source_type=source_type, is_scanned=scanned))

    doc.close()
    return pages
