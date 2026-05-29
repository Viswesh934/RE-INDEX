from __future__ import annotations

from PIL import Image
import pytesseract


def ocr_pil_image(image: Image.Image, lang: str = "eng") -> str:
    """OCR a PIL image using Tesseract."""
    return pytesseract.image_to_string(image, lang=lang)
