"""
document_loader.py
==================
Universal document → text extractor.

Supported types:
  Images  : jpg, jpeg, png, bmp, tiff, tif, webp  → EasyOCR
  PDF     : pdf  → PyMuPDF text extraction; scanned pages fall back to EasyOCR
  Word    : docx → python-docx
  Text    : txt, md, markdown, csv, json, xml, html, htm → read as-is
"""

from __future__ import annotations

import io
import tempfile
import os
from pathlib import Path
from typing import Optional

# ── image extensions that go through EasyOCR ──────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── plain-text extensions read directly ───────────────────────────────────────
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".xml",
             ".html", ".htm", ".rst", ".log", ".yaml", ".yml"}


def extract_text(
    file_bytes: bytes,
    filename: str,
    ocr_model=None,       # easyocr.Reader instance (required for images / scanned PDFs)
) -> str:
    """
    Extract text from *any* supported document.

    Parameters
    ----------
    file_bytes : raw bytes of the uploaded/read file
    filename   : original file name (used to detect extension)
    ocr_model  : an initialised easyocr.Reader; required for image & scanned-PDF paths

    Returns
    -------
    Extracted text as a single string.
    """
    ext = Path(filename).suffix.lower()

    if ext in IMAGE_EXTS:
        return _ocr_image_bytes(file_bytes, ocr_model)

    if ext == ".pdf":
        return _extract_pdf(file_bytes, ocr_model)

    if ext == ".docx":
        return _extract_docx(file_bytes)

    if ext in TEXT_EXTS:
        return _read_text_bytes(file_bytes)

    # Unknown extension — try plain text first, else OCR
    try:
        return _read_text_bytes(file_bytes)
    except Exception:
        if ocr_model:
            return _ocr_image_bytes(file_bytes, ocr_model)
        return "Unsupported file type and no OCR model available."


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_image_bytes(image_bytes: bytes, ocr_model) -> str:
    """Run EasyOCR on raw image bytes."""
    if ocr_model is None:
        raise ValueError("ocr_model is required for image files.")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name
    try:
        results = ocr_model.readtext(tmp_path)
        return "\n".join(item[1] for item in results) if results else "No text detected."
    finally:
        os.unlink(tmp_path)


def _extract_pdf(pdf_bytes: bytes, ocr_model=None) -> str:
    """
    Extract text from a PDF.
    - Pages with selectable text → extracted directly (fast, accurate).
    - Pages without text (scanned images) → rendered and OCR'd if ocr_model given.
    """
    import fitz  # PyMuPDF

    pages_text: list[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if text:
            pages_text.append(f"--- Page {page_num} ---\n{text}")
        elif ocr_model:
            # Scanned page: render to image then OCR
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            ocr_text = _ocr_image_bytes(img_bytes, ocr_model)
            pages_text.append(f"--- Page {page_num} (OCR) ---\n{ocr_text}")
        else:
            pages_text.append(f"--- Page {page_num} ---\n[Scanned page — no OCR model available]")

    doc.close()
    return "\n\n".join(pages_text) if pages_text else "No text found in PDF."


def _extract_docx(docx_bytes: bytes) -> str:
    """Extract text from a .docx file using python-docx."""
    from docx import Document

    doc = Document(io.BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs) if paragraphs else "No text found in document."


def _read_text_bytes(raw: bytes) -> str:
    """Decode raw bytes as UTF-8 (with fallback to latin-1)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper (used by ocr_system.py)
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_path(file_path: str, ocr_model=None) -> str:
    """Convenience wrapper that reads a file from disk and calls extract_text."""
    path = Path(file_path)
    file_bytes = path.read_bytes()
    return extract_text(file_bytes, path.name, ocr_model)
