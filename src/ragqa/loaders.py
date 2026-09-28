"""Turn uploaded files into page texts that the chunker understands."""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


class DocumentLoadError(ValueError):
    """Raised when a file can't be turned into text."""


@dataclass
class LoadedDocument:
    pages: list[str]
    paged: bool  # True when page numbers are meaningful (PDFs)
    ocr_pages: list[int] = field(default_factory=list)  # 1-based pages whose text came from OCR


def load_document(filename: str, data: bytes, ocr: bool = True) -> LoadedDocument:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            raise DocumentLoadError(f"{filename} is empty")
        return LoadedDocument(pages=[text], paged=False)
    if ext in PDF_EXTENSIONS:
        return _load_pdf(filename, data, ocr)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise DocumentLoadError(f"Unsupported file type {ext or '(none)'}; supported: {supported}")


def _load_pdf(filename: str, data: bytes, ocr: bool) -> LoadedDocument:
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise DocumentLoadError(f"{filename} is not a readable PDF: {e}") from e

    # Scanned pages are images of text with no text layer: read them with OCR.
    # Pages that already have text skip OCR, which is slower and can misread characters.
    empty = [i for i, text in enumerate(pages) if not text.strip()]
    ocr_pages = []
    if empty and ocr:
        from .ocr import ocr_pdf_pages

        for i, text in ocr_pdf_pages(data, empty).items():
            if text.strip():
                pages[i] = text
                ocr_pages.append(i + 1)

    if not any(p.strip() for p in pages):
        reason = "OCR found no text either" if ocr else "OCR is turned off (RAGQA_OCR=off)"
        raise DocumentLoadError(f"{filename} contains no extractable text; {reason}")
    return LoadedDocument(pages=pages, paged=True, ocr_pages=ocr_pages)
