"""Turn uploaded files into page texts that the chunker understands."""

from dataclasses import dataclass
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


def load_document(filename: str, data: bytes) -> LoadedDocument:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            raise DocumentLoadError(f"{filename} is empty")
        return LoadedDocument(pages=[text], paged=False)
    if ext in PDF_EXTENSIONS:
        return LoadedDocument(pages=_pdf_pages(filename, data), paged=True)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise DocumentLoadError(f"Unsupported file type {ext or '(none)'}; supported: {supported}")


def _pdf_pages(filename: str, data: bytes) -> list[str]:
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise DocumentLoadError(f"{filename} is not a readable PDF: {e}") from e
    if not any(p.strip() for p in pages):
        # Scanned PDFs are images of text; they'd need OCR, which is out of scope here.
        raise DocumentLoadError(f"{filename} contains no extractable text (is it a scanned PDF?)")
    return pages
