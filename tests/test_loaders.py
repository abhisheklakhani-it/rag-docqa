import pytest

from conftest import make_pdf, make_scanned_pdf
from ragqa.loaders import DocumentLoadError, load_document


def test_pdf_is_loaded_page_by_page(pdf_bytes):
    doc = load_document("guide.pdf", pdf_bytes)
    assert doc.paged
    assert len(doc.pages) == 2
    assert "3x faster" in doc.pages[1]


@pytest.mark.parametrize("name", ["notes.md", "notes.txt", "NOTES.MD"])
def test_text_files_are_one_unpaged_page(name):
    doc = load_document(name, "Hello RAG".encode())
    assert doc.pages == ["Hello RAG"]
    assert not doc.paged


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("model.bin", b"x", "Unsupported file type"),
        ("noextension", b"x", "Unsupported file type"),
        ("empty.txt", b"   ", "empty"),
        ("broken.pdf", b"not a pdf", "not a readable PDF"),
        ("blank.pdf", make_pdf([""]), "OCR found no text either"),
    ],
)
def test_bad_files_are_rejected(name, data, message):
    with pytest.raises(DocumentLoadError, match=message):
        load_document(name, data)


def test_scanned_pdf_is_read_with_ocr(scanned_pdf):
    doc = load_document("scan.pdf", scanned_pdf)
    assert doc.ocr_pages == [1]
    assert "three times faster" in doc.pages[0]
    assert "Quarterly Report 2026" in doc.pages[0]


def test_ocr_only_runs_on_pages_without_text(pdf_bytes):
    assert load_document("guide.pdf", pdf_bytes).ocr_pages == []


def test_scanned_pdf_with_ocr_off_is_rejected(scanned_pdf):
    with pytest.raises(DocumentLoadError, match="OCR is turned off"):
        load_document("scan.pdf", scanned_pdf, ocr=False)


def test_mixed_pdf_keeps_page_numbers():
    """Page 1 has a text layer, page 2 is a scan: both are kept with the right page numbers."""
    from pypdf import PdfReader, PdfWriter
    import io

    writer = PdfWriter()
    for data in (make_pdf(["Typed page about retrieval."]), make_scanned_pdf(["Scanned page about embeddings."])):
        writer.add_page(PdfReader(io.BytesIO(data)).pages[0])
    out = io.BytesIO()
    writer.write(out)
    doc = load_document("mixed.pdf", out.getvalue())
    assert doc.ocr_pages == [2]
    assert "retrieval" in doc.pages[0] and "embeddings" in doc.pages[1]
