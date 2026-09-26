import pytest

from conftest import make_pdf
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
        ("scan.pdf", make_pdf([""]), "no extractable text"),
    ],
)
def test_bad_files_are_rejected(name, data, message):
    with pytest.raises(DocumentLoadError, match=message):
        load_document(name, data)
