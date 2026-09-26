from ragqa.chunking import chunk_pages


def test_chunks_respect_size_limit():
    chunks = chunk_pages(["word " * 500], "doc.txt", paged=False, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_neighbouring_chunks_overlap():
    text = " ".join(f"sentence{i}." for i in range(200))
    first, second = chunk_pages([text], "doc.txt", paged=False, chunk_size=200, chunk_overlap=50)[:2]
    assert first.text[-30:] in second.text


def test_page_numbers_and_indexes():
    chunks = chunk_pages(["first page", "second page"], "doc.pdf")
    assert [c.page for c in chunks] == [1, 2]
    assert [c.index for c in chunks] == [0, 1]
    assert all(c.source == "doc.pdf" for c in chunks)


def test_unpaged_documents_have_no_page_numbers():
    chunks = chunk_pages(["just text"], "notes.md", paged=False)
    assert chunks[0].page is None
