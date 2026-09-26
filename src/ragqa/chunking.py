"""Split documents into small, overlapping chunks for retrieval."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    source: str  # file name the chunk came from
    page: int | None  # 1-based page number (None for plain-text files)
    index: int  # position of the chunk within its document


def chunk_pages(
    pages: list[str],
    source: str,
    *,
    paged: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk a document given as a list of page texts.

    For documents without real pages (.txt, .md), pass the whole text as one
    page with paged=False so chunks get page=None.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        for text in splitter.split_text(page_text):
            chunks.append(Chunk(text=text, source=source, page=page_no if paged else None, index=len(chunks)))
    return chunks
