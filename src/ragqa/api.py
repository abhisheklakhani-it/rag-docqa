"""FastAPI app: upload documents, then ask questions about them."""

import logging

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .chunking import chunk_pages
from .config import Settings
from .generator import ClaudeGenerator, extractive_answer
from .loaders import DocumentLoadError, load_document
from .retriever import TfidfRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ragqa")

settings = Settings()
retriever = TfidfRetriever.load(settings.index_dir)
generator = ClaudeGenerator(settings.model) if settings.use_llm else None

app = FastAPI(
    title="RAG Document QA",
    version=__version__,
    description="Upload documents, then ask questions answered from them with citations.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["How are client updates aggregated?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    source: str
    page: int | None
    chunk: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    mode: str  # "llm" or "extractive"
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "documents": len(retriever.sources),
        "chunks": len(retriever.chunks),
        "llm": settings.model if generator else None,
    }


@app.get("/documents")
def list_documents() -> list[str]:
    return retriever.sources


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile) -> dict:
    if not file.filename:
        raise HTTPException(400, "The upload has no file name")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
    try:
        doc = load_document(file.filename, data)
    except DocumentLoadError as e:
        raise HTTPException(400, str(e)) from e

    chunks = chunk_pages(
        doc.pages, file.filename, paged=doc.paged,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
    )
    replaced = retriever.remove_source(file.filename) > 0  # re-uploading a file replaces it
    retriever.add(chunks)
    retriever.save(settings.index_dir)
    logger.info("Indexed %s: %d pages, %d chunks", file.filename, len(doc.pages), len(chunks))
    return {"source": file.filename, "pages": len(doc.pages), "chunks": len(chunks), "replaced": replaced}


@app.delete("/documents/{name}")
def delete_document(name: str) -> dict:
    removed = retriever.remove_source(name)
    if not removed:
        raise HTTPException(404, f"Document {name!r} not found")
    retriever.save(settings.index_dir)
    return {"source": name, "removed_chunks": removed}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    hits = retriever.search(req.question, req.top_k or settings.top_k)
    contexts = [chunk for chunk, _ in hits]

    answer, mode = extractive_answer(contexts), "extractive"
    if generator and contexts:
        try:
            answer, mode = generator.answer(req.question, contexts), "llm"
        except anthropic.APIConnectionError:
            logger.exception("Could not reach the Claude API; using extractive answer")
        except anthropic.APIStatusError as e:
            logger.error("Claude API error %s; using extractive answer", e.status_code)

    return AskResponse(
        answer=answer,
        mode=mode,
        sources=[
            Source(source=c.source, page=c.page, chunk=c.index, score=round(s, 4), text=c.text)
            for c, s in hits
        ],
    )
