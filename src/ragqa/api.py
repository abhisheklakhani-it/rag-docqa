"""FastAPI app: upload documents, then ask questions about them."""

import json
import logging
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .chunking import chunk_pages
from .config import Settings
from .generator import ClaudeGenerator, extractive_answer
from .loaders import DocumentLoadError, load_document
from .retriever import FastEmbedder, make_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ragqa")

settings = Settings()
retriever = make_retriever(
    settings.retriever,
    embedder_factory=lambda: FastEmbedder(settings.embedding_model, settings.embedding_cache_dir),
    min_dense_score=settings.min_dense_score,
).load_from(settings.index_dir)
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
        "retriever": retriever.name,
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
        doc = load_document(file.filename, data, ocr=settings.ocr)
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
    return {"source": file.filename, "pages": len(doc.pages), "chunks": len(chunks), "replaced": replaced,
            "ocr_pages": doc.ocr_pages}


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

    return AskResponse(answer=answer, mode=mode, sources=to_sources(hits))


def to_sources(hits) -> list[Source]:
    return [Source(source=c.source, page=c.page, chunk=c.index, score=round(s, 4), text=c.text) for c, s in hits]


def sse(event: str, data) -> str:
    """One Server-Sent Event. Data is JSON, so newlines inside the text can't break the format."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Like /ask, but streams the answer as Server-Sent Events: `sources`, then `token`s, then `done`."""
    hits = retriever.search(req.question, req.top_k or settings.top_k)
    contexts = [chunk for chunk, _ in hits]

    def events():
        yield sse("sources", [s.model_dump() for s in to_sources(hits)])
        sent = False
        if generator and contexts:
            try:
                for text in generator.stream(req.question, contexts):
                    sent = True
                    yield sse("token", {"text": text})
                yield sse("done", {"mode": "llm"})
                return
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
                logger.error("LLM streaming failed (%s); falling back", type(e).__name__)
                if sent:  # part of the answer is already on screen: say it stopped
                    yield sse("error", {"message": "The answer was interrupted. Please try again."})
                    return
        yield sse("token", {"text": extractive_answer(contexts)})
        yield sse("done", {"mode": "extractive"})

    # no-cache and no proxy buffering, so each piece reaches the browser immediately
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    """A minimal page to upload documents and watch answers stream in."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")
