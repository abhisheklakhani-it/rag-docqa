"""Settings read from environment variables, so behaviour can change without code changes."""

import os
from dataclasses import dataclass, field


def _llm_enabled() -> bool:
    mode = os.getenv("RAGQA_LLM", "auto").lower()
    if mode == "none":
        return False
    if mode == "claude":
        return True
    # "auto": use Claude only when credentials are available
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


@dataclass(frozen=True)
class Settings:
    index_dir: str = field(default_factory=lambda: os.getenv("RAGQA_INDEX_DIR", "index"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAGQA_CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAGQA_CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAGQA_TOP_K", "4")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("RAGQA_MAX_UPLOAD_MB", "20")))
    model: str = field(default_factory=lambda: os.getenv("RAGQA_MODEL", "claude-opus-5"))
    use_llm: bool = field(default_factory=_llm_enabled)
    # Retrieval: "hybrid" (BM25 + embeddings), "bm25", "dense", or "tfidf" (the original baseline).
    retriever: str = field(default_factory=lambda: os.getenv("RAGQA_RETRIEVER", "hybrid").lower())
    embedding_model: str = field(default_factory=lambda: os.getenv("RAGQA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_cache_dir: str | None = field(default_factory=lambda: os.getenv("RAGQA_EMBEDDING_CACHE_DIR") or None)
    # Dense hits below this cosine similarity are ignored, so unrelated questions find nothing. 0.60 was
    # calibrated for bge-small-en-v1.5 (scripts/calibrate_min_score.py); recalibrate if you change the model.
    min_dense_score: float = field(default_factory=lambda: float(os.getenv("RAGQA_MIN_DENSE_SCORE", "0.60")))
    # OCR for scanned PDF pages that have no text layer: "auto" (default) or "off".
    ocr: bool = field(default_factory=lambda: os.getenv("RAGQA_OCR", "auto").lower() != "off")
