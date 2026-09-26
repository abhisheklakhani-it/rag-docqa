"""TF-IDF retriever: a fast, lightweight baseline that needs no GPU or embedding API."""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .chunking import Chunk

INDEX_FILE = "chunks.joblib"


class TfidfRetriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._fit()

    def remove_source(self, source: str) -> int:
        """Remove every chunk of one document. Returns how many were removed."""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != source]
        self._fit()
        return before - len(self.chunks)

    def _fit(self) -> None:
        # Rebuild the TF-IDF matrix from all chunks. Fast enough for thousands of chunks.
        if not self.chunks:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, score) pairs, best first. Chunks with score 0 are skipped."""
        if self._vectorizer is None:
            return []
        # TF-IDF vectors are L2-normalised, so the dot product equals cosine similarity.
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def save(self, directory: str | Path) -> None:
        # Only the chunks are saved; the TF-IDF matrix is rebuilt on load.
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump([asdict(c) for c in self.chunks], Path(directory) / INDEX_FILE)

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfRetriever":
        retriever = cls()
        path = Path(directory) / INDEX_FILE
        if path.exists():
            retriever.add([Chunk(**c) for c in joblib.load(path)])
        return retriever
