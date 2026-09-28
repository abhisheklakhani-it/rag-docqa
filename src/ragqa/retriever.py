"""Retrievers: BM25 (words), dense (meaning), hybrid (both, fused with RRF), and the original TF-IDF.

All share one interface: add(chunks), remove_source(name), search(query, k), save(dir), load(dir).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from .chunking import Chunk

INDEX_FILE = "chunks.joblib"
EMBEDDINGS_FILE = "embeddings.npy"
Hits = list[tuple[Chunk, float]]

# A small English stop-word list (the same idea as scikit-learn's, without the dependency).
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())
TOKEN = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer

    return Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase words without stop words, stemmed ("cells" -> "cell"), for BM25."""
    return _stemmer().stemWords([w for w in TOKEN.findall(text.lower()) if w not in STOP_WORDS])


class Retriever(ABC):
    name = "retriever"

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._on_change(added=chunks)

    def remove_source(self, source: str) -> int:
        """Remove every chunk of one document. Returns how many were removed."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self._on_change(kept=keep)
        return removed

    @abstractmethod
    def _on_change(self, added: list[Chunk] | None = None, kept: list[int] | None = None) -> None:
        """Update the index after chunks were added or (with `kept` row numbers) removed."""

    @abstractmethod
    def search(self, query: str, k: int = 4) -> Hits:
        """Return up to k (chunk, score) pairs, best first."""

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump([asdict(c) for c in self.chunks], Path(directory) / INDEX_FILE)

    def load_from(self, directory: str | Path) -> "Retriever":
        path = Path(directory) / INDEX_FILE
        if path.exists():
            self.add([Chunk(**c) for c in joblib.load(path)])
        return self


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Row numbers of the k highest scores, best first."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    best = np.argpartition(-scores, k - 1)[:k]
    return best[np.argsort(-scores[best], kind="stable")]


class BM25Retriever(Retriever):
    """Okapi BM25 over stemmed words, with an inverted index. Chunks sharing no word with the query are skipped.

    score(chunk) = sum over query words of  idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_length))
    k1 limits how much repeating a word helps; b penalises long chunks.
    idf uses Lucene's formula log(1 + (N - n + 0.5) / (n + 0.5)), which is never negative. (The classic
    formula without "1 +" goes negative for words in over half the chunks, e.g. in a one-document index.)
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.k1, self.b = k1, b
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # word -> (chunk rows, counts)
        self._lengths = np.zeros(0)

    def _on_change(self, added=None, kept=None) -> None:
        # Rebuilding is fast (a few seconds for 17,000 chunks) and keeps the statistics exact.
        rows: dict[str, list[int]] = {}
        counts: dict[str, list[int]] = {}
        lengths = []
        for row, chunk in enumerate(self.chunks):
            words = tokenize(chunk.text)
            lengths.append(len(words))
            for word in set(words):
                rows.setdefault(word, []).append(row)
                counts.setdefault(word, []).append(words.count(word))
        self._postings = {w: (np.array(rows[w]), np.array(counts[w], dtype=float)) for w in rows}
        self._lengths = np.array(lengths, dtype=float)

    def scores(self, query: str) -> np.ndarray:
        n_chunks = len(self.chunks)
        scores = np.zeros(n_chunks)
        if not n_chunks:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self._lengths / max(self._lengths.mean(), 1e-9))
        for word in tokenize(query):
            if word not in self._postings:
                continue
            rows, tf = self._postings[word]
            idf = np.log(1 + (n_chunks - len(rows) + 0.5) / (len(rows) + 0.5))
            scores[rows] += idf * tf * (self.k1 + 1) / (tf + norm[rows])
        return scores

    def search(self, query: str, k: int = 4) -> Hits:
        scores = self.scores(query)
        return [(self.chunks[i], float(scores[i])) for i in top_k(scores, k) if scores[i] > 0]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Sentence embeddings with fastembed (ONNX Runtime, no PyTorch). Vectors are L2-normalised."""

    # BGE models were trained with this instruction in front of search queries (see the model card).
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name, cache_dir=cache_dir)
        self.query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts, batch_size=64)), dtype="float32").reshape(len(texts), -1)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + text])[0]


class DenseRetriever(Retriever):
    """Cosine similarity between embeddings. Vectors are saved next to the chunks, so restarts don't re-embed."""

    name = "dense"

    def __init__(self, embedder: Embedder, min_score: float = 0.0) -> None:
        super().__init__()
        self.embedder, self.min_score = embedder, min_score
        self._vectors = np.zeros((0, 0), dtype="float32")
        self._preloaded: np.ndarray | None = None

    def _on_change(self, added=None, kept=None) -> None:
        if kept is not None:
            self._vectors = self._vectors[kept]
        elif added:
            if self._preloaded is not None and len(self._preloaded) == len(added):
                new = self._preloaded  # vectors saved with the index
            else:
                new = self.embedder.embed_documents([c.text for c in added])
            self._preloaded = None
            self._vectors = new if not len(self._vectors) else np.vstack([self._vectors, new])

    def search(self, query: str, k: int = 4) -> Hits:
        if not self.chunks:
            return []
        scores = self._vectors @ self.embedder.embed_query(query)  # cosine, since vectors are normalised
        return [(self.chunks[i], float(scores[i])) for i in top_k(scores, k) if scores[i] >= self.min_score]

    def save(self, directory: str | Path) -> None:
        super().save(directory)
        np.save(Path(directory) / EMBEDDINGS_FILE, self._vectors)

    def load_from(self, directory: str | Path) -> "Retriever":
        path = Path(directory) / EMBEDDINGS_FILE
        if path.exists():
            self._preloaded = np.load(path)
        return super().load_from(directory)


def reciprocal_rank_fusion(rankings: list[Hits], k: int = 60) -> Hits:
    """score(chunk) = sum of 1 / (k + rank) over the rankings. Uses ranks only, so no score scaling is needed."""
    fused: dict[tuple[str, int], float] = {}
    by_key: dict[tuple[str, int], Chunk] = {}
    for hits in rankings:
        for rank, (chunk, _) in enumerate(hits, 1):
            key = (chunk.source, chunk.index)
            by_key[key] = chunk
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(by_key[key], score) for key, score in ordered]


class HybridRetriever(Retriever):
    """BM25 + dense, merged with Reciprocal Rank Fusion (best method in the retrieval-eval-bench project)."""

    name = "hybrid"

    def __init__(self, embedder: Embedder, min_dense_score: float = 0.0, depth: int = 50) -> None:
        super().__init__()
        self.lexical = BM25Retriever()
        self.dense = DenseRetriever(embedder, min_score=min_dense_score)
        self.depth = depth

    def _on_change(self, added=None, kept=None) -> None:
        for part in (self.lexical, self.dense):
            part.chunks = self.chunks
            part._on_change(added=added, kept=kept)

    def search(self, query: str, k: int = 4) -> Hits:
        fused = reciprocal_rank_fusion([self.lexical.search(query, self.depth), self.dense.search(query, self.depth)])
        return fused[:k]

    def save(self, directory: str | Path) -> None:
        self.dense.save(directory)  # chunks + vectors

    def load_from(self, directory: str | Path) -> "Retriever":
        path = Path(directory) / EMBEDDINGS_FILE
        if path.exists():
            self.dense._preloaded = np.load(path)
        return super().load_from(directory)


class TfidfRetriever(Retriever):
    """The original baseline: TF-IDF (1-2 grams) with cosine similarity. Needs scikit-learn."""

    name = "tfidf"

    def __init__(self) -> None:
        super().__init__()
        self._vectorizer, self._matrix = None, None

    def _on_change(self, added=None, kept=None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self.chunks:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 4) -> Hits:
        if self._vectorizer is None:
            return []
        scores = (self._vectorizer.transform([query]) @ self._matrix.T).toarray().ravel()
        return [(self.chunks[i], float(scores[i])) for i in top_k(scores, k) if scores[i] > 0]

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfRetriever":
        return cls().load_from(directory)


def make_retriever(kind: str, embedder_factory=None, min_dense_score: float = 0.0) -> Retriever:
    """Build a retriever by name. `embedder_factory` is only called for dense/hybrid (it loads a model)."""
    if kind == "bm25":
        return BM25Retriever()
    if kind == "tfidf":
        return TfidfRetriever()
    if kind in ("dense", "hybrid"):
        embedder = (embedder_factory or FastEmbedder)()
        if kind == "dense":
            return DenseRetriever(embedder, min_score=min_dense_score)
        return HybridRetriever(embedder, min_dense_score=min_dense_score)
    raise ValueError(f"Unknown retriever {kind!r}; use hybrid, bm25, dense or tfidf")
