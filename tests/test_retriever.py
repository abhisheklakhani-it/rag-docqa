import numpy as np
import pytest

from ragqa.chunking import chunk_pages
from ragqa.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    TfidfRetriever,
    make_retriever,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = {
    "federated.md": "Federated learning aggregates client model updates with Federated Averaging.",
    "onnx.md": "ONNX Runtime quantization makes CPU inference faster.",
    "docker.md": "Docker packages an application into a portable container image.",
}
SYNONYMS = {"vehicle": "car", "automobile": "car"}


class FakeEmbedder:
    """Bag-of-words vectors with a tiny synonym table: shows that dense search matches meaning, not spelling."""

    def __init__(self):
        self.vocab = sorted({w for text in DOCS.values() for w in tokenize(text)} | {"car"})
        self.calls = 0

    def _vector(self, text):
        words = [SYNONYMS.get(w, w) for w in tokenize(text)]
        v = np.array([words.count(w) for w in self.vocab], dtype="float32")
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        self.calls += 1
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, len(self.vocab)), "float32")

    def embed_query(self, text):
        return self._vector(text)


def build(retriever):
    for name, text in DOCS.items():
        retriever.add(chunk_pages([text], name, paged=False))
    return retriever


ALL = [BM25Retriever, TfidfRetriever, lambda: DenseRetriever(FakeEmbedder()), lambda: HybridRetriever(FakeEmbedder())]
IDS = ["bm25", "tfidf", "dense", "hybrid"]


@pytest.mark.parametrize("factory", ALL, ids=IDS)
def test_most_relevant_document_ranks_first(factory):
    r = build(factory())
    assert r.search("How are client updates aggregated?")[0][0].source == "federated.md"
    assert r.search("faster CPU inference")[0][0].source == "onnx.md"
    assert r.search("what is a container image")[0][0].source == "docker.md"


@pytest.mark.parametrize("factory", ALL, ids=IDS)
def test_scores_are_sorted_and_k_is_respected(factory):
    hits = build(factory()).search("model inference container", k=2)
    assert len(hits) <= 2
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


@pytest.mark.parametrize("factory", ALL, ids=IDS)
def test_remove_source(factory):
    r = build(factory())
    assert r.remove_source("docker.md") == 1
    assert r.remove_source("docker.md") == 0
    assert "docker.md" not in r.sources
    assert all(c.source != "docker.md" for c, _ in r.search("container image", k=3))


@pytest.mark.parametrize("factory", ALL, ids=IDS)
def test_save_and_load(factory, tmp_path):
    build(factory()).save(tmp_path)
    loaded = factory().load_from(tmp_path)
    assert loaded.sources == sorted(DOCS)
    assert loaded.search("container image")[0][0].source == "docker.md"


@pytest.mark.parametrize("factory", ALL, ids=IDS)
def test_empty_index_returns_nothing(factory):
    assert factory().search("anything") == []


def test_lexical_retrievers_skip_unrelated_questions():
    assert build(BM25Retriever()).search("recipe for pizza") == []
    assert build(TfidfRetriever()).search("recipe for pizza") == []


def test_dense_min_score_filters_unrelated_questions():
    assert build(DenseRetriever(FakeEmbedder(), min_score=0.3)).search("recipe for pizza") == []


def test_dense_matches_meaning_where_bm25_cannot():
    docs = {"cars.md": "An electric car runs on a battery."}
    bm25, dense = BM25Retriever(), DenseRetriever(FakeEmbedder())
    for r in (bm25, dense):
        r.add(chunk_pages([docs["cars.md"]], "cars.md", paged=False))
    assert bm25.search("electric vehicle")[0][0].source == "cars.md"   # "electric" matches
    assert bm25.search("vehicle") == []                                # no shared word
    assert dense.search("vehicle")[0][0].source == "cars.md"           # synonym matched by meaning


def test_saved_vectors_are_reused_after_restart(tmp_path):
    build(HybridRetriever(FakeEmbedder())).save(tmp_path)
    embedder = FakeEmbedder()
    HybridRetriever(embedder).load_from(tmp_path)
    assert embedder.calls == 0  # nothing re-embedded


def test_rrf_rewards_chunks_ranked_well_by_both():
    a, b, c = (chunk_pages([t], n, paged=False)[0] for n, t in DOCS.items())
    fused = reciprocal_rank_fusion([[(a, 9.0), (b, 5.0), (c, 1.0)], [(b, 0.9), (c, 0.8), (a, 0.1)]])
    assert [ch.source for ch, _ in fused] == ["onnx.md", "federated.md", "docker.md"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_tokenize_stems_and_drops_stop_words():
    assert tokenize("The cells are running") == ["cell", "run"]


def test_make_retriever():
    assert make_retriever("bm25").name == "bm25"
    assert make_retriever("hybrid", embedder_factory=FakeEmbedder).name == "hybrid"
    with pytest.raises(ValueError, match="Unknown retriever"):
        make_retriever("magic")


def test_bm25_works_with_a_single_chunk():
    """Regression: the classic Okapi IDF is negative when a word is in over half the chunks, so a
    one-document index returned nothing. Lucene's IDF is always positive."""
    r = BM25Retriever()
    r.add(chunk_pages(["An electric car runs on a battery."], "car.md", paged=False))
    assert r.search("battery")[0][0].source == "car.md"


def test_bm25_rewards_rare_words_and_saturates_repeats():
    r = BM25Retriever()
    r.add(chunk_pages(["model model model model"], "repeat.md", paged=False))
    r.add(chunk_pages(["model quantization"], "rare.md", paged=False))
    r.add(chunk_pages(["model training"], "other.md", paged=False))
    assert r.search("model quantization")[0][0].source == "rare.md"  # the rare word beats repetition
