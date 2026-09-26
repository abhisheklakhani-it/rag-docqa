from ragqa.chunking import chunk_pages
from ragqa.retriever import TfidfRetriever

DOCS = {
    "federated.md": "Federated learning aggregates client model updates with Federated Averaging.",
    "onnx.md": "ONNX Runtime quantization makes CPU inference faster.",
    "docker.md": "Docker packages an application into a portable container image.",
}


def build() -> TfidfRetriever:
    r = TfidfRetriever()
    for name, text in DOCS.items():
        r.add(chunk_pages([text], name, paged=False))
    return r


def test_most_relevant_document_ranks_first():
    r = build()
    assert r.search("How are client updates aggregated?")[0][0].source == "federated.md"
    assert r.search("faster CPU inference")[0][0].source == "onnx.md"
    assert r.search("what is a container image")[0][0].source == "docker.md"


def test_scores_are_sorted_and_k_is_respected():
    hits = build().search("model inference container", k=2)
    assert len(hits) <= 2
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_unrelated_query_returns_nothing():
    assert build().search("recipe for pizza") == []


def test_empty_index_returns_nothing():
    assert TfidfRetriever().search("anything") == []


def test_remove_source():
    r = build()
    assert r.remove_source("docker.md") == 1
    assert r.remove_source("docker.md") == 0
    assert "docker.md" not in r.sources


def test_save_and_load(tmp_path):
    build().save(tmp_path)
    loaded = TfidfRetriever.load(tmp_path)
    assert loaded.sources == sorted(DOCS)
    assert loaded.search("container image")[0][0].source == "docker.md"


def test_load_from_missing_directory_gives_empty_index(tmp_path):
    assert TfidfRetriever.load(tmp_path / "nope").chunks == []
