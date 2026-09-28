import json
import math

import pytest

from ragqa.evaluation import load_beir, ndcg_at_k, reciprocal_rank, recall_at_k, score_run, unique_sources

RANKED = ["d3", "d1", "d7", "d2"]
RELEVANT = {"d1": 1, "d2": 1}


def test_unique_sources_keeps_best_rank():
    assert unique_sources(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_and_reciprocal_rank():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 4) == 1.0
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5
    assert reciprocal_rank(["x"], RELEVANT) == 0.0


def test_ndcg_by_hand():
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT) == pytest.approx(expected)
    assert ndcg_at_k(["d1", "d2"], RELEVANT) == 1.0


def test_score_run_averages_and_missing_queries_score_zero():
    result = score_run({"q1": ["d1"]}, {"q1": {"d1": 1}, "q2": {"d9": 1}})
    assert result["recall@1"] == 0.5 and result["mrr@10"] == 0.5


def test_load_beir_from_local_folder(tmp_path):
    folder = tmp_path / "tiny"
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "title": "Cats", "text": "Cats purr."}))
    (folder / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "Do cats purr?"}) + "\n" + json.dumps({"_id": "q2", "text": "unlabelled"}))
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    data = load_beir("tiny", "test", tmp_path)  # corpus exists locally, so nothing is downloaded
    assert data.corpus == {"d1": "Cats\n\nCats purr."}
    assert data.queries == {"q1": "Do cats purr?"}
    assert data.qrels == {"q1": {"d1": 1}}
