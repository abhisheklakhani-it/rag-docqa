"""Choose RAGQA_MIN_DENSE_SCORE from data: dense search always returns something, so off-topic questions
need a similarity floor. The floor should keep almost every correct match and drop off-topic questions.

  python scripts/calibrate_min_score.py      # needs the vectors cached by evaluate_retrieval.py --retriever dense
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragqa.chunking import chunk_pages  # noqa: E402
from ragqa.evaluation import load_beir  # noqa: E402
from ragqa.retriever import FastEmbedder  # noqa: E402

KEEP = 0.99  # share of correct matches the floor must keep


def main() -> None:
    data = load_beir("scifact", "test", ROOT / "data" / "eval")
    chunks = [c for doc_id, text in data.corpus.items() for c in chunk_pages([text], doc_id, paged=False)]
    vectors = np.load(ROOT / "data" / "eval" / "cache" / "scifact-800-120.npy")
    assert len(vectors) == len(chunks), "cache does not match the chunking; rerun evaluate_retrieval.py"
    rows_by_doc: dict[str, list[int]] = {}
    for row, chunk in enumerate(chunks):
        rows_by_doc.setdefault(chunk.source, []).append(row)

    embedder = FastEmbedder()
    # Score of the best chunk of each correct document, for every labelled question.
    correct = []
    for qid, question in data.queries.items():
        scores = vectors @ embedder.embed_query(question)
        correct += [scores[rows_by_doc[doc]].max() for doc in data.qrels[qid]]
    # Best score of any chunk for questions that have nothing to do with the documents.
    off_topic_questions = (ROOT / "data" / "eval_offtopic" / "questions.txt").read_text().split("\n")
    off_topic = [float((vectors @ embedder.embed_query(q)).max()) for q in off_topic_questions if q.strip()]

    correct, off_topic = np.array(correct), np.array(off_topic)
    floor = round(float(np.floor(np.quantile(correct, 1 - KEEP) * 100) / 100), 2)  # rounded down to 2 decimals
    result = {
        "correct_matches": len(correct), "off_topic_questions": len(off_topic),
        "correct_score_percentiles": {p: round(float(np.quantile(correct, p / 100)), 3) for p in (1, 5, 50)},
        "off_topic_best_score": {"median": round(float(np.median(off_topic)), 3), "max": round(float(off_topic.max()), 3)},
        "chosen_floor": floor,
        "correct_kept": round(float((correct >= floor).mean()), 3),
        "off_topic_rejected": round(float((off_topic < floor).mean()), 3),
    }
    (ROOT / "results" / "min_dense_score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
