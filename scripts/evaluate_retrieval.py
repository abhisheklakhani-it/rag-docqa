"""Evaluate the app's retrieval on a labelled question set (default: BEIR SciFact test split).

  python scripts/evaluate_retrieval.py                     # current retriever
  python scripts/evaluate_retrieval.py --retriever tfidf   # a specific one

The dataset is downloaded to data/eval/ (gitignored). Results are appended to results/retrieval_eval.json.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragqa.chunking import chunk_pages  # noqa: E402
from ragqa.evaluation import load_beir, score_run, unique_sources  # noqa: E402
from ragqa.retriever import make_retriever  # noqa: E402

RESULTS = ROOT / "results" / "retrieval_eval.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--retriever", default="hybrid", choices=["hybrid", "bm25", "dense", "tfidf"])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--depth", type=int, default=50, help="chunks retrieved per question before de-duplication")
    args = parser.parse_args()

    data = load_beir(args.dataset, args.split, ROOT / "data" / "eval")
    chunks = [c for doc_id, text in data.corpus.items()
              for c in chunk_pages([text], doc_id, paged=False,
                                   chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)]
    retriever = make_retriever(args.retriever)

    # Embedding ~17,000 chunks takes many minutes on a CPU, so vectors are cached per chunking setup.
    dense = getattr(retriever, "dense", retriever if args.retriever == "dense" else None)
    cache = ROOT / "data" / "eval" / "cache" / f"{args.dataset}-{args.chunk_size}-{args.chunk_overlap}.npy"
    cached = dense is not None and cache.exists()
    if cached:
        dense._preloaded = np.load(cache)

    start = time.perf_counter()
    retriever.add(chunks)
    index_seconds = time.perf_counter() - start
    if dense is not None and not cached:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, dense._vectors)

    start = time.perf_counter()
    run = {qid: unique_sources([c.source for c, _ in retriever.search(q, args.depth)])
           for qid, q in data.queries.items()}
    ms_per_query = (time.perf_counter() - start) * 1000 / len(data.queries)

    metrics = score_run(run, data.qrels)
    result = {"date": date.today().isoformat(), "dataset": f"{args.dataset}/{args.split}",
              "retriever": args.retriever, "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
              "docs": len(data.corpus), "chunks": len(chunks), "queries": len(data.queries),
              **{k: round(v, 4) for k, v in metrics.items()},
              "index_seconds": None if cached else round(index_seconds, 1),  # None: vectors came from cache
              "ms_per_query": round(ms_per_query, 2)}
    history = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(history + [result], indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
