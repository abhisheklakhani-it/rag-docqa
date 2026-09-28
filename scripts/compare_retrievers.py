"""Results table + significance tests between retrievers (paired randomization test on per-question nDCG@10).

  python scripts/compare_retrievers.py        # after evaluate_retrieval.py has run for each retriever
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragqa.evaluation import paired_randomization_test  # noqa: E402

ORDER = ["tfidf", "bm25", "dense", "hybrid"]
PAIRS = [("bm25", "tfidf"), ("dense", "bm25"), ("hybrid", "tfidf"), ("hybrid", "bm25"), ("hybrid", "dense")]


def main() -> None:
    runs = {r["retriever"]: r for r in json.loads((ROOT / "results" / "retrieval_eval.json").read_text())
            if r.get("min_dense_score", 0.0) == 0.0}  # latest run per retriever, without a score floor
    lines = ["| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms / question |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for name in ORDER:
        r = runs[name]
        lines.append(f"| {name} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
                     f"{r['mrr@10']:.3f} | {r['ndcg@10']:.3f} | {r['ms_per_query']:.2f} |")
    lines += ["", "| A | B | nDCG@10 A − B | p-value | Significant (p < 0.05)? |", "|:--|:--|--:|--:|:--|"]
    per_query = {n: json.loads((ROOT / "results" / "per_query" / f"{n}.json").read_text()) for n in ORDER}
    for a, b in PAIRS:
        qids = list(per_query[a])
        xa, xb = [per_query[a][q] for q in qids], [per_query[b][q] for q in qids]
        p = paired_randomization_test(xa, xb)
        diff = (sum(xa) - sum(xb)) / len(qids)
        lines.append(f"| {a} | {b} | {diff:+.3f} | {'< 0.001' if p < 0.001 else f'{p:.3f}'} | {'yes' if p < 0.05 else 'no'} |")
    table = "\n".join(lines) + "\n"
    (ROOT / "results" / "retrieval_eval.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
