"""Measure retrieval quality on a labelled question set (BEIR format).

Each corpus document is ingested like an uploaded file (named after its id), chunked with the app's
own chunker, and retrieved with the app's own retriever. A retrieved chunk counts as a hit when it
comes from a document that humans labelled as relevant for the question.
"""

import csv
import io
import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class LabelledSet:
    corpus: dict[str, str]              # doc id -> text
    queries: dict[str, str]             # query id -> question
    qrels: dict[str, dict[str, int]]    # query id -> {relevant doc id: grade}


def load_beir(name: str, split: str, data_dir: Path) -> LabelledSet:
    """Download a BEIR dataset once, then load one split (only queries with a relevant document)."""
    folder = data_dir / name
    if not (folder / "corpus.jsonl").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(data_dir)
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    corpus = {d["_id"]: f"{d.get('title', '').strip()}\n\n{d['text']}".strip() for d in docs}
    qrels: dict[str, dict[str, int]] = {}
    with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        queries = {q["_id"]: q["text"] for q in map(json.loads, f) if q["_id"] in qrels}
    return LabelledSet(corpus, queries, qrels)


def unique_sources(ranked_sources: list[str]) -> list[str]:
    """Several chunks of one document can be retrieved; keep each document once, at its best rank."""
    return list(dict.fromkeys(ranked_sources))


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & relevant.keys()) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    return next((1.0 / rank for rank, d in enumerate(ranked[:k], 1) if d in relevant), 0.0)


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def score_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Average recall@1/5/10, MRR@10 and nDCG@10 over all labelled questions."""
    totals = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        for k in (1, 5, 10):
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        totals["mrr@10"] += reciprocal_rank(ranked, relevant)
        totals["ndcg@10"] += ndcg_at_k(ranked, relevant)
    return {name: value / len(qrels) for name, value in totals.items()}
