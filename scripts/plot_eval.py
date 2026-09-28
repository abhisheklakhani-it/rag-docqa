"""Draw the README chart: nDCG@10 per retriever with 95% bootstrap confidence intervals (light and dark).

  python scripts/plot_eval.py     # after evaluate_retrieval.py has run for each retriever
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"tfidf": "TF-IDF (before)", "bm25": "BM25", "dense": "Dense (BGE-small)", "hybrid": "Hybrid: BM25 + dense (RRF)"}
FAMILY = {"tfidf": "lexical", "bm25": "lexical", "dense": "dense", "hybrid": "hybrid"}
MARKER = {"lexical": "o", "dense": "s", "hybrid": "D"}
# Colour-blind-safe categorical palette (same as retrieval-eval-bench), plus a marker shape per family.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "lexical": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#383835",
             "lexical": "#3987e5", "dense": "#d95926", "hybrid": "#199e70"},
}


def bootstrap_ci(scores: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(scores)
    means = x[np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot(theme: str) -> None:
    t = THEMES[theme]
    rows = []
    for name in ["tfidf", "bm25", "dense", "hybrid"]:
        scores = list(json.loads((ROOT / "results" / "per_query" / f"{name}.json").read_text()).values())
        rows.append((name, float(np.mean(scores)), *bootstrap_ci(scores)))

    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=200, facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for y, (name, mean, low, high) in enumerate(rows):
        f = FAMILY[name]
        ax.plot([low, high], [y, y], color=t[f], linewidth=2, solid_capstyle="round", alpha=0.55)
        ax.plot(mean, y, marker=MARKER[f], markersize=9, color=t[f], markeredgecolor=t["surface"], markeredgewidth=2)
        ax.text(high + 0.006, y, f"{mean:.3f}", va="center", fontsize=10, color=t["text"])
    ax.set_yticks(range(len(rows)), [LABELS[r[0]] for r in rows], color=t["text"], fontsize=10)
    ax.set_xlim(0.5, 0.82)
    ax.set_xlabel("nDCG@10 (higher is better) · lines show the 95% bootstrap confidence interval",
                  color=t["muted"], fontsize=9.5)
    ax.set_title("Retrieval quality in rag-docqa (SciFact, 300 questions)", loc="left",
                 color=t["text"], fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / f"retrieval_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "docs").mkdir(exist_ok=True)
    for theme in THEMES:
        plot(theme)
    print("Saved docs/retrieval_light.png and docs/retrieval_dark.png")
