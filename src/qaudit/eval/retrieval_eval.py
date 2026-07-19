"""Retrieval evaluation: BM25 vs dense against the retrieval gold set.

For every gold query (a real support conversation) we retrieve the top passages and
score them against the hand-checked set of relevant policy passages, reporting hit
rate@k, recall@k and MRR at several k. Running both retrievers side by side is what
turns "which retriever?" from an assumption into a measured decision - and on this KB
the gap is large, because the policy is written in policy language, not the customer's.

Run::

    python -m qaudit.eval.retrieval_eval --retrievers bm25 dense --k 1 3 5
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib import resources

from qaudit import config
from qaudit.eval.metrics import hit_rate_at_k, recall_at_k, reciprocal_rank
from qaudit.retrieval import build_retriever

logger = logging.getLogger(__name__)


def load_retrieval_gold() -> dict:
    """Load the packaged retrieval gold set."""
    with resources.files("qaudit.eval.labels").joinpath("retrieval_gold.json").open(
        encoding="utf-8"
    ) as fh:
        return json.load(fh)


def evaluate_retriever(gold: dict, retriever_name: str, ks: list[int]) -> dict:
    """Score one retriever over every gold query."""
    retriever = build_retriever(retriever_name)
    max_k = max(ks)
    queries = [q for q in gold["queries"] if q["relevant_passages"]]

    per_k = {k: {"hit": 0.0, "recall": 0.0} for k in ks}
    mrr = 0.0
    for q in queries:
        relevant = set(q["relevant_passages"])
        ranked = [h.passage.id for h in retriever.search(q["query"], k=max_k)]
        for k in ks:
            per_k[k]["hit"] += hit_rate_at_k(relevant, ranked, k)
            per_k[k]["recall"] += recall_at_k(relevant, ranked, k)
        mrr += reciprocal_rank(relevant, ranked)

    n = len(queries)
    return {
        "retriever": retriever_name,
        "n_queries": n,
        "mrr": round(mrr / n, 3) if n else 0.0,
        "at_k": {
            str(k): {
                "hit_rate": round(per_k[k]["hit"] / n, 3) if n else 0.0,
                "recall": round(per_k[k]["recall"] / n, 3) if n else 0.0,
            }
            for k in ks
        },
    }


def run(retrievers: list[str], ks: list[int]) -> dict:
    """Evaluate the retrievers and write JSON + Markdown reports."""
    config.ensure_dirs()
    gold = load_retrieval_gold()
    results = [evaluate_retriever(gold, r, ks) for r in retrievers]
    report = {
        "gold": "retrieval_gold.json",
        "labelling_status": gold.get("labelling_status", "unknown"),
        "n_queries": results[0]["n_queries"] if results else 0,
        "ks": ks,
        "results": results,
    }
    (config.REPORT_DIR / "retrieval_eval.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Retrieval evaluation (BM25 vs dense)",
        "",
        f"Gold: `retrieval_gold.json` ({report['n_queries']} queries with >=1 relevant passage). "
        f"Status: {report['labelling_status']}",
        "",
        "| Retriever | MRR | " + " | ".join(f"Hit@{k}" for k in ks) + " | "
        + " | ".join(f"Recall@{k}" for k in ks) + " |",
        "|---|---|" + "---|" * (2 * len(ks)),
    ]
    for r in results:
        hits = " | ".join(str(r["at_k"][str(k)]["hit_rate"]) for k in ks)
        recs = " | ".join(str(r["at_k"][str(k)]["recall"]) for k in ks)
        lines.append(f"| {r['retriever']} | {r['mrr']} | {hits} | {recs} |")
    (config.REPORT_DIR / "retrieval_eval.md").write_text("\n".join(lines) + "\n")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "dense"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5], dest="ks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run(args.retrievers, args.ks)
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    _main()
