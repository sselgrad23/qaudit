"""Judge validation: does the LLM-as-judge agree with human labels?

An LLM judge is only worth anything if it agrees with the humans whose judgement it
is standing in for. This module measures that agreement against the hand-labelled
gold set - per criterion and pooled - with three lenses:

* **Cohen's kappa**: chance-corrected agreement, per criterion and overall. The
  headline "is the judge trustworthy" number.
* **Violation-detection F1**: precision/recall/F1 on the ``violation`` class, pooled.
  Catching the bad responses is the actual job, and accuracy alone hides it because
  violations are rare.
* **Systematic disagreement**: the single most common (human, judge) mismatch per
  criterion, surfaced rather than averaged away - if the judge is soft on tone or
  trigger-happy on policy, that shows up here.

Run::

    python -m qaudit.eval.judge_eval --backends heuristic transformers
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib import resources

from qaudit import config
from qaudit.eval.metrics import cohen_kappa, prf_for_class, top_disagreement
from qaudit.eval.runner import labelled_conversations, run_judge_over_set
from qaudit.judge import JudgeResult
from qaudit.rubric.rubric import rubric_ids

logger = logging.getLogger(__name__)


def load_judge_gold() -> dict[str, dict]:
    """Load the judge gold set as ``conversation_id -> {criterion_id: label}``."""
    gold: dict[str, dict] = {}
    with resources.files("qaudit.eval.labels").joinpath("judge_gold.jsonl").open(
        encoding="utf-8"
    ) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                gold[row["conversation_id"]] = row
    return gold


def _paired_labels(
    gold: dict[str, dict], preds: list[JudgeResult], criterion: str
) -> tuple[list[str], list[str]]:
    """Aligned (human, judge) label lists for one criterion over labelled convs."""
    human, judge = [], []
    by_id = {p.conversation_id: p for p in preds}
    for conv_id, row in gold.items():
        label = row.get("labels", {}).get(criterion)
        pred = by_id.get(conv_id)
        if label is None or pred is None:
            continue
        human.append(label)
        judge.append(pred.verdict_map.get(criterion, "na"))
    return human, judge


def evaluate_backend(gold: dict[str, dict], backend: str, context_mode: str) -> dict:
    """Score one judge backend against the gold set."""
    preds = run_judge_over_set(backend, context_mode, labelled_conversations())

    per_criterion = {}
    pooled_human, pooled_judge = [], []
    for cid in rubric_ids():
        human, judge = _paired_labels(gold, preds, cid)
        pooled_human += human
        pooled_judge += judge
        dis = top_disagreement(human, judge)
        per_criterion[cid] = {
            "n": len(human),
            "agreement": round(sum(h == j for h, j in zip(human, judge, strict=True)) / len(human), 3)
            if human else None,
            "kappa": cohen_kappa(human, judge),
            "top_disagreement": (
                f"human={dis[0]} -> judge={dis[1]} (x{dis[2]})" if dis else None
            ),
        }

    viol = prf_for_class(pooled_human, pooled_judge, "violation")
    valid_rate = round(sum(p.schema_valid_first_try for p in preds) / len(preds), 3)
    return {
        "backend": backend,
        "context_mode": context_mode,
        "n_labelled": len(gold),
        "pooled_kappa": cohen_kappa(pooled_human, pooled_judge),
        "pooled_agreement": round(
            sum(h == j for h, j in zip(pooled_human, pooled_judge, strict=True)) / len(pooled_human), 3
        ) if pooled_human else None,
        "violation_detection": {
            "precision": viol.precision, "recall": viol.recall, "f1": viol.f1,
            "tp": viol.tp, "fp": viol.fp, "fn": viol.fn,
        },
        "schema_valid_first_try_rate": valid_rate,
        "per_criterion": per_criterion,
    }


def run(backends: list[str], context_mode: str = "retrieved") -> dict:
    """Evaluate the given judge backends and write JSON + Markdown reports."""
    config.ensure_dirs()
    gold = load_judge_gold()
    results = [evaluate_backend(gold, b, context_mode) for b in backends]
    report = {
        "gold": "judge_gold.jsonl",
        "labelling_status": "DRAFT human labels - review before citing final numbers.",
        "context_mode": context_mode,
        "results": results,
    }
    (config.REPORT_DIR / "judge_eval.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Judge validation (agreement with human gold)",
        "",
        f"Gold: `judge_gold.jsonl` ({results[0]['n_labelled']} labelled conversations). "
        "Labels are DRAFT - see the labelling note.",
        "",
        "| Backend | Context | Pooled kappa | Agreement | Violation F1 (P/R) | Valid JSON |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        vd = r["violation_detection"]
        lines.append(
            f"| {r['backend']} | {r['context_mode']} | {r['pooled_kappa']} | "
            f"{r['pooled_agreement']} | {vd['f1']} ({vd['precision']}/{vd['recall']}) | "
            f"{r['schema_valid_first_try_rate']} |"
        )
    lines += ["", "## Per-criterion kappa and top disagreement", ""]
    for r in results:
        lines.append(f"### {r['backend']} ({r['context_mode']})")
        lines.append("")
        lines.append("| Criterion | n | Agreement | Kappa | Top disagreement |")
        lines.append("|---|---|---|---|---|")
        for cid, m in r["per_criterion"].items():
            lines.append(
                f"| {cid} | {m['n']} | {m['agreement']} | {m['kappa']} | {m['top_disagreement'] or '-'} |"
            )
        lines.append("")
    (config.REPORT_DIR / "judge_eval.md").write_text("\n".join(lines) + "\n")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["heuristic"])
    parser.add_argument("--context-mode", default="retrieved", choices=["retrieved", "full"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run(args.backends, args.context_mode)
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    _main()
