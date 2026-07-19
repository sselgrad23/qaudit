"""Ablation: retrieved top-k context vs. whole-KB stuffing - and a ship decision.

The question this answers is real and recurring: should the judge see only the
top-k *retrieved* policy passages, or the *entire* knowledge base stuffed into the
prompt? Stuffing is simpler and can't miss a relevant passage; retrieval is cheaper
and keeps the judge focused. We run the same LLM judge both ways over the labelled
set and compare on what matters:

* judge quality - pooled Cohen's kappa and violation-detection F1 against human gold,
* cost - mean prompt size (a proxy for tokens, and for latency and money at scale).

The module prints a recommendation with the numbers behind it. On a small KB the
two often tie on quality, at which point the decision is made on cost - which is
exactly the kind of evidence-backed call the target role is asking for.

Run::

    python -m qaudit.eval.ablation --modes retrieved full
"""

from __future__ import annotations

import argparse
import json
import logging

from qaudit import config
from qaudit.eval.judge_eval import evaluate_backend, load_judge_gold
from qaudit.eval.runner import labelled_conversations, run_judge_over_set

logger = logging.getLogger(__name__)


def _cost(backend: str, mode: str) -> dict:
    """Mean prompt size and latency for a configuration (from cached predictions)."""
    preds = run_judge_over_set(backend, mode, labelled_conversations())
    n = len(preds)
    return {
        "mean_prompt_chars": round(sum(p.prompt_chars for p in preds) / n, 1) if n else 0.0,
        "mean_latency_s": round(sum(p.latency_s for p in preds) / n, 3) if n else 0.0,
    }


def _decide(rows: list[dict]) -> str:
    """Pick a config: best kappa, breaking near-ties (<=0.02) on prompt size."""
    ranked = sorted(
        rows,
        key=lambda r: (-(r["pooled_kappa"] or -1), r["mean_prompt_chars"]),
    )
    best = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        k1 = best["pooled_kappa"] or 0.0
        k2 = second["pooled_kappa"] or 0.0
        if abs(k1 - k2) <= 0.02 and best["mean_prompt_chars"] > second["mean_prompt_chars"]:
            best = second  # a tie on quality is decided by cost
    saving = ""
    other = next((r for r in rows if r["context_mode"] != best["context_mode"]), None)
    if other and other["mean_prompt_chars"]:
        pct = 100 * (1 - best["mean_prompt_chars"] / other["mean_prompt_chars"])
        if pct > 0:
            saving = f" at {pct:.0f}% smaller prompts than '{other['context_mode']}'"
    return (
        f"Ship context_mode='{best['context_mode']}': pooled kappa {best['pooled_kappa']}, "
        f"violation-F1 {best['violation_f1']}{saving}."
    )


def run(backend: str = "transformers", modes: list[str] | None = None) -> dict:
    """Run the ablation and write JSON + Markdown reports."""
    config.ensure_dirs()
    modes = modes or ["retrieved", "full"]
    gold = load_judge_gold()

    rows = []
    for mode in modes:
        judged = evaluate_backend(gold, backend, mode)
        cost = _cost(backend, mode)
        rows.append(
            {
                "context_mode": mode,
                "pooled_kappa": judged["pooled_kappa"],
                "pooled_agreement": judged["pooled_agreement"],
                "violation_f1": judged["violation_detection"]["f1"],
                "violation_precision": judged["violation_detection"]["precision"],
                "violation_recall": judged["violation_detection"]["recall"],
                **cost,
            }
        )

    decision = _decide(rows)
    report = {"backend": backend, "modes": modes, "rows": rows, "decision": decision}
    (config.REPORT_DIR / "ablation.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Ablation: retrieved top-k context vs. full-KB stuffing",
        "",
        f"Judge backend: `{backend}`. Same labelled set, same rubric; only the policy "
        "context the judge sees changes.",
        "",
        "| Context mode | Pooled kappa | Violation F1 (P/R) | Mean prompt chars | Mean latency (s) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['context_mode']} | {r['pooled_kappa']} | "
            f"{r['violation_f1']} ({r['violation_precision']}/{r['violation_recall']}) | "
            f"{r['mean_prompt_chars']} | {r['mean_latency_s']} |"
        )
    lines += ["", f"**Decision.** {decision}", ""]
    (config.REPORT_DIR / "ablation.md").write_text("\n".join(lines) + "\n")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--modes", nargs="+", default=["retrieved", "full"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run(args.backend, args.modes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
