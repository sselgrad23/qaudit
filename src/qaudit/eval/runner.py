"""Run a judge backend over the labelled set once, and cache the predictions.

The LLM judge is the slow part, and both :mod:`qaudit.eval.judge_eval` and
:mod:`qaudit.eval.ablation` need its output. This module runs a (backend,
context_mode) configuration over the labelled conversations and caches the resulting
``JudgeResult``s to ``reports/preds_<backend>_<mode>.jsonl`` so a second consumer -
or a re-run of the README numbers - doesn't pay for the model twice.
"""

from __future__ import annotations

import logging

from qaudit import config
from qaudit.data.load import Conversation, load_conversations
from qaudit.judge import JudgeResult, build_judge, score_conversation
from qaudit.judge.judge import Judge
from qaudit.retrieval import build_retriever

logger = logging.getLogger(__name__)


def labelled_conversations() -> list[Conversation]:
    """The real eval set plus the synthetic rare-violation examples, if present.

    The synthetic file is optional (built with ``qaudit.synth``); when it exists its
    known-label violations are included so the judge is measured on the hard cases,
    not just the compliant majority.
    """
    convs = load_conversations()
    synth_path = config.DATA_DIR / "synthetic.jsonl"
    if synth_path.exists():
        with synth_path.open(encoding="utf-8") as fh:
            convs += [Conversation.model_validate_json(line) for line in fh if line.strip()]
    return convs


def run_judge_over_set(
    backend: str,
    context_mode: str,
    conversations: list[Conversation] | None = None,
    *,
    use_cache: bool = True,
    retriever_name: str | None = None,
) -> list[JudgeResult]:
    """Score every conversation with one configuration, caching to reports/."""
    config.ensure_dirs()
    conversations = conversations or labelled_conversations()
    retriever_name = retriever_name or config.RETRIEVER
    cache = config.REPORT_DIR / f"preds_{backend}_{context_mode}_{retriever_name}.jsonl"

    if use_cache and cache.exists():
        with cache.open(encoding="utf-8") as fh:
            cached = [JudgeResult.model_validate_json(line) for line in fh if line.strip()]
        if {r.conversation_id for r in cached} >= {c.id for c in conversations}:
            logger.info("Using cached predictions: %s", cache.name)
            by_id = {r.conversation_id: r for r in cached}
            return [by_id[c.id] for c in conversations]

    judge: Judge = build_judge(backend)
    retriever = build_retriever(retriever_name)
    results: list[JudgeResult] = []
    for i, conv in enumerate(conversations, start=1):
        results.append(
            score_conversation(
                conv, judge, context_mode=context_mode, retriever=retriever
            )
        )
        if i % 10 == 0 or i == len(conversations):
            logger.info("Scored %d/%d (%s/%s)", i, len(conversations), backend, context_mode)

    with cache.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(r.model_dump_json() + "\n")
    return results
