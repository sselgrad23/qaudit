"""The LLM-as-judge: score a conversation against the rubric, with evidence.

Two backends behind one call. :class:`~qaudit.judge.judge.HeuristicJudge` is
deterministic, GPU-free, and the CI floor; :class:`~qaudit.judge.judge.LLMJudge`
runs a local instruct model. Both return the same typed
:class:`~qaudit.judge.schema.JudgeResult`. Whether the judge should be trusted is
not assumed here - it is measured against human labels in
:mod:`qaudit.eval.judge_eval`.
"""

from __future__ import annotations

from qaudit.judge.judge import HeuristicJudge, build_judge, score_conversation
from qaudit.judge.schema import CriterionVerdict, JudgeResult

__all__ = [
    "CriterionVerdict",
    "JudgeResult",
    "HeuristicJudge",
    "build_judge",
    "score_conversation",
]
