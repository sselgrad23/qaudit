"""Prompt construction for the LLM judge.

The judge is shown three things: the rubric criteria (id + what a violation looks
like), the policy context (either the retrieved top-k passages or the whole KB - the
ablation switch), and the conversation. It must return strict JSON: one object per
criterion, each with a verdict, a verbatim evidence quote, and a one-line rationale.

Grounding the verdict in the retrieved policy is deliberate: it is what lets a small
model judge *policy accuracy* rather than guess it, and it is the mechanism the
retrieval eval is ultimately protecting.
"""

from __future__ import annotations

from qaudit.data.load import Conversation
from qaudit.kb import Passage
from qaudit.rubric.rubric import Criterion

SYSTEM = (
    "You are a meticulous customer-support quality auditor for Northwind Retail. "
    "You score one agent response against a fixed rubric, using only the policy "
    "excerpts provided as your source of truth for what is correct. For every "
    "criterion you return a verdict of pass, violation, or na (not applicable), a "
    "short verbatim quote from the conversation as evidence, and a one-line reason. "
    "You never invent policy that is not in the excerpts. You output JSON only."
)


def _rubric_block(criteria: list[Criterion]) -> str:
    lines = []
    for c in criteria:
        lines.append(
            f"- {c.id} ({c.name}): {c.definition} "
            f"VIOLATION when: {c.violation_when} NA when: {c.na_when}"
        )
    return "\n".join(lines)


def _policy_block(passages: list[Passage]) -> str:
    if not passages:
        return "(no policy excerpts retrieved)"
    return "\n".join(f"- [{p.id}] {p.title}: {p.text}" for p in passages)


def build_judge_prompt(
    conversation: Conversation, criteria: list[Criterion], passages: list[Passage]
) -> str:
    """Assemble the full user-turn prompt for the judge."""
    ids = ", ".join(c.id for c in criteria)
    return (
        "POLICY EXCERPTS (your only source of truth):\n"
        f"{_policy_block(passages)}\n\n"
        "RUBRIC (score every criterion):\n"
        f"{_rubric_block(criteria)}\n\n"
        "CONVERSATION:\n"
        f"{conversation.render()}\n\n"
        "Return ONLY a compact JSON object, one entry per criterion, of this exact form:\n"
        '{"verdicts": [{"criterion_id": "<id>", "verdict": "pass|violation|na", '
        '"evidence": "<short quote, max 12 words>"}]}\n'
        f"Include exactly one entry for each criterion_id, in this order: {ids}. "
        "Keep each evidence to a few words copied from the conversation (empty for na). "
        "Base policy_accuracy strictly on the excerpts; if the response makes no checkable "
        "factual claim, mark it na. No rationale, no prose, JSON only."
    )
