"""Typed structured-output contract for the judge.

The judge must return, for each rubric criterion, a verdict and - crucially - a
verbatim evidence quote from the conversation. "With evidence" is the whole point of
the target problem, so a verdict without a supporting span is treated as lower
quality, and the score never depends on the free-text rationale.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from qaudit.rubric.rubric import VERDICTS, load_rubric


class CriterionVerdict(BaseModel):
    """The judge's decision on one rubric criterion."""

    criterion_id: str
    verdict: str = Field(..., description="One of pass | violation | na.")
    evidence: str = Field("", description="Verbatim quote from the conversation, or '' if na.")
    rationale: str = Field("", description="One-line justification; never scored, only shown.")

    @field_validator("verdict")
    @classmethod
    def _known_verdict(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {v!r}")
        return v


class JudgeResult(BaseModel):
    """A full rubric scoring of one conversation."""

    conversation_id: str
    backend: str
    context_mode: str = Field(..., description="'retrieved' (top-k passages) or 'full' (whole KB).")
    retrieved_passage_ids: list[str] = Field(default_factory=list)
    verdicts: list[CriterionVerdict]
    schema_valid_first_try: bool = True
    prompt_chars: int = 0
    latency_s: float = 0.0

    @property
    def verdict_map(self) -> dict[str, str]:
        """criterion_id -> verdict, for quick lookup in the eval."""
        return {v.criterion_id: v.verdict for v in self.verdicts}

    @property
    def overall_score(self) -> float:
        """Weighted pass rate over the applicable (non-na) criteria, in [0, 1].

        ``na`` criteria are excluded rather than counted as passes, so "nothing to
        check" cannot inflate a score. Returns 1.0 if every criterion is n/a.
        """
        weights = {c.id: c.weight for c in load_rubric()}
        num = den = 0.0
        for v in self.verdicts:
            if v.verdict == "na":
                continue
            w = weights.get(v.criterion_id, 1.0)
            den += w
            if v.verdict == "pass":
                num += w
        return round(num / den, 3) if den else 1.0

    @property
    def violations(self) -> list[str]:
        """Criterion ids the judge marked as violations."""
        return [v.criterion_id for v in self.verdicts if v.verdict == "violation"]
