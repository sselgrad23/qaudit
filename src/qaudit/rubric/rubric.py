"""Load the customer QA rubric.

The rubric is a small, fixed set of criteria (data, not code). The judge scores a
conversation against each one as ``pass`` / ``violation`` / ``na`` with an evidence
quote; the overall QA score is the weighted pass rate over the applicable criteria.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from pydantic import BaseModel

# The three verdicts a criterion can take. ``na`` (not applicable) is kept distinct
# from ``pass`` so "nothing to check here" never counts as evidence the agent did
# well - it is excluded from the score rather than inflating it.
VERDICTS = ("pass", "violation", "na")


class Criterion(BaseModel):
    """One rubric criterion."""

    id: str
    name: str
    weight: float = 1.0
    definition: str
    violation_when: str
    na_when: str


@lru_cache(maxsize=1)
def load_rubric() -> list[Criterion]:
    """Load and cache the packaged rubric."""
    with resources.files("qaudit.rubric").joinpath("criteria.json").open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Criterion.model_validate(c) for c in raw["criteria"]]


def rubric_ids() -> list[str]:
    """The criterion ids, in rubric order."""
    return [c.id for c in load_rubric()]
