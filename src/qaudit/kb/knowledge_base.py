"""Load the policy knowledge base.

The KB ships as a packaged JSON so it is available whether the code runs from a
checkout, a wheel, a Docker image or a Space. Passages are authored at a natural
size (a few sentences each) and are the unit of retrieval; ``intents`` records which
Bitext intents a passage is authoritative for, which seeds - but does not replace -
the hand-checked retrieval gold set.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from pydantic import BaseModel, Field


class Passage(BaseModel):
    """One retrievable policy passage."""

    id: str
    doc: str
    title: str
    category: str
    intents: list[str] = Field(default_factory=list)
    text: str

    def for_index(self) -> str:
        """The text a retriever indexes: title + body (the doc name is metadata)."""
        return f"{self.title}. {self.text}"


@lru_cache(maxsize=1)
def load_kb() -> list[Passage]:
    """Load and cache the packaged knowledge base."""
    with resources.files("qaudit.kb").joinpath("documents.json").open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Passage.model_validate(p) for p in raw["passages"]]


def passages_by_intent(intent: str) -> list[str]:
    """Passage ids a given intent is authoritative for (seeds the retrieval gold)."""
    return [p.id for p in load_kb() if intent in p.intents]
