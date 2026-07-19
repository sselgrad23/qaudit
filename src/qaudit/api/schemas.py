"""Pydantic request/response models for the qaudit API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from qaudit.judge.schema import JudgeResult


class TurnIn(BaseModel):
    """One turn of a conversation submitted to the API."""

    role: str = Field("agent", description="'customer' or 'agent'.")
    text: str = Field(..., min_length=1)


class ScoreRequest(BaseModel):
    """Score a support conversation against the rubric.

    Provide either a list of ``turns`` or the two shorthand fields ``customer`` and
    ``agent`` for the common single-exchange case.
    """

    intent: str = Field("unknown", description="Optional intent hint; not required.")
    turns: list[TurnIn] | None = None
    customer: str | None = Field(None, description="Shorthand: the customer message.")
    agent: str | None = Field(None, description="Shorthand: the agent response.")


class RetrievedPassage(BaseModel):
    """A retrieved policy passage with its rank and score."""

    id: str
    title: str
    text: str
    rank: int
    score: float


class ScoreResponse(BaseModel):
    """The judge's rubric scoring plus the policy passages it was grounded on."""

    conversation_id: str
    overall_score: float
    violations: list[str]
    result: JudgeResult
    retrieved: list[RetrievedPassage]


class RetrieveResponse(BaseModel):
    """Top-k policy passages for a free-text query."""

    query: str
    retriever: str
    passages: list[RetrievedPassage]


class HealthResponse(BaseModel):
    """Liveness / readiness payload."""

    status: str
    version: str
    retriever: str
    judge_backend: str
    tracing: bool
