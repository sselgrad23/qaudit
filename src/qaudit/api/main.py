"""FastAPI service: retrieve the relevant policy, then judge the conversation.

Endpoints:

* ``POST /score``    - score a support conversation against the rubric, returning
  per-criterion verdicts with evidence quotes, an overall score, and the policy
  passages the judge was grounded on,
* ``POST /retrieve`` - the retrieval step alone: top-k policy passages for a query,
* ``GET  /health``, ``GET /config`` - probe and operative configuration.

The retriever and judge backend are chosen by configuration, so the same service
runs the fast BM25 + heuristic judge in CI and the dense retriever + local LLM judge
on a GPU box. Retrievers and the heuristic judge are built once at startup; the LLM
judge is built lazily on first use so importing the app never loads a model.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qaudit import __version__, config
from qaudit.api.schemas import (
    HealthResponse,
    RetrievedPassage,
    RetrieveResponse,
    ScoreRequest,
    ScoreResponse,
)
from qaudit.data.load import Conversation, Turn
from qaudit.judge import build_judge, score_conversation
from qaudit.judge.judge import Judge
from qaudit.retrieval import build_retriever
from qaudit.rubric.rubric import load_rubric

logger = logging.getLogger(__name__)

app = FastAPI(
    title="qaudit API",
    version=__version__,
    summary="Retrieve a customer's policy, then judge a support conversation against its rubric.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("QAUDIT_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever = build_retriever()


@lru_cache(maxsize=1)
def _judge() -> Judge:
    """Lazily construct the judge so app import never loads a model."""
    return build_judge()


def _conversation_from_request(req: ScoreRequest) -> Conversation:
    """Build a Conversation from either the turns list or the customer/agent shorthand."""
    if req.turns:
        turns = [Turn(role=t.role, text=t.text) for t in req.turns]
    else:
        turns = []
        if req.customer:
            turns.append(Turn(role="customer", text=req.customer))
        if req.agent:
            turns.append(Turn(role="agent", text=req.agent))
    if not turns:
        turns = [Turn(role="agent", text="")]
    return Conversation(id="api", intent=req.intent, category="UNKNOWN", turns=turns)


def _passages_from_ids(ids: list[str], query: str) -> list[RetrievedPassage]:
    """Re-run retrieval to attach titles/scores to the judge's grounding passages."""
    hits = {h.passage.id: h for h in _retriever.search(query, k=max(len(ids), config.TOP_K))}
    out = []
    for rank, pid in enumerate(ids, start=1):
        hit = hits.get(pid)
        if hit is None:
            continue
        out.append(
            RetrievedPassage(
                id=hit.passage.id, title=hit.passage.title, text=hit.passage.text,
                rank=rank, score=round(hit.score, 3),
            )
        )
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe for Docker and the orchestrator."""
    return HealthResponse(
        status="ok", version=__version__, retriever=_retriever.name,
        judge_backend=config.JUDGE_BACKEND, tracing=config.TRACING_ENABLED,
    )


@app.get("/config")
def get_config() -> dict[str, Any]:
    """Expose the operative configuration and the rubric being applied."""
    return {
        "retriever": _retriever.name,
        "judge_backend": config.JUDGE_BACKEND,
        "judge_context_mode": config.JUDGE_CONTEXT_MODE,
        "top_k": config.TOP_K,
        "rubric": [{"id": c.id, "name": c.name, "weight": c.weight} for c in load_rubric()],
        "tracing_enabled": config.TRACING_ENABLED,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(query: str, k: int = config.TOP_K) -> RetrieveResponse:
    """Return the top-k policy passages for a free-text query."""
    hits = _retriever.search(query, k=k)
    return RetrieveResponse(
        query=query, retriever=_retriever.name,
        passages=[
            RetrievedPassage(
                id=h.passage.id, title=h.passage.title, text=h.passage.text,
                rank=h.rank, score=round(h.score, 3),
            )
            for h in hits
        ],
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Retrieve the relevant policy and score the conversation against the rubric."""
    conversation = _conversation_from_request(request)
    result = score_conversation(conversation, _judge(), retriever=_retriever)
    return ScoreResponse(
        conversation_id=result.conversation_id,
        overall_score=result.overall_score,
        violations=result.violations,
        result=result,
        retrieved=_passages_from_ids(result.retrieved_passage_ids, conversation.customer_text),
    )
