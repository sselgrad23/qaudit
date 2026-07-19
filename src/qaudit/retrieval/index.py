"""BM25 and dense retrievers over the policy knowledge base.

Both implement the same tiny interface - :meth:`Retriever.search` returns the
top-``k`` passages for a query text as ``RetrievalHit``s - so the eval harness, the
judge and the API are agnostic to which one is in use. That symmetry is the point:
the retriever is a swappable component whose choice is settled by measurement.

* **BM25** (``rank_bm25``) is lexical, needs no model download, and is the default /
  CI baseline. It rewards word overlap, which is exactly where it struggles on this
  KB - the policy is written in policy language, not the customer's words.
* **Dense** (``sentence-transformers``, all-MiniLM-L6-v2) embeds query and passages
  into one vector space and ranks by cosine similarity, so it can match "penalty for
  breaking the contract" to "early-termination charge" without shared words. It is an
  optional extra (the ``dense`` install) and loads a ~80 MB model on first use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qaudit import config
from qaudit.kb import Passage, load_kb

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens - the shared, dependency-free tokeniser for BM25."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievalHit:
    """One retrieved passage with its rank (1-based) and raw score."""

    passage: Passage
    rank: int
    score: float


class Retriever(Protocol):
    """Common interface: return the top-``k`` passages for a query."""

    name: str

    def search(self, query: str, k: int = config.TOP_K) -> list[RetrievalHit]: ...


class BM25Retriever:
    """Lexical Okapi-BM25 retriever over the KB passages."""

    name = "bm25"

    def __init__(self, passages: list[Passage] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self.passages = passages or load_kb()
        self._bm25 = BM25Okapi([_tokenize(p.for_index()) for p in self.passages])

    def search(self, query: str, k: int = config.TOP_K) -> list[RetrievalHit]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [
            RetrievalHit(passage=self.passages[i], rank=rank, score=float(scores[i]))
            for rank, i in enumerate(order, start=1)
        ]


class DenseRetriever:
    """Dense retriever: cosine similarity over sentence-transformer embeddings.

    The KB is small (tens of passages), so a plain normalised matrix multiply is the
    whole index - no faiss, no server, one dependency. Embeddings are computed once at
    construction and reused for every query.
    """

    name = "dense"

    def __init__(self, passages: list[Passage] | None = None, model: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.passages = passages or load_kb()
        self._model = SentenceTransformer(model or config.EMBED_MODEL, device=config.EMBED_DEVICE)
        self._matrix = self._embed([p.for_index() for p in self.passages])

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def search(self, query: str, k: int = config.TOP_K) -> list[RetrievalHit]:
        q = self._embed([query])[0]
        scores = self._matrix @ q  # cosine similarity, vectors are unit-normalised
        order = np.argsort(scores)[::-1][:k]
        return [
            RetrievalHit(passage=self.passages[i], rank=rank, score=float(scores[i]))
            for rank, i in enumerate(order, start=1)
        ]


def build_retriever(name: str | None = None, passages: list[Passage] | None = None) -> Retriever:
    """Construct a retriever by name ('bm25' or 'dense')."""
    name = (name or config.RETRIEVER).lower()
    if name == "bm25":
        return BM25Retriever(passages)
    if name == "dense":
        return DenseRetriever(passages)
    raise ValueError(f"Unknown retriever {name!r} (expected 'bm25' or 'dense').")
