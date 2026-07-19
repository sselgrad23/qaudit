"""Retrieval over the policy knowledge base.

Two retrievers behind one interface: a lexical BM25 baseline (no model download, the
CI path) and a dense sentence-transformers retriever. They are compared head to head
in :mod:`qaudit.eval.retrieval_eval`; picking between them is one of the project's
decisions, not an assumption.
"""

from __future__ import annotations

from qaudit.retrieval.index import RetrievalHit, Retriever, build_retriever

__all__ = ["Retriever", "RetrievalHit", "build_retriever"]
