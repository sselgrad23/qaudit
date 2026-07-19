"""Tests for the BM25 retriever and the KB (no model download)."""

from __future__ import annotations

from qaudit.kb import load_kb
from qaudit.retrieval import build_retriever


def test_bm25_returns_ranked_topk() -> None:
    r = build_retriever("bm25")
    hits = r.search("how do I change my delivery address", k=3)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_bm25_finds_a_relevant_passage() -> None:
    r = build_retriever("bm25")
    ids = {h.passage.id for h in r.search("what payment methods do you accept", k=5)}
    assert "pol-payment-methods" in ids


def test_kb_passage_ids_are_unique_and_nonempty() -> None:
    passages = load_kb()
    ids = [p.id for p in passages]
    assert len(ids) == len(set(ids))
    assert all(p.text and p.title for p in passages)
