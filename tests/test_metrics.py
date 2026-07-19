"""Tests for the metric primitives - the numbers must not drift silently."""

from __future__ import annotations

from qaudit.eval.metrics import (
    cohen_kappa,
    hit_rate_at_k,
    prf_for_class,
    recall_at_k,
    reciprocal_rank,
    top_disagreement,
)


def test_hit_rate_at_k() -> None:
    rel = {"a", "b"}
    assert hit_rate_at_k(rel, ["x", "a", "y"], k=2) == 1.0
    assert hit_rate_at_k(rel, ["x", "a", "y"], k=1) == 0.0


def test_recall_at_k() -> None:
    rel = {"a", "b"}
    assert recall_at_k(rel, ["a", "x", "b"], k=3) == 1.0
    assert recall_at_k(rel, ["a", "x", "y"], k=3) == 0.5
    assert recall_at_k(set(), ["a"], k=1) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank({"b"}, ["a", "b", "c"]) == 0.5
    assert reciprocal_rank({"z"}, ["a", "b"]) == 0.0


def test_cohen_kappa_perfect_and_undefined() -> None:
    assert cohen_kappa(["pass", "violation", "pass"], ["pass", "violation", "pass"]) == 1.0
    # Only one label value present across both raters -> undefined.
    assert cohen_kappa(["pass", "pass"], ["pass", "pass"]) is None
    # Too few items -> undefined.
    assert cohen_kappa(["pass"], ["pass"]) is None


def test_prf_for_violation_class() -> None:
    human = ["violation", "pass", "violation", "pass"]
    judge = ["violation", "pass", "pass", "violation"]
    prf = prf_for_class(human, judge, "violation")
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 1
    assert prf.precision == 0.5 and prf.recall == 0.5 and prf.f1 == 0.5


def test_top_disagreement() -> None:
    human = ["pass", "pass", "violation"]
    judge = ["na", "na", "violation"]
    assert top_disagreement(human, judge) == ("pass", "na", 2)
    assert top_disagreement(["pass"], ["pass"]) is None
