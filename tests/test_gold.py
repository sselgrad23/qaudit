"""Integrity tests for the gold sets - they must stay consistent with the KB, the
rubric and the conversation set as any of those evolve."""

from __future__ import annotations

from qaudit.eval.judge_eval import load_judge_gold
from qaudit.eval.retrieval_eval import load_retrieval_gold
from qaudit.eval.runner import labelled_conversations
from qaudit.kb import load_kb
from qaudit.rubric.rubric import VERDICTS, rubric_ids


def test_retrieval_gold_passage_ids_exist() -> None:
    kb_ids = {p.id for p in load_kb()}
    gold = load_retrieval_gold()
    for q in gold["queries"]:
        for pid in q["relevant_passages"]:
            assert pid in kb_ids, f"{q['conversation_id']} references unknown passage {pid}"


def test_retrieval_gold_covers_every_conversation() -> None:
    conv_ids = {c.id for c in labelled_conversations() if c.source == "bitext"}
    gold_ids = {q["conversation_id"] for q in load_retrieval_gold()["queries"]}
    assert conv_ids == gold_ids


def test_judge_gold_labels_are_valid() -> None:
    criteria = set(rubric_ids())
    gold = load_judge_gold()
    for conv_id, row in gold.items():
        assert set(row["labels"]) == criteria, f"{conv_id} missing/extra criteria"
        for verdict in row["labels"].values():
            assert verdict in VERDICTS, f"{conv_id} has bad verdict {verdict}"


def test_judge_gold_covers_every_labelled_conversation() -> None:
    conv_ids = {c.id for c in labelled_conversations()}
    assert conv_ids == set(load_judge_gold())


def test_judge_gold_has_a_violation_class() -> None:
    """A validation set with no violations makes kappa meaningless - guard it."""
    gold = load_judge_gold()
    violations = [
        (cid, k) for cid, row in gold.items() for k, v in row["labels"].items() if v == "violation"
    ]
    assert len(violations) >= 5
