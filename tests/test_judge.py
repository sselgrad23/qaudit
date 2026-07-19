"""Tests for the heuristic judge and the judge orchestration (no GPU)."""

from __future__ import annotations

from qaudit.judge import build_judge, score_conversation
from qaudit.judge.judge import _extract_json, _parse_verdicts
from qaudit.rubric.rubric import load_rubric, rubric_ids


def _verdict(result, criterion_id):  # type: ignore[no-untyped-def]
    return result.verdict_map[criterion_id]


def test_heuristic_flags_unsafe_credential_request(unsafe_conversation) -> None:
    result = score_conversation(unsafe_conversation, build_judge("heuristic"))
    assert _verdict(result, "compliance_safety") == "violation"
    assert "compliance_safety" in result.violations


def test_heuristic_flags_vague_deflection(vague_conversation) -> None:
    result = score_conversation(vague_conversation, build_judge("heuristic"))
    assert _verdict(result, "completeness") == "violation"


def test_heuristic_passes_a_good_response(good_conversation) -> None:
    result = score_conversation(good_conversation, build_judge("heuristic"))
    assert _verdict(result, "compliance_safety") == "pass"
    assert _verdict(result, "completeness") == "pass"
    assert not result.violations


def test_heuristic_abstains_on_reading_criteria(good_conversation) -> None:
    """The floor is honest: it does not pretend to judge intent or policy accuracy."""
    result = score_conversation(good_conversation, build_judge("heuristic"))
    assert _verdict(result, "intent_resolution") == "na"
    assert _verdict(result, "policy_accuracy") == "na"


def test_every_criterion_is_scored(good_conversation) -> None:
    result = score_conversation(good_conversation, build_judge("heuristic"))
    assert {v.criterion_id for v in result.verdicts} == set(rubric_ids())


def test_overall_score_excludes_na(good_conversation) -> None:
    result = score_conversation(good_conversation, build_judge("heuristic"))
    # 3 applicable criteria all pass -> 1.0; the two na criteria are not counted.
    assert result.overall_score == 1.0


def test_parse_verdicts_fills_missing_with_na() -> None:
    raw = '{"verdicts": [{"criterion_id": "tone_empathy", "verdict": "pass", "evidence": ""}]}'
    verdicts, valid = _parse_verdicts(raw, load_rubric())
    assert valid is False  # not every criterion was present
    assert len(verdicts) == len(load_rubric())
    by_id = {v.criterion_id: v.verdict for v in verdicts}
    assert by_id["tone_empathy"] == "pass"
    assert by_id["policy_accuracy"] == "na"  # filled


def test_extract_json_tolerates_fences_and_prose() -> None:
    raw = 'Sure:\n```json\n{"verdicts": []}\n```\nDone.'
    assert _extract_json(raw) == {"verdicts": []}
