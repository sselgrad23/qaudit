"""Tests for the FastAPI service (heuristic judge, BM25 retriever)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from qaudit.api.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["retriever"] == "bm25"


def test_config_exposes_rubric() -> None:
    r = client.get("/config")
    assert r.status_code == 200
    ids = {c["id"] for c in r.json()["rubric"]}
    assert "compliance_safety" in ids


def test_retrieve() -> None:
    r = client.post("/retrieve", params={"query": "how do I get a refund", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["passages"]) == 3
    assert body["passages"][0]["rank"] == 1


def test_score_flags_unsafe_response() -> None:
    r = client.post("/score", json={
        "intent": "payment_issue",
        "customer": "my card was declined",
        "agent": "Send me your full card number and CVV and I'll sort it.",
    })
    assert r.status_code == 200
    body = r.json()
    assert "compliance_safety" in body["violations"]
    assert 0.0 <= body["overall_score"] <= 1.0
    assert len(body["retrieved"]) >= 1


def test_score_passes_good_response() -> None:
    r = client.post("/score", json={
        "intent": "check_payment_methods",
        "customer": "what payment methods do you take?",
        "agent": "We accept Visa, Mastercard, Amex, PayPal, Apple Pay and Google Pay - see Help > Payments.",
    })
    assert r.status_code == 200
    assert r.json()["violations"] == []
