"""Shared pytest fixtures.

The whole suite runs on the deterministic **BM25 retriever** and **heuristic judge**,
so it needs no GPU, no model download and no network - it finishes in seconds and is
safe for CI. The LLM judge and the dense retriever are exercised offline through the
eval scripts, not the unit tests.
"""

from __future__ import annotations

import os

# Force the no-GPU / no-download paths before any qaudit import reads config.
os.environ.setdefault("QAUDIT_JUDGE_BACKEND", "heuristic")
os.environ.setdefault("QAUDIT_RETRIEVER", "bm25")

import pytest  # noqa: E402

from qaudit.data.load import Conversation, Turn  # noqa: E402


@pytest.fixture()
def good_conversation() -> Conversation:
    """A well-handled cancellation: concrete steps, safe, on-intent."""
    return Conversation(
        id="t-good", intent="cancel_order", category="ORDER",
        turns=[
            Turn(role="customer", text="I want to cancel my order please"),
            Turn(role="agent", text=(
                "Of course. To cancel, sign in to Your Account, open Orders, select the "
                "order and choose Cancel order while it is still Processing. The full "
                "amount is refunded to your original payment method within 5 business days."
            )),
        ],
    )


@pytest.fixture()
def unsafe_conversation() -> Conversation:
    """An agent asking for sensitive credentials in chat (compliance violation)."""
    return Conversation(
        id="t-unsafe", intent="payment_issue", category="PAYMENT",
        turns=[
            Turn(role="customer", text="my payment failed"),
            Turn(role="agent", text=(
                "Please reply with your full card number, the CVV and your account "
                "password so I can fix it for you."
            )),
        ],
    )


@pytest.fixture()
def vague_conversation() -> Conversation:
    """A non-answer that deflects (completeness violation)."""
    return Conversation(
        id="t-vague", intent="check_cancellation_fee", category="CANCEL",
        turns=[
            Turn(role="customer", text="what is the cancellation fee?"),
            Turn(role="agent", text="Please refer to the terms and conditions or get in touch with us."),
        ],
    )
