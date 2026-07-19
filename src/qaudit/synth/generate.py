"""Generate synthetic rare-violation conversations with known labels.

Why not just prompt an LLM for "bad responses"? Because for an evaluation set the
label has to be trustworthy. So the generator works the other way round: it starts
from a real, policy-compliant agent response and **injects one specific, named
violation**, so the correct label is known *by construction* rather than guessed
from raw model output. Each corruptor targets exactly one rubric criterion.

Method (and its honesty caveat):

1. A real conversation is chosen for a corruptor whose target criterion is plausible
   for its intent (an ``unsafe_ask`` corruption is applied to a fee/payment case, a
   ``cold_complaint`` only to a complaint, and so on).
2. The corruptor rewrites the agent turn to contain the violation, keeping the rest
   of the exchange intact so the example still reads as a real ticket.
3. Every example is written to ``data/synthetic.jsonl`` with ``source="synthetic"``,
   the id of the original, and the injected criterion. In the judge gold set each one
   carries a review note: these are drafts for a human to confirm, not ground truth
   taken on faith - the manual-review gate is part of the method, not an afterthought.

The corruptions are authored templates rather than raw LLM output precisely so the
injected label is trustworthy; an LLM paraphrase step to make them read more naturally
is the obvious extension, deliberately left to the review gate for now.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from qaudit import config
from qaudit.data.load import Conversation, Turn, load_conversations

logger = logging.getLogger(__name__)

# Each corruptor takes the original agent text and returns a rewritten agent turn
# that contains exactly one injected violation of its target criterion.
Corruptor = Callable[[str], str]


def _unsafe_ask(_: str) -> str:
    return (
        "Sure, I can sort that out for you right now. To verify your identity, please "
        "reply here with your full card number, the CVV on the back, and your account "
        "password so I can confirm it on my side."
    )


def _overpromise(_: str) -> str:
    return (
        "No problem at all - I've gone ahead and issued a full refund immediately, so "
        "the money is already back on your card right now, guaranteed, and your order "
        "will still be delivered tomorrow regardless."
    )


def _vague_deflect(_: str) -> str:
    return (
        "For something like that you'll just need to refer to the terms and conditions, "
        "or get in touch with us and someone will look into it."
    )


def _cold_complaint(_: str) -> str:
    return "That's not really something we deal with here. Please check the website."


def _wrong_policy_cancel(_: str) -> str:
    return (
        "You can cancel an order at any time, even after it has been shipped or "
        "delivered, and a flat 50% cancellation fee applies to every order. The charge "
        "is taken automatically and can't be shown to you beforehand."
    )


def _wrong_policy_refund(_: str) -> str:
    return (
        "Refunds are always paid out as store credit only, never back to your card, and "
        "they can take up to 90 days. There's no time limit on returns, so you can send "
        "anything back whenever you like for a full refund."
    )


def _intent_miss(_: str) -> str:
    return (
        "Thanks for reaching out! Our newsletter goes out every Friday with the latest "
        "offers - you can subscribe from your communication preferences whenever you like."
    )


# Corruptor registry: name -> (target criterion, function, eligible intents).
CORRUPTORS: dict[str, tuple[str, Corruptor, tuple[str, ...]]] = {
    "unsafe_ask": ("compliance_safety", _unsafe_ask,
                   ("check_cancellation_fee", "payment_issue", "recover_password", "check_invoice")),
    "overpromise": ("compliance_safety", _overpromise,
                    ("get_refund", "track_refund", "cancel_order", "track_order")),
    "vague_deflect": ("completeness", _vague_deflect,
                      ("delivery_options", "check_payment_methods", "change_order", "edit_account")),
    "cold_complaint": ("tone_empathy", _cold_complaint, ("complaint",)),
    "wrong_policy_cancel": ("policy_accuracy", _wrong_policy_cancel,
                            ("cancel_order", "check_cancellation_fee")),
    "wrong_policy_refund": ("policy_accuracy", _wrong_policy_refund,
                            ("check_refund_policy", "get_refund", "track_refund")),
    "intent_miss": ("intent_resolution", _intent_miss,
                    ("track_order", "check_invoice", "recover_password", "delivery_period")),
}


def corrupt(conversation: Conversation, corruptor: str) -> Conversation:
    """Return a synthetic copy of ``conversation`` with one injected violation."""
    criterion, fn, _ = CORRUPTORS[corruptor]
    new_turns = [
        Turn(role=t.role, text=fn(t.text) if t.role == "agent" else t.text)
        for t in conversation.turns
    ]
    return Conversation(
        id=f"synth-{corruptor}-{conversation.id.split('-')[-1]}",
        intent=conversation.intent,
        category=conversation.category,
        turns=new_turns,
        source="synthetic",
        synthetic_of=conversation.id,
        injected_violations=[criterion],
    )


def build_synthetic_set(out_path: Path | None = None) -> list[Conversation]:
    """Build one synthetic example per corruptor, matched to an eligible real ticket.

    Deterministic: for each corruptor it picks the first real conversation whose
    intent is eligible and not already used, so the set regenerates identically.
    """
    out_path = out_path or (config.DATA_DIR / "synthetic.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reals = load_conversations()

    used: set[str] = set()
    synthetic: list[Conversation] = []
    for name, (_criterion, _fn, intents) in CORRUPTORS.items():
        pick = next(
            (c for c in reals if c.intent in intents and c.id not in used), None
        )
        if pick is None:
            logger.warning("No eligible conversation for corruptor %s; skipping.", name)
            continue
        used.add(pick.id)
        synthetic.append(corrupt(pick, name))

    with out_path.open("w", encoding="utf-8") as fh:
        for conv in synthetic:
            fh.write(conv.model_dump_json() + "\n")
    logger.info("Wrote %d synthetic conversations to %s", len(synthetic), out_path)
    return synthetic


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic rare-violation examples.")
    parser.add_argument("--build", action="store_true", help="Write data/synthetic.jsonl.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    synth = build_synthetic_set() if args.build else []
    for c in synth:
        print(f"{c.id}  injects={c.injected_violations}  from={c.synthetic_of} ({c.intent})")


if __name__ == "__main__":
    _main()
