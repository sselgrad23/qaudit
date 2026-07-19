"""Load and sample real customer-support conversations.

The source is the Bitext customer-support corpus (CDLA-Sharing-1.0): ~27k rows, each
a customer utterance (``instruction``) plus a canonical agent ``response``, tagged
with an ``intent`` (27 of them) and a coarse ``category`` (11). We treat each row as
a minimal two-turn conversation - customer, then agent - which is the unit the judge
scores and the query the retriever runs.

Two functions matter:

* :func:`build_eval_set` samples a small, intent-stratified, seed-fixed subset and
  writes it to ``data/conversations.jsonl`` with stable ids (``conv-0001`` ...). That
  file is committed so the gold labels have something permanent to point at.
* :func:`load_conversations` reads that committed file back. Everything downstream
  (retrieval, judging, eval) goes through it, so nothing but this module needs the
  ``datasets`` dependency or the network.

Reproducibility over volume: the set is deliberately small (~40 conversations). The
value of this project is label quality, not row count.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from qaudit import config

logger = logging.getLogger(__name__)


class Turn(BaseModel):
    """One turn of a support conversation."""

    role: str = Field(..., description="'customer' or 'agent'.")
    text: str


class Conversation(BaseModel):
    """A support conversation - the unit retrieved for and scored by the judge."""

    id: str
    intent: str
    category: str
    turns: list[Turn]
    # Provenance. Real Bitext rows are ``bitext``; corrupted rare-violation examples
    # (see :mod:`qaudit.synth`) are ``synthetic`` and carry their intended labels.
    source: str = "bitext"
    synthetic_of: str | None = Field(
        default=None, description="Id of the real conversation a synthetic one was derived from."
    )
    injected_violations: list[str] = Field(
        default_factory=list, description="Criterion ids a synthetic example was built to violate."
    )

    @property
    def customer_text(self) -> str:
        """The customer's utterance - the retrieval query and the intent signal."""
        return " ".join(t.text for t in self.turns if t.role == "customer")

    @property
    def agent_text(self) -> str:
        """The agent's response - the text the rubric is scored against."""
        return " ".join(t.text for t in self.turns if t.role == "agent")

    def render(self) -> str:
        """Human/LLM-readable transcript."""
        return "\n".join(f"{t.role.capitalize()}: {t.text}" for t in self.turns)


def load_conversations(path: Path | None = None) -> list[Conversation]:
    """Read the committed evaluation set of conversations (JSONL)."""
    path = path or config.CONVERSATIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it once with `python -m qaudit.data.load --build`."
        )
    with path.open(encoding="utf-8") as fh:
        return [Conversation.model_validate_json(line) for line in fh if line.strip()]


def build_eval_set(
    size: int = config.EVAL_SAMPLE_SIZE,
    seed: int = config.SAMPLE_SEED,
    out_path: Path | None = None,
) -> list[Conversation]:
    """Sample an intent-stratified subset of Bitext and write it as stable JSONL.

    Sampling is round-robin across intents (so all 27 appear before any repeats) and
    seed-shuffled within each intent, which keeps the set balanced rather than
    mirroring Bitext's own skew - the imbalance is handled deliberately in the gold
    sets and the synthetic data, not smuggled in through the sample.
    """
    from datasets import load_dataset  # local import: only `--build` needs `datasets`

    out_path = out_path or config.CONVERSATIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config.BITEXT_DATASET, split="train")
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, intent in enumerate(ds["intent"]):
        by_intent[intent].append(i)

    import random

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptographic
    for rows in by_intent.values():
        rng.shuffle(rows)

    # Round-robin across intents until we have `size` rows.
    intents = sorted(by_intent)
    picked: list[int] = []
    cursors = dict.fromkeys(intents, 0)
    while len(picked) < size and any(cursors[k] < len(by_intent[k]) for k in intents):
        for intent in intents:
            if len(picked) >= size:
                break
            rows = by_intent[intent]
            if cursors[intent] < len(rows):
                picked.append(rows[cursors[intent]])
                cursors[intent] += 1

    conversations: list[Conversation] = []
    for n, row_idx in enumerate(sorted(picked), start=1):
        row = ds[row_idx]  # type: ignore
        conversations.append(
            Conversation(
                id=f"conv-{n:04d}",
                intent=row["intent"], # type: ignore
                category=row["category"], # type: ignore
                turns=[
                    Turn(role="customer", text=row["instruction"].strip()), # type: ignore
                    Turn(role="agent", text=row["response"].strip()), # type: ignore
                ],
            )
        )

    with out_path.open("w", encoding="utf-8") as fh:
        for conv in conversations:
            fh.write(conv.model_dump_json() + "\n")
    logger.info("Wrote %d conversations to %s", len(conversations), out_path)
    return conversations


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build/inspect the conversation eval set.")
    parser.add_argument("--build", action="store_true", help="Rebuild data/conversations.jsonl from Bitext.")
    parser.add_argument("--size", type=int, default=config.EVAL_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=config.SAMPLE_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.build:
        convs = build_eval_set(size=args.size, seed=args.seed)
    else:
        convs = load_conversations()
    intents = sorted({c.intent for c in convs})
    print(f"{len(convs)} conversations across {len(intents)} intents")
    print("intents:", ", ".join(intents))


if __name__ == "__main__":
    _main()
