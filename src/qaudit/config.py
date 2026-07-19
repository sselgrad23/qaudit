"""Central configuration for qaudit.

Every path resolves relative to the repository root so the code behaves the same
on a laptop, in a Docker container, or in CI. Anything that differs between local
development and a deployed Space is overridable with an environment variable.

The design mirrors the target problem: scoring support conversations against
*each customer's own* rubric. Nothing here is hardcoded to one customer - the
rubric and the policy knowledge base are data files (``kb/``, ``rubric/``), so a
new customer is a new rubric + KB, not new code.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# ``config.py`` lives at ``<root>/src/qaudit/config.py`` -> three parents up.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.getenv("QAUDIT_DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR: Path = Path(os.getenv("QAUDIT_MODEL_DIR", ROOT_DIR / "models"))
REPORT_DIR: Path = Path(os.getenv("QAUDIT_REPORT_DIR", ROOT_DIR / "reports"))

# The stable, committed evaluation set of conversations the gold sets reference by
# id. Rebuild/extend it from the Bitext source with ``python -m qaudit.data.load``.
CONVERSATIONS_PATH: Path = Path(
    os.getenv("QAUDIT_CONVERSATIONS", DATA_DIR / "conversations.jsonl")
)

# --- Dataset ----------------------------------------------------------------
# Real customer-support turns (customer instruction + agent response) with intent
# and category labels. CDLA-Sharing-1.0; see README "Dataset & licence".
BITEXT_DATASET: str = os.getenv(
    "QAUDIT_BITEXT_DATASET", "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)
# How many real conversations to sample into the committed eval set, and the seed
# that makes that sample reproducible. Kept small on purpose - the value is label
# quality, not row count.
EVAL_SAMPLE_SIZE: int = int(os.getenv("QAUDIT_EVAL_SAMPLE_SIZE", "40"))
SAMPLE_SEED: int = int(os.getenv("QAUDIT_SAMPLE_SEED", "20260717"))

# --- Retrieval --------------------------------------------------------------
# "bm25" (lexical, no download, the CI baseline) or "dense" (sentence-transformers).
RETRIEVER: str = os.getenv("QAUDIT_RETRIEVER", "bm25")
# all-MiniLM-L6-v2: Apache-2.0, 384-dim, ~80 MB, strong quality/size trade-off and
# free to run locally. Only loaded for the dense retriever.
EMBED_MODEL: str = os.getenv("QAUDIT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# The embedder runs on CPU by default: the KB is tiny, so encoding is instant there,
# and it keeps the GPU free for the LLM judge on an 8 GB card. Override to "cuda".
EMBED_DEVICE: str = os.getenv("QAUDIT_EMBED_DEVICE", "cpu")
TOP_K: int = int(os.getenv("QAUDIT_TOP_K", "5"))

# --- LLM judge --------------------------------------------------------------
# Backend: "transformers" (local instruct model), "heuristic" (rule-based, no LLM;
# the CI/no-GPU fallback and the eval floor), or "auto" (transformers when a CUDA
# GPU + the model are importable, else heuristic).
JUDGE_BACKEND: str = os.getenv("QAUDIT_JUDGE_BACKEND", "auto")
# Qwen2.5-3B-Instruct: Apache-2.0, strong instruction-following, runs in 4-bit next
# to nothing else on an 8 GB card. Zero marginal cost, no key that can bill.
JUDGE_MODEL: str = os.getenv("QAUDIT_JUDGE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
JUDGE_LOAD_4BIT: bool = os.getenv("QAUDIT_JUDGE_4BIT", "1") == "1"
# 320 is enough for the compact one-object-per-criterion output and keeps generation
# fast on an 8 GB card; it is the setting the reported numbers were produced with.
JUDGE_MAX_NEW_TOKENS: int = int(os.getenv("QAUDIT_JUDGE_MAX_NEW_TOKENS", "320"))
# How much rubric context the judge is shown: "retrieved" (top-k retrieved criteria)
# or "full" (the whole rubric stuffed into the prompt). The ablation compares them.
JUDGE_CONTEXT_MODE: str = os.getenv("QAUDIT_JUDGE_CONTEXT_MODE", "retrieved")

# --- LangSmith tracing ------------------------------------------------------
# Best-effort, and off unless a key is present. The free Developer plan is capped at
# 5,000 traces/month and a *personal* org is hard-capped there until a card is added
# - so, with no card on file, it cannot bill. See README "Tracing with LangSmith".
LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "qaudit")
TRACING_ENABLED: bool = bool(os.getenv("LANGCHAIN_API_KEY")) and os.getenv(
    "LANGCHAIN_TRACING_V2", "true"
) not in ("0", "false", "False")


def ensure_dirs() -> None:
    """Create the data/model/report directories if they do not exist."""
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
