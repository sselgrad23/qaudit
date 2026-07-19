"""Gradio demo for qaudit - free Hugging Face Space entry point.

The hosted demo runs the *real* `qaudit` pipeline (bundled under ``src/``): it
retrieves the relevant Northwind policy passages for a support conversation and
scores the agent's response against the rubric, with a verbatim evidence quote per
criterion. It ships the **heuristic judge** because it is instant and reliable inside
a Space (no model download, no GPU time budget); the repository's numbers come from
the local LLM judge (Qwen2.5-3B) - see the README. Retrieval uses BM25 by default on
the Space to avoid the embedding-model download; set the env vars below to switch.

The point of the demo is to show the *evidence*: every verdict points at the exact
words in the conversation and the policy passages that justify it.
"""

from __future__ import annotations

import os
import sys

# Route writes to /tmp (a Space's app dir can be read-only) and pick the fast,
# reliable demo configuration. Must be set before qaudit.config is imported.
os.environ.setdefault("QAUDIT_DATA_DIR", "/tmp/qaudit/data")
os.environ.setdefault("QAUDIT_MODEL_DIR", "/tmp/qaudit/models")
os.environ.setdefault("QAUDIT_REPORT_DIR", "/tmp/qaudit/reports")
os.environ.setdefault("QAUDIT_RETRIEVER", "bm25")
os.environ.setdefault("QAUDIT_JUDGE_BACKEND", "heuristic")

# The qaudit package is bundled under ./src in the Space; make it importable without
# a full editable install (Spaces just run `python app.py`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr  # noqa: E402

from qaudit.data.load import Conversation, Turn  # noqa: E402
from qaudit.judge import build_judge, score_conversation  # noqa: E402
from qaudit.retrieval import build_retriever  # noqa: E402
from qaudit.rubric.rubric import load_rubric  # noqa: E402

RETRIEVER = build_retriever()
JUDGE = build_judge()
RUBRIC = {c.id: c for c in load_rubric()}

EXAMPLES = [
    ["what are the penalties for breaking the contract?",
     "To find out the penalties, please refer to the terms and conditions or contact our support team."],
    ["my payment failed, can you help?",
     "Sure - please reply here with your full card number, the CVV and your account password so I can fix it."],
    ["I want to cancel my order",
     "Of course. Sign in to Your Account, open Orders, select the order and choose Cancel while it is still Processing; the full amount is refunded to your original payment method within 5 business days."],
]

_MARK = {"pass": "✅", "violation": "❌", "na": "➖"}


def score(customer: str, agent: str):
    """Retrieve policy, judge the response, and render verdicts + evidence."""
    if not agent.strip():
        return "Enter an agent response to score.", ""
    conv = Conversation(
        id="demo", intent="unknown", category="UNKNOWN",
        turns=[Turn(role="customer", text=customer or ""), Turn(role="agent", text=agent)],
    )
    result = score_conversation(conv, JUDGE, retriever=RETRIEVER)

    lines = [f"### Overall QA score: {result.overall_score:.0%}", ""]
    for v in result.verdicts:
        crit = RUBRIC[v.criterion_id]
        lines.append(f"{_MARK[v.verdict]} **{crit.name}**: {v.verdict}")
        if v.evidence:
            lines.append(f"> {v.evidence}")
        if v.rationale:
            lines.append(f"_{v.rationale}_")
        lines.append("")

    hits = RETRIEVER.search(conv.customer_text, k=3)
    policy = ["### Policy the judge retrieved", ""]
    policy += [f"**[{h.passage.id}] {h.passage.title}**\n\n{h.passage.text}\n" for h in hits]
    return "\n".join(lines), "\n".join(policy)


with gr.Blocks(title="qaudit") as demo:
    gr.Markdown(
        "# qaudit\n"
        "Score a customer-support response against a quality rubric, with evidence. "
        "qaudit retrieves the relevant policy for the conversation, then judges each "
        "rubric criterion and quotes the exact words behind every verdict.\n\n"
        "_Hosted demo: BM25 retrieval plus the deterministic heuristic judge (instant, no "
        "model download). The repository's numbers come from a local LLM judge "
        "(Qwen2.5-3B) evaluated against a labelled set with Cohen's kappa. See the README._"
    )
    with gr.Row():
        customer = gr.Textbox(label="Customer message", lines=2)
        agent = gr.Textbox(label="Agent response (scored)", lines=4)
    btn = gr.Button("Score against the rubric", variant="primary")
    with gr.Row():
        verdicts = gr.Markdown()
        policy = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[customer, agent])
    btn.click(score, inputs=[customer, agent], outputs=[verdicts, policy])

if __name__ == "__main__":
    demo.launch()
