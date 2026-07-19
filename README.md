# qaudit

A support conversation goes in; out comes a score against a customer's quality rubric, with a verbatim quote behind every verdict. The assistant retrieves the relevant policy for the conversation and an LLM-as-judge grades each rubric criterion. But the retrieval and the judge are the easy part — the point of this project is the **evaluation harness** around them: does retrieval find the right policy, does the judge agree with human reviewers, and which context configuration should ship. A support-QA vendor's real problem isn't scoring a transcript; it's *knowing, with evidence, how well the scoring performs on every customer's unique rubric.* qaudit is built to that problem.

Live demo: ready to deploy — the Gradio Space is in [`deploy/huggingface-gradio/`](deploy/huggingface-gradio/) (push to a free HF Space; not yet hosted).

This is a small system on purpose. No React console, no Spark — the only surfaces are a REST API and the harness. What makes it a portfolio piece is measurement quality: real retrieval metrics, a judge validated against human labels with Cohen's kappa (not assumed), synthetic data to test the rare violation class, and an ablation that ends in a decision. It runs free and local, with no key that can bill.

## What it does

```
conversation --> retrieve policy (BM25 | dense) --> judge vs rubric --> per-criterion verdict + evidence
   (query)        top-k passages from the KB       LLM | heuristic      pass / violation / na + score
```

The retriever pulls the policy passages an auditor would need from the customer's knowledge base. The judge (a local Qwen2.5-3B, or a deterministic heuristic floor) scores the response against five rubric criteria — intent resolution, policy accuracy, completeness, tone, compliance — each with a verdict, a reason, and the exact words that justify it. A new customer is a new rubric + KB (`src/qaudit/rubric/`, `src/qaudit/kb/`), not new code.

## Real numbers

All from real runs; regenerate with `make eval`. Judge numbers are against **draft** human labels (see [Gold sets](#gold-sets-and-what-is-yours-to-label)).

**Retrieval** — BM25 vs dense, against the retrieval gold set (40 queries):

| Retriever | MRR | Hit@1 | Hit@3 | Hit@5 | Recall@5 |
|---|---|---|---|---|---|
| BM25 | 0.459 | 0.300 | 0.575 | 0.725 | 0.554 |
| **dense (MiniLM)** | **0.829** | **0.775** | **0.875** | **0.950** | **0.787** |

Dense nearly doubles MRR (0.46 → 0.83): the policy is written in policy language, not the customer's, so lexical search misses it. **Ship dense**; keep BM25 as the free CI baseline.

**Judge validation** — agreement with human gold (Cohen's kappa), on 47 conversations / 235 criterion cells:

| Judge | Pooled kappa | Violation F1 (P/R) | Valid JSON | Notes |
|---|---|---|---|---|
| heuristic (floor) | **0.360** | 0.706 (0.75/0.67) | 1.00 | strong on mechanical criteria (tone κ 1.0, compliance 0.85, completeness 0.79); **abstains** on intent & policy accuracy |
| LLM (Qwen2.5-3B) | 0.187 | 0.105 (0.06/0.56) | 0.64 | attempts all five, but **over-flags** (see below) |

The honest headline — and it's the whole reason the harness exists: **the local 3B judge is miscalibrated and scores *worse* than the deterministic heuristic floor** (κ 0.19 vs 0.36). Its systematic error, reported rather than smoothed away, is over-flagging: it marks compliant responses as violations on most criteria (human `pass` → judge `violation`: completeness 31/47, compliance 23, intent 21). So it catches real violations (recall 0.56) but drowns them in false positives (precision 0.06). The heuristic is the opposite — reliable where it has rules, but it *abstains* on the two criteria that need real reading (intent, policy accuracy), which is why its per-criterion κ there is 0. Neither is production-ready as-is; the eval says exactly why, and points at the fix (a stronger/calibrated judge for the reading-heavy criteria, kept honest by this same harness). These are against **draft** labels — some over-flags may be the judge being right and my draft `pass` being generous, which the hand-labelling will settle.

**Ablation** — retrieved top-k context vs. whole-KB stuffing for the judge:

| Context | Pooled kappa | Violation F1 (P/R) | Valid JSON | Mean prompt chars | Mean latency |
|---|---|---|---|---|---|
| **retrieved (top-5)** | **0.187** | 0.105 (0.06/0.56) | 0.64 | 5,143 | 24.2 s |
| full (stuffed) | 0.134 | 0.127 (0.07/0.78) | 0.91 | 13,647 | 42.3 s |

**Ship retrieved:** equal-or-better agreement at **62% smaller prompts and ~half the latency**. Full-context is the honest trade-off — it produces cleaner JSON (0.91 vs 0.64 valid) and higher violation *recall* (0.78), but lower precision and lower overall agreement, at ~2.7× the prompt cost. On a bigger KB the balance shifts toward retrieval, which is the transferable point.

## Requirements

- Python 3.10+
- A GPU is recommended for the LLM judge (an 8 GB card runs Qwen2.5-3B in 4-bit); everything also runs CPU-only on the heuristic judge + BM25 (the CI path).

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + dev tools (BM25 + heuristic judge)
pip install -e ".[dev,dense,llm,data,trace]"   # + dense retrieval, LLM judge, dataset build, tracing
```

## 2. Build the data & run the harness

```bash
make data          # sample the Bitext eval set -> data/conversations.jsonl
make synth         # build the synthetic rare-violation set -> data/synthetic.jsonl
make retrieval-eval   # hit@k / recall@k / MRR, BM25 vs dense -> reports/
make judge-eval       # judge vs human gold (Cohen's kappa) -> reports/
make ablation         # retrieved vs full-context -> decision -> reports/
```

`make eval` runs all three. On a machine without a GPU, drop `transformers` from the judge backends — the heuristic judge and both retrievers still run. Everything is also runnable directly, e.g. `python -m qaudit.eval.retrieval_eval --retrievers bm25 dense --k 1 3 5`.

## 3. Run the service

```bash
make api    # uvicorn qaudit.api.main:app --port 8000 --app-dir src   (docs at /docs)
```

```bash
# Score a conversation against the rubric:
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "customer": "what are the penalties for breaking the contract?",
  "agent": "Please refer to the terms and conditions or contact our support team."
}'

# Retrieve the relevant policy for a query:
curl -X POST "localhost:8000/retrieve?query=how%20do%20I%20get%20a%20refund&k=3"
```

The API defaults to BM25 + the heuristic judge (CPU, no download). Set `QAUDIT_RETRIEVER=dense` and `QAUDIT_JUDGE_BACKEND=transformers` on a GPU host for the full stack.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/score` | Retrieve policy + score a conversation against the rubric (verdicts + evidence) |
| `POST` | `/retrieve` | Top-k policy passages for a free-text query |
| `GET`  | `/health`, `/config` | Probe / operative configuration + the active rubric |

## Tracing with LangSmith

The judge traces to LangSmith when a key is present, and is a silent no-op otherwise:

```bash
export LANGCHAIN_API_KEY=ls__...      # free Developer tier
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=qaudit
```

**It cannot bill.** The free Developer plan is 5,000 traces/month with no credit card, and a personal org is hard-capped there until a card is added — so with no card on file, there is no billing path. Tracing failures never break scoring.

## Layout

```
src/qaudit/
  data/load.py         Bitext -> typed Conversation; intent-stratified seed-fixed sample
  kb/                  the customer's policy knowledge base (retrieval corpus)
  rubric/              the customer's QA rubric (5 criteria, data not code)
  retrieval/index.py   BM25 + dense retrievers behind one interface
  judge/               LLM + heuristic judge; typed output, validate/repair; LangSmith tracing
  synth/generate.py    synthetic rare-violation generation (known labels, review-gated)
  eval/                retrieval metrics, judge kappa, ablation, and the metric primitives
  eval/labels/         the gold sets + LABELLING.md (the written labelling rubric)
  api/                 FastAPI service (/score, /retrieve)
tests/                 pytest suite (BM25 + heuristic, no GPU/network)
deploy/huggingface-gradio/   free Gradio Space (live demo)
```

## Gold sets, and what is yours to label

The two gold sets are separate and documented, and ship **pre-filled with draft judgements** for review — not blank. The written labelling rubric (what counts as "relevant", what counts as a "violation") and step-by-step instructions are in [`src/qaudit/eval/labels/LABELLING.md`](src/qaudit/eval/labels/LABELLING.md).

- **`retrieval_gold.json`** — each conversation → the relevant policy passage ids. Task: confirm/correct the relevant set per query.
- **`judge_gold.jsonl`** — each conversation → a `pass`/`violation`/`na` verdict per criterion. Task: confirm/correct each verdict.

Every row is marked `"reviewed": false`. Priorities: the 9 draft violations first (they drive kappa most), then every `policy_accuracy` call, then the rest. Until that review is done, the judge and ablation numbers are provisional — real runs against draft labels.

## Dataset & licence

Conversations are from the **Bitext customer-support dataset** ([`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)), ~27k intent-tagged customer/agent turns, licensed **CDLA-Sharing-1.0** — which permits use and redistribution of derived data under the same licence, with attribution. The committed eval set and the query text in the retrieval gold set are used under that licence; the code is MIT (see `LICENSE`). The Northwind policy KB and rubric are authored for this project.

## Deploy

The live demo is a free Hugging Face Gradio Space (heuristic judge + BM25 — instant, no model download); source in `deploy/huggingface-gradio/`. The `Dockerfile` builds the same CPU configuration for any container platform; CI builds the image on every push.

## License

MIT (see `LICENSE`). Bundled eval data derived from Bitext (CDLA-Sharing-1.0).
