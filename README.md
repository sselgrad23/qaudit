# qaudit

qaudit scores customer-support conversations against a per-customer quality rubric. A conversation goes in, the relevant policy is retrieved from that customer's knowledge base, and an LLM-as-judge grades each rubric criterion, quoting the text that supports each verdict.

The focus of the project is the evaluation harness rather than the scorer: measuring whether retrieval surfaces the right policy, whether the judge agrees with human labels, and which context configuration to ship. Running a model over a transcript is easy. Knowing how well it scores against a given rubric, with evidence, is the harder problem and the one that matters to a support-QA product.

The surface area is small: a REST API and the evaluation harness, no UI layer. Everything runs locally with no paid APIs and nothing that can incur charges.

Live demo: the Gradio Space in [`deploy/huggingface-gradio/`](deploy/huggingface-gradio/) is ready to deploy but not currently hosted.

## What it does

```
conversation --> retrieve policy (BM25 | dense) --> judge vs rubric --> per-criterion verdict + evidence
   (query)        top-k passages from the KB       LLM | heuristic      pass / violation / na + score
```

The retriever pulls the policy passages an auditor would need out of the customer's knowledge base. The judge (a local Qwen2.5-3B, or a deterministic heuristic as the floor) scores the response against five criteria: intent resolution, policy accuracy, completeness, tone, and compliance. Each verdict carries a short reason and the exact words that justify it. A new customer means a new rubric and KB (`src/qaudit/rubric/`, `src/qaudit/kb/`), not new code.

## Results

All figures regenerate with `make eval`. The judge numbers are measured against draft labels (see [Gold sets](#gold-sets)).

**Retrieval**, BM25 vs dense, against the retrieval gold set (40 queries):

| Retriever | MRR | Hit@1 | Hit@3 | Hit@5 | Recall@5 |
|---|---|---|---|---|---|
| BM25 | 0.459 | 0.300 | 0.575 | 0.725 | 0.554 |
| **dense (MiniLM)** | **0.829** | **0.775** | **0.875** | **0.950** | **0.787** |

Dense roughly doubles MRR (0.46 to 0.83). The policy is written in policy language rather than the customer's wording, so keyword search misses it. Dense is the default; BM25 remains as the free, download-free baseline used in CI.

**Judge validation**, agreement with human gold (Cohen's kappa), on 47 conversations and 235 criterion cells:

| Judge | Pooled kappa | Violation F1 (P/R) | Valid JSON | Notes |
|---|---|---|---|---|
| heuristic (floor) | **0.360** | 0.706 (0.75/0.67) | 1.00 | strong on mechanical criteria (tone κ 1.0, compliance 0.85, completeness 0.79); abstains on intent and policy accuracy |
| LLM (Qwen2.5-3B) | 0.187 | 0.105 (0.06/0.56) | 0.64 | attempts all five, but over-flags (see below) |

The main result, and the reason the harness exists, is that the local 3B judge is miscalibrated and scores below the heuristic floor (κ 0.19 vs 0.36). Its systematic error is over-flagging: it marks compliant responses as violations on most criteria (human `pass`, judge `violation`: completeness 31 of 47, compliance 23, intent 21), so it catches real violations (recall 0.56) but buries them in false positives (precision 0.06). The heuristic is the opposite. It is reliable where it has explicit rules but abstains on the two criteria that need real reading, intent and policy accuracy, giving kappa 0 there. Neither is production-ready; the eval shows why, and points at the fix, a stronger or calibrated judge for the reading-heavy criteria. The numbers run against draft labels, so some over-flags may prove correct where the draft `pass` labels were lenient.

**Ablation**, retrieved top-k context vs whole-KB stuffing for the judge:

| Context | Pooled kappa | Violation F1 (P/R) | Valid JSON | Mean prompt chars | Mean latency |
|---|---|---|---|---|---|
| **retrieved (top-5)** | **0.187** | 0.105 (0.06/0.56) | 0.64 | 5,143 | 24.2 s |
| full (stuffed) | 0.134 | 0.127 (0.07/0.78) | 0.91 | 13,647 | 42.3 s |

Retrieved (top-k) is the shipped configuration: agreement is as good or better than full-context stuffing at 62% smaller prompts and roughly half the latency. Full context is a genuine trade-off, with cleaner JSON (0.91 vs 0.64 valid) and higher violation recall (0.78) but lower precision and lower overall agreement at about 2.7x the prompt size. The balance shifts back toward retrieval as the knowledge base grows.

## Requirements

- Python 3.10+
- A GPU helps for the LLM judge (an 8 GB card runs Qwen2.5-3B in 4-bit). Everything also runs CPU-only on the heuristic judge plus BM25, which is the CI path.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + dev tools (BM25 + heuristic judge)
pip install -e ".[dev,dense,llm,data,trace]"   # + dense retrieval, LLM judge, dataset build, tracing
```

## 2. Build the data and run the harness

```bash
make data          # sample the Bitext eval set -> data/conversations.jsonl
make synth         # build the synthetic rare-violation set -> data/synthetic.jsonl
make retrieval-eval   # hit@k / recall@k / MRR, BM25 vs dense -> reports/
make judge-eval       # judge vs human gold (Cohen's kappa) -> reports/
make ablation         # retrieved vs full-context -> decision -> reports/
```

`make eval` runs all three. Without a GPU, drop `transformers` from the judge backends; the heuristic judge and both retrievers still run. Each step can also be run directly, for example `python -m qaudit.eval.retrieval_eval --retrievers bm25 dense --k 1 3 5`.

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

The API defaults to BM25 plus the heuristic judge (CPU, no download). Setting `QAUDIT_RETRIEVER=dense` and `QAUDIT_JUDGE_BACKEND=transformers` on a GPU host runs the full stack.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/score` | Retrieve policy and score a conversation against the rubric (verdicts + evidence) |
| `POST` | `/retrieve` | Top-k policy passages for a free-text query |
| `GET`  | `/health`, `/config` | Probe / operative configuration + the active rubric |

## Tracing with LangSmith

The judge traces to LangSmith when a key is present, and is a silent no-op otherwise:

```bash
export LANGCHAIN_API_KEY=ls__...      # free Developer tier
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=qaudit
```

There is no billing path. The free Developer plan gives 5,000 traces/month with no credit card, and a personal org is hard-capped there until a card is added. Tracing failures never break scoring.

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

## Gold sets

Two separate gold sets back the evaluation. `retrieval_gold.json` maps each conversation to the relevant policy passage ids, and `judge_gold.jsonl` holds a `pass`/`violation`/`na` verdict per criterion for each conversation. Both currently carry draft labels (`"reviewed": false`), so the judge and ablation numbers are provisional against those labels. The labelling protocol and the definitions of "relevant" and "violation" are documented in [`src/qaudit/eval/labels/LABELLING.md`](src/qaudit/eval/labels/LABELLING.md).

## Dataset and licence

Conversations come from the Bitext customer-support dataset ([`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)), about 27k intent-tagged customer/agent turns, licensed CDLA-Sharing-1.0. That licence allows use and redistribution of derived data under the same licence, with attribution. The committed eval set and the query text in the retrieval gold set fall under that licence. The code is MIT (see `LICENSE`). The Northwind policy KB and rubric are written for this project.

## Deploy

The demo is a free Hugging Face Gradio Space (heuristic judge plus BM25, so it is instant with no model download); source in `deploy/huggingface-gradio/`. The `Dockerfile` builds the same CPU setup for any container platform, and CI builds the image on every push.

## License

MIT (see `LICENSE`). Bundled eval data derived from Bitext (CDLA-Sharing-1.0).
