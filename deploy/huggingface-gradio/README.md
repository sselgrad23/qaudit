---
title: qaudit
emoji: 🔍
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# qaudit: support-QA scoring with evidence

This Space runs the real `qaudit` pipeline. It retrieves the relevant policy for a support conversation and scores the agent's response against a quality rubric, quoting the exact evidence behind every verdict.

The demo uses BM25 retrieval and the deterministic heuristic judge, so it is instant and reliable inside a free Space with no model download and no GPU-time budget. The repository's evaluation numbers (retrieval hit/recall/MRR, judge-vs-human Cohen's kappa, and the retrieved-vs-stuffed ablation) come from the local LLM judge (Qwen2.5-3B-Instruct) run on a GPU. See the main project README.

## How this Space is built

The Space bundles the `qaudit` package under `src/` and runs `app.py` directly. Updating it means copying the current `src/qaudit/` into this directory's `src/` alongside `app.py` and `requirements.txt`, then pushing to the Space repo.

## Free-tier note

Hugging Face gates free CPU Gradio Spaces behind PRO; a free ZeroGPU Space runs this CPU-only workload fine, since the judge and retriever never touch the GPU. Nothing here can bill: no paid API, no key.
