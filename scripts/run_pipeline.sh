#!/usr/bin/env bash
# End-to-end: build the data, then run the three evaluations that are the point of
# the project. Free and local. On a machine without a GPU, drop 'transformers' from
# the judge backends - the heuristic judge and BM25/dense retrieval still run.
set -euo pipefail
cd "$(dirname "$0")/.."

RETRIEVER="${QAUDIT_RETRIEVER:-dense}"

echo "== 1. Build the conversation eval set (from Bitext) =="
python -m qaudit.data.load --build

echo "== 2. Build the synthetic rare-violation set =="
python -m qaudit.synth.generate --build

echo "== 3. Retrieval eval (BM25 vs dense) =="
python -m qaudit.eval.retrieval_eval --retrievers bm25 dense --k 1 3 5

echo "== 4. Judge validation (kappa vs human gold) =="
QAUDIT_RETRIEVER="$RETRIEVER" python -m qaudit.eval.judge_eval --backends heuristic transformers

echo "== 5. Ablation (retrieved vs full-context) =="
QAUDIT_RETRIEVER="$RETRIEVER" python -m qaudit.eval.ablation --backend transformers --modes retrieved full

echo "Done. Reports in reports/."
