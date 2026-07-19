"""The evaluation harness - the actual point of the project.

Three measurements, each ending in a number the README quotes and an interviewer can
probe:

* :mod:`qaudit.eval.retrieval_eval` - hit rate@k, recall@k and MRR for BM25 vs dense
  against the retrieval gold set,
* :mod:`qaudit.eval.judge_eval` - the LLM-as-judge's agreement with the human-labelled
  gold set (Cohen's kappa), with the systematic disagreements surfaced, not smoothed,
* :mod:`qaudit.eval.ablation` - retrieved top-k context vs. whole-KB stuffing, ending
  in a ship decision backed by those metrics.
"""

from __future__ import annotations
