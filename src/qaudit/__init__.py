"""qaudit - a support-ticket RAG QA assistant, built around its evaluation harness.

The product is small on purpose; the point is measurement. A customer support
conversation goes in, the relevant slice of that customer's own quality rubric and
policy knowledge base is *retrieved*, and an LLM-as-judge scores the conversation
against each retrieved criterion with a verbatim evidence quote. What makes it a
portfolio piece rather than a demo is the harness around it:

* :mod:`qaudit.retrieval` - BM25 and dense retrievers over the rubric/policy KB,
* :mod:`qaudit.judge`     - the LLM-as-judge (LLM + heuristic backends), and
* :mod:`qaudit.eval`      - the three measurements that decide whether it works:
    retrieval quality (hit rate@k, recall@k, MRR), judge validity against a
    human-labelled gold set (Cohen's kappa), and a retrieved-vs-stuffed ablation
    that ends in a ship decision.
"""

from __future__ import annotations

__version__ = "0.1.0"
