"""The retrieval corpus: a customer's policy / help-centre knowledge base.

The KB is a data file (``documents.json``) of policy passages, one customer's
"unique rubric" material in the sense the target problem means it. Retrieval finds
the passages relevant to a conversation; the judge uses them as evidence. A new
customer is a new KB, not new code.
"""

from __future__ import annotations

from qaudit.kb.knowledge_base import Passage, load_kb

__all__ = ["Passage", "load_kb"]
