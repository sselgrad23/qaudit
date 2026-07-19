"""Data layer: real support conversations from the Bitext corpus, typed.

:class:`~qaudit.data.load.Conversation` is the single unit the whole system scores.
:func:`~qaudit.data.load.load_conversations` reads the committed evaluation set that
the gold labels reference by id; :func:`~qaudit.data.load.build_eval_set` rebuilds
that set from the Bitext source.
"""

from __future__ import annotations

from qaudit.data.load import Conversation, Turn, build_eval_set, load_conversations

__all__ = ["Conversation", "Turn", "build_eval_set", "load_conversations"]
