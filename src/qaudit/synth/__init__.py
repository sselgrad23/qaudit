"""Synthetic rare-violation generation.

Real Bitext responses are canonical and almost always compliant, so rubric
violations are vanishingly rare - which would make a judge-validation set trivially
all-pass and a kappa meaningless. This module fills the violation class with
*known-label* hard negatives: a real, compliant response is corrupted with one
specific, deliberately-injected violation. See :mod:`qaudit.synth.generate`.
"""

from __future__ import annotations

from qaudit.synth.generate import CORRUPTORS, build_synthetic_set, corrupt

__all__ = ["CORRUPTORS", "build_synthetic_set", "corrupt"]
