"""The evaluation seam ``run_group`` drives a Group's numbers through.

Evaluation is expressed as a protocol so that the heavy path — loading the
Student and running ``lmms-eval`` on a GPU — is injected rather than imported.
``run_group`` depends only on :class:`Evaluator`; the real implementation lives
in ``ceed_eval`` and a fake one stands in for the fast test tier. This is what
lets the whole spine be exercised on CPU while B0 still runs the real harness in
production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ceed_core import DecodingConfig, GroupConfig


@dataclass(frozen=True)
class EvaluationOutcome:
    """What an evaluator reports back for a Group.

    Attributes:
        accuracies: Per-dataset accuracy, keyed by dataset name.
        harness_version: The version of the evaluation harness that produced the
            accuracies, recorded so the numbers are attributable.
        decoding: The decoding settings actually used.
        extra_metrics: Any additional scalar metrics to record alongside
            accuracy.
    """

    accuracies: dict[str, float]
    harness_version: str
    decoding: DecodingConfig
    extra_metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Something that can evaluate a resolved Group and report its numbers."""

    def evaluate(self, config: GroupConfig) -> EvaluationOutcome:
        """Evaluate ``config`` and return its accuracies and decoding provenance."""
        ...
