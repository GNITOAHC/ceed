"""Accuracy and hallucination evaluation, plus the grounding-selectivity protocol."""

from ceed_eval.direct import DIRECT_HARNESS_VERSION, DirectEvaluator
from ceed_eval.harness import DATASET_TASKS, LmmsEvalEvaluator, harness_version
from ceed_eval.metrics import (
    DATASET_METRICS,
    anls,
    exact_match,
    normalize_answer,
    relaxed_accuracy,
    score_answer,
)

__version__ = "0.1.0"

__all__ = [
    "DATASET_METRICS",
    "DATASET_TASKS",
    "DIRECT_HARNESS_VERSION",
    "DirectEvaluator",
    "LmmsEvalEvaluator",
    "anls",
    "exact_match",
    "harness_version",
    "normalize_answer",
    "relaxed_accuracy",
    "score_answer",
]
