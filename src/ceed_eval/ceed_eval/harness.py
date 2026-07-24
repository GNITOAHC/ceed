"""The ``lmms-eval`` adapter that produces a Group's accuracy numbers.

Accuracy is produced by ``lmms-eval`` rather than a bespoke metric, so B0's
numbers are comparable to published baselines. The adapter's job is to wire the
Student — loaded in fp16 with greedy decoding enforced — to the harness for the
Group's datasets, and to report the accuracies together with the harness version
and the decoding actually used, so the run record is self-describing.

:func:`harness_version` is a light metadata lookup and is tested in the fast
tier. :class:`LmmsEvalEvaluator` needs the real Student and a GPU and so runs out
of the fast tier.
"""

from __future__ import annotations

from importlib.metadata import version

from ceed_core import GroupConfig
from ceed_student import EvaluationOutcome, enforce_greedy, load_student

# Maps CEED dataset names to the lmms-eval task that scores them.
DATASET_TASKS = {
    "docvqa": "docvqa_val",
    "gqa": "gqa",
    "chartqa": "chartqa",
}


def harness_version() -> str:
    """Return the installed ``lmms-eval`` version, for recording on the run record."""
    return version("lmms-eval")


class LmmsEvalEvaluator:
    """Evaluates a Group by driving the Student through ``lmms-eval``."""

    def __init__(self, device: str = "cuda", limit: int | None = None) -> None:
        """Configure the evaluator.

        Args:
            device: The device to load the Student on.
            limit: An optional cap on examples per dataset, for smoke runs.
        """
        self.device = device
        self.limit = limit

    def evaluate(self, config: GroupConfig) -> EvaluationOutcome:  # pragma: no cover - needs GPU
        """Evaluate ``config``'s Student on its datasets and report accuracies.

        Args:
            config: A resolved Group configuration whose ``evaluation`` names the
                datasets and decoding.

        Returns:
            The per-dataset accuracies, the harness version, and the decoding
            used.

        Raises:
            ValueError: If the Group declares no evaluation.
        """
        if config.evaluation is None:
            raise ValueError(f"Group {config.group_code} declares no evaluation")

        from lmms_eval.evaluator import simple_evaluate

        decoding = config.evaluation.decoding
        model, _ = load_student(
            config.student.model, decoding, dtype=config.student.dtype, device=self.device
        )
        enforce_greedy(model.generation_config, decoding)

        tasks = [DATASET_TASKS[name] for name in config.evaluation.datasets]
        results = simple_evaluate(model=model, tasks=tasks, limit=self.limit)

        accuracies = {
            name: _accuracy(results["results"][DATASET_TASKS[name]])
            for name in config.evaluation.datasets
        }
        return EvaluationOutcome(
            accuracies=accuracies, harness_version=harness_version(), decoding=decoding
        )


def _accuracy(task_result: dict[str, float]) -> float:  # pragma: no cover - needs GPU
    """Pull a single accuracy scalar out of an lmms-eval task result."""
    for key, value in task_result.items():
        if "acc" in key or "anls" in key or "relaxed" in key:
            return float(value)
    raise KeyError(f"no accuracy-like metric in {sorted(task_result)}")
