"""Student wrapper, removable probes, auxiliary signals, and the training loop."""

from ceed_student.evaluation import EvaluationOutcome, Evaluator
from ceed_student.run import run_group
from ceed_student.student import enforce_greedy, load_student

__version__ = "0.1.0"

__all__ = [
    "EvaluationOutcome",
    "Evaluator",
    "enforce_greedy",
    "load_student",
    "run_group",
]
