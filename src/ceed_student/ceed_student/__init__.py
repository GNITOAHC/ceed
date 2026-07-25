"""Student wrapper, removable probes, auxiliary signals, and the training loop."""

from ceed_student.auxiliary import (
    AuxiliarySignal,
    AuxTerm,
    ForwardView,
    WarmupSchedule,
    coupling_gate,
    required_artefact_kinds,
    required_forward_views,
    training_loss,
    verify_store_supports,
)
from ceed_student.backbone import BackboneLoss, backbone_loss
from ceed_student.evaluation import EvaluationOutcome, Evaluator
from ceed_student.run import run_group
from ceed_student.student import enforce_greedy, load_student
from ceed_student.training import (
    AccelerateTrainer,
    TrainableStudent,
    Trainer,
    TrainingBatch,
    TrainingOutcome,
)

__version__ = "0.1.0"

__all__ = [
    "AccelerateTrainer",
    "AuxTerm",
    "AuxiliarySignal",
    "BackboneLoss",
    "EvaluationOutcome",
    "Evaluator",
    "ForwardView",
    "TrainableStudent",
    "Trainer",
    "TrainingBatch",
    "TrainingOutcome",
    "WarmupSchedule",
    "backbone_loss",
    "coupling_gate",
    "enforce_greedy",
    "load_student",
    "required_artefact_kinds",
    "required_forward_views",
    "run_group",
    "training_loss",
    "verify_store_supports",
]
