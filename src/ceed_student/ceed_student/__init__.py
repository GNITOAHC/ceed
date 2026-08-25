"""Student wrapper, removable probes, auxiliary signals, and the training loop."""

from ceed_student.auxiliary import (
    AuxiliarySignal,
    AuxTerm,
    ForwardView,
    WarmupSchedule,
    aux_terms,
    coupling_gate,
    required_artefact_kinds,
    required_forward_views,
    required_student_layers,
    training_loss,
    verify_store_supports,
)
from ceed_student.backbone import BackboneLoss, backbone_loss, topk_kd_per_token
from ceed_student.dataset import build_batches, encode_example, load_corpus_split
from ceed_student.evaluation import EvaluationOutcome, Evaluator
from ceed_student.run import run_group
from ceed_student.signals import (
    CombineWeightProbeSignal,
    HiddenStateProjectionSignal,
    VisualAdvantageSignal,
    build_signals,
    signal_artefact_kinds,
    va_group_weights,
)
from ceed_student.student import (
    CeedStudent,
    apply_adapter,
    enforce_greedy,
    load_student,
    load_trained_student,
)
from ceed_student.training import (
    AccelerateTrainer,
    SignalContext,
    StudentForward,
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
    "CeedStudent",
    "CombineWeightProbeSignal",
    "EvaluationOutcome",
    "Evaluator",
    "ForwardView",
    "HiddenStateProjectionSignal",
    "SignalContext",
    "StudentForward",
    "TrainableStudent",
    "Trainer",
    "TrainingBatch",
    "TrainingOutcome",
    "VisualAdvantageSignal",
    "WarmupSchedule",
    "apply_adapter",
    "aux_terms",
    "backbone_loss",
    "build_batches",
    "build_signals",
    "coupling_gate",
    "encode_example",
    "enforce_greedy",
    "load_corpus_split",
    "load_student",
    "load_trained_student",
    "required_artefact_kinds",
    "required_forward_views",
    "required_student_layers",
    "run_group",
    "signal_artefact_kinds",
    "topk_kd_per_token",
    "training_loss",
    "va_group_weights",
    "verify_store_supports",
]
