"""Hooked teacher forward passes, artifact extraction, and the CEA ablation engine."""

from ceed_teacher.ablation import (
    AttributionResult,
    attribution_vector,
    replace_expert_output,
    select_probed_experts,
)
from ceed_teacher.protocol import HybridForward
from ceed_teacher.synthetic import SyntheticConfig, SyntheticHybridTeacher

__version__ = "0.1.0"

__all__ = [
    "AttributionResult",
    "HybridForward",
    "SyntheticConfig",
    "SyntheticHybridTeacher",
    "attribution_vector",
    "replace_expert_output",
    "select_probed_experts",
]
