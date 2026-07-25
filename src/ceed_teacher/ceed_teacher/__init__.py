"""Hooked teacher forward passes, artifact extraction, and the CEA ablation engine."""

from ceed_teacher.ablation import (
    AttributionResult,
    attribution_vector,
    replace_expert_output,
    select_probed_experts,
)
from ceed_teacher.extraction import (
    COMBINE_WEIGHTS,
    HIDDEN_STATES,
    TOP_K_LOGIT_IDS,
    TOP_K_LOGIT_VALUES,
    ExtractionSource,
    ExtractionSpec,
    create_store,
    extract_example,
    run_extraction,
    shard_examples,
    store_schema,
)
from ceed_teacher.protocol import HybridForward
from ceed_teacher.synthetic import SyntheticConfig, SyntheticHybridTeacher

__version__ = "0.1.0"

__all__ = [
    "COMBINE_WEIGHTS",
    "HIDDEN_STATES",
    "TOP_K_LOGIT_IDS",
    "TOP_K_LOGIT_VALUES",
    "AttributionResult",
    "ExtractionSource",
    "ExtractionSpec",
    "HybridForward",
    "SyntheticConfig",
    "SyntheticHybridTeacher",
    "attribution_vector",
    "create_store",
    "extract_example",
    "replace_expert_output",
    "run_extraction",
    "select_probed_experts",
    "shard_examples",
    "store_schema",
]
