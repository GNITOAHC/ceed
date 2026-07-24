"""CEED domain types, configuration schema, artifact store, and extraction fingerprinting.

This package is the shared vocabulary every other CEED package imports. It
deliberately depends on no model, dataset, or training framework, so that the
heavy and mutually hostile dependency trees of the other packages cannot reach
it (see docs/adr/0004-seven-package-uv-workspace.md).
"""

from ceed_core.config import (
    AuxiliarySignalConfig,
    CorpusConfig,
    ExtractionConfig,
    GroupConfig,
    LayerMapping,
    ParamEfficiencyMode,
    Phase2Variant,
    StudentConfig,
    canonical_json,
    extraction_fingerprint,
    resolve_group_config,
    run_hash,
)
from ceed_core.metrics import MetricsSink, Tracker
from ceed_core.run_record import RunRecord

__version__ = "0.1.0"

__all__ = [
    "AuxiliarySignalConfig",
    "CorpusConfig",
    "ExtractionConfig",
    "GroupConfig",
    "LayerMapping",
    "MetricsSink",
    "ParamEfficiencyMode",
    "Phase2Variant",
    "RunRecord",
    "StudentConfig",
    "Tracker",
    "canonical_json",
    "extraction_fingerprint",
    "resolve_group_config",
    "run_hash",
]
