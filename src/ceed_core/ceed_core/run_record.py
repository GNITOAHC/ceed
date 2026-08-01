"""The run record: the on-disk statement of what a Group run was.

A :class:`RunRecord` is the return value of
:func:`ceed_student.run.run_group` and is also written to disk, so that the
Phase 1 comparison table is a computation over files the researcher owns rather
than a reading off a dashboard. It records the identity of the run — its Group,
seed, parameter-efficiency mode, layer mapping, and the two configuration
hashes — and where the run's metrics were written.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ceed_core.config import DecodingConfig, ParamEfficiencyMode
from ceed_core.layer_mapping import LayerMapping

RUN_RECORD_FILENAME = "run_record.json"


class RunRecord(BaseModel):
    """A durable statement of a single Group run's identity and outputs.

    Attributes:
        group_code: The Group that was run.
        seed: The seed it ran under.
        param_efficiency: Whether the Student was LoRA or fully fine-tuned. A
            LoRA number must never reach the Phase 1 headline table, so this is
            recorded unambiguously.
        layer_mapping: The Teacher-to-Student mapping the run used, recorded so
            the layer-mapping robustness claim is checkable.
        config_hash: The run identifier — the hash of the whole configuration.
        extraction_fingerprint: The hash of the extraction subset, identifying
            the artifact store the run read from.
        metrics_path: Where the run's JSONL metrics were written, as the source
            of truth.
        metrics: The headline metrics the run produced, such as per-dataset
            accuracy and, for a trained Group, the final backbone loss terms;
            empty for a run that produced none.
        checkpoint_dir: Where the trained Student checkpoint was frozen, or
            ``None`` for a Group that trained nothing (B0). A baseline's
            checkpoint is reused rather than retrained, so this is the durable
            handle to it.
        harness_version: The evaluation harness version the metrics came from,
            or ``None`` if the run did not evaluate.
        decoding: The decoding settings actually used, or ``None`` if the run
            did not decode.
    """

    model_config = ConfigDict(frozen=True)

    group_code: str
    seed: int
    param_efficiency: ParamEfficiencyMode
    layer_mapping: LayerMapping
    config_hash: str
    extraction_fingerprint: str
    metrics_path: str
    metrics: dict[str, float] = {}
    checkpoint_dir: str | None = None
    harness_version: str | None = None
    decoding: DecodingConfig | None = None

    def write(self, directory: Path) -> Path:
        """Write the record as pretty-printed JSON into ``directory``.

        Args:
            directory: The run directory to write into. It must already exist.

        Returns:
            The path of the written file.
        """
        path = directory / RUN_RECORD_FILENAME
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def read(cls, directory: Path) -> RunRecord:
        """Read a record previously written by :meth:`write`.

        Args:
            directory: The run directory holding the record.

        Returns:
            The parsed record.
        """
        return cls.model_validate_json((directory / RUN_RECORD_FILENAME).read_text())
