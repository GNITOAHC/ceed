"""The ``run_group`` spine: the single entry point for running one Group.

Everything the plan calls a Group — B0 through E4, at any seed, under either
parameter-efficiency mode — is one :func:`run_group` call. This module is the
highest seam in the system: data loading, store reads, training, evaluation, and
metric emission all sit below it, and behaviour is tested by driving this
function and asserting on the :class:`~ceed_core.run_record.RunRecord` it
returns.

Evaluation enters through an injected :class:`~ceed_student.evaluation.Evaluator`
rather than an import, so the spine stays free of the model and harness at import
time and remains exercisable on CPU. A Group with an ``evaluation`` config and an
evaluator — B0 is the first — runs the harness and records its accuracies; a
Group with neither — the null Group — still runs end to end and writes a valid
record.
"""

from __future__ import annotations

from pathlib import Path

from ceed_core import (
    GroupConfig,
    MetricsSink,
    RunRecord,
    Tracker,
    extraction_fingerprint,
    run_hash,
)
from ceed_student.evaluation import Evaluator


def run_group(
    config: GroupConfig,
    output_dir: Path,
    tracker: Tracker | None = None,
    evaluator: Evaluator | None = None,
) -> RunRecord:
    """Run one Group and return its run record.

    The configuration is hashed to a run identifier, which names a run directory
    under ``output_dir``. A JSONL metrics sink is opened there as the source of
    truth. If the Group declares an ``evaluation`` and an evaluator is supplied,
    the harness runs and its accuracies, harness version, and decoding settings
    are recorded; otherwise the run records only the Group's identity. Either
    way a run record is written to the run directory and returned.

    The run directory is created if absent and reused if present, so re-running
    the same configuration writes to the same place — the run identifier is a
    pure function of the configuration.

    Args:
        config: A fully-resolved Group configuration.
        output_dir: The directory under which the run directory is created.
        tracker: An optional tracking-service callback mirrored with each
            emitted metric. Its absence is not an error; disk is authoritative.
        evaluator: The evaluator to run when the Group declares an evaluation.
            Injected so the model and harness never reach this module at import
            time.

    Returns:
        The run record, also written to ``output_dir/<run_hash>/run_record.json``.
    """
    config_hash = run_hash(config)
    fingerprint = extraction_fingerprint(config)

    run_dir = output_dir / config_hash
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, float] = {}
    harness_version: str | None = None
    decoding = None

    with MetricsSink(run_dir / "metrics.jsonl", tracker) as sink:
        sink.log(
            {
                "event": "run_group.start",
                "group_code": config.group_code,
                "seed": config.seed,
                "param_efficiency": config.param_efficiency.value,
                "config_hash": config_hash,
                "extraction_fingerprint": fingerprint,
            }
        )

        if config.evaluation is not None and evaluator is not None:
            outcome = evaluator.evaluate(config)
            metrics = {**outcome.accuracies, **outcome.extra_metrics}
            harness_version = outcome.harness_version
            decoding = outcome.decoding
            for dataset, accuracy in outcome.accuracies.items():
                sink.log(
                    {
                        "event": "run_group.accuracy",
                        "dataset": dataset,
                        "accuracy": accuracy,
                        "harness_version": harness_version,
                    }
                )

        sink.log({"event": "run_group.finish", "group_code": config.group_code})

        record = RunRecord(
            group_code=config.group_code,
            seed=config.seed,
            param_efficiency=config.param_efficiency,
            layer_mapping=config.layer_mapping,
            config_hash=config_hash,
            extraction_fingerprint=fingerprint,
            metrics_path=str(sink.path),
            metrics=metrics,
            harness_version=harness_version,
            decoding=decoding,
        )

    record.write(run_dir)
    return record
