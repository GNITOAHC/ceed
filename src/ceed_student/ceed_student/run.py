"""The ``run_group`` spine: the single entry point for running one Group.

Everything the plan calls a Group — B0 through E4, at any seed, under either
parameter-efficiency mode — is one :func:`run_group` call. This module is the
highest seam in the system: data loading, store reads, training, evaluation, and
metric emission all sit below it, and behaviour is tested by driving this
function and asserting on the :class:`~ceed_core.run_record.RunRecord` it
returns.

Training and evaluation both enter through injected seams — a
:class:`~ceed_student.training.Trainer` and a
:class:`~ceed_student.evaluation.Evaluator` — rather than imports, so the spine
stays free of ``accelerate``, the model and the harness at import time and
remains exercisable on CPU. A Group with a ``training`` config and a trainer —
B1 is the first — trains the Student and records the frozen checkpoint before it
evaluates; a Group with an ``evaluation`` and an evaluator — B0 is the first —
runs the harness and records its accuracies; a Group with neither — the null
Group — still runs end to end and writes a valid record.
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
from ceed_student.training import Trainer


def run_group(
    config: GroupConfig,
    output_dir: Path,
    tracker: Tracker | None = None,
    trainer: Trainer | None = None,
    evaluator: Evaluator | None = None,
) -> RunRecord:
    """Run one Group and return its run record.

    The configuration is hashed to a run identifier, which names a run directory
    under ``output_dir``. A JSONL metrics sink is opened there as the source of
    truth. If the Group declares a ``training`` and a trainer is supplied, the
    Student is trained first and its backbone metrics, parameter-efficiency mode,
    and frozen checkpoint are recorded. If the Group declares an ``evaluation``
    and an evaluator is supplied, the harness then runs and its accuracies,
    harness version, and decoding settings are recorded; otherwise the run
    records only the Group's identity. Either way a run record is written to the
    run directory and returned.

    The run directory is created if absent and reused if present, so re-running
    the same configuration writes to the same place — the run identifier is a
    pure function of the configuration, and a completed training checkpoint is
    resumed rather than retrained.

    Args:
        config: A fully-resolved Group configuration.
        output_dir: The directory under which the run directory is created.
        tracker: An optional tracking-service callback mirrored with each
            emitted metric. Its absence is not an error; disk is authoritative.
        trainer: The trainer to run when the Group declares training. Injected
            so ``accelerate`` and the model never reach this module at import
            time.
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
    checkpoint_dir: str | None = None
    # Defaults to the configured mode; overwritten by the mode the trainer
    # actually ran, so the record can never claim a mode that did not train
    # (ADR-0005).
    param_efficiency = config.param_efficiency
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

        if config.training is not None and trainer is not None:
            training = trainer.train(config)
            checkpoint_dir = training.checkpoint_dir
            param_efficiency = training.param_efficiency
            metrics = {
                **metrics,
                "train.final_loss": training.final_loss,
                "train.cross_entropy": training.backbone_metrics["cross_entropy"],
                "train.kd": training.backbone_metrics["kd"],
            }
            if training.lora_rank is not None:
                metrics["train.lora_rank"] = float(training.lora_rank)
            sink.log(
                {
                    "event": "run_group.trained",
                    "param_efficiency": training.param_efficiency.value,
                    "lora_rank": training.lora_rank,
                    "steps": training.steps,
                    "steps_run": training.steps_run,
                    "resumed": training.resumed,
                    "final_loss": training.final_loss,
                    "checkpoint_dir": checkpoint_dir,
                    "trainable_parameters": training.trainable_parameters,
                    "total_parameters": training.total_parameters,
                }
            )

        if config.evaluation is not None and evaluator is not None:
            outcome = evaluator.evaluate(config)
            metrics = {**metrics, **outcome.accuracies, **outcome.extra_metrics}
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
            param_efficiency=param_efficiency,
            layer_mapping=config.layer_mapping,
            config_hash=config_hash,
            extraction_fingerprint=fingerprint,
            metrics_path=str(sink.path),
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            harness_version=harness_version,
            decoding=decoding,
        )

    record.write(run_dir)
    return record
