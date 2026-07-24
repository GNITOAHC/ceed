"""The ``run_group`` spine: the single entry point for running one Group.

Everything the plan calls a Group — B0 through E4, at any seed, under either
parameter-efficiency mode — is one :func:`run_group` call. This module is the
highest seam in the system: data loading, store reads, training, evaluation, and
metric emission all sit below it, and behaviour is tested by driving this
function and asserting on the :class:`~ceed_core.run_record.RunRecord` it
returns.

At present the spine is complete but hollow. A Group with no auxiliary signals,
no model, and no corpus — the null Group — runs end to end and writes a valid
run record; later tickets fill in the corpus, the store reads, the training
loop, and evaluation beneath this same seam without changing its shape.

This module deliberately imports no model or training framework at import time,
so the null Group's spine is exercisable on CPU in the fast test tier.
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


def run_group(
    config: GroupConfig,
    output_dir: Path,
    tracker: Tracker | None = None,
) -> RunRecord:
    """Run one Group and return its run record.

    The configuration is hashed to a run identifier, which names a run directory
    under ``output_dir``. A JSONL metrics sink is opened there as the source of
    truth, the Group is run (a no-op for the null Group today), and a run record
    stating the Group's identity is both written to that directory and returned.

    The run directory is created if absent and reused if present, so re-running
    the same configuration writes to the same place — the run identifier is a
    pure function of the configuration.

    Args:
        config: A fully-resolved Group configuration.
        output_dir: The directory under which the run directory is created.
        tracker: An optional tracking-service callback mirrored with each
            emitted metric. Its absence is not an error; disk is authoritative.

    Returns:
        The run record, also written to ``output_dir/<run_hash>/run_record.json``.
    """
    config_hash = run_hash(config)
    fingerprint = extraction_fingerprint(config)

    run_dir = output_dir / config_hash
    run_dir.mkdir(parents=True, exist_ok=True)

    with MetricsSink(run_dir / "metrics.jsonl", tracker) as metrics:
        metrics.log(
            {
                "event": "run_group.start",
                "group_code": config.group_code,
                "seed": config.seed,
                "param_efficiency": config.param_efficiency.value,
                "config_hash": config_hash,
                "extraction_fingerprint": fingerprint,
            }
        )
        # Later tickets: load the corpus, read the artifact store, train the
        # Student under its auxiliary signals, evaluate, and log real metrics
        # here. The null Group has none of these, and that is a valid run.
        metrics.log({"event": "run_group.finish", "group_code": config.group_code})

        record = RunRecord(
            group_code=config.group_code,
            seed=config.seed,
            param_efficiency=config.param_efficiency,
            layer_mapping=config.layer_mapping,
            config_hash=config_hash,
            extraction_fingerprint=fingerprint,
            metrics_path=str(metrics.path),
        )

    record.write(run_dir)
    return record
