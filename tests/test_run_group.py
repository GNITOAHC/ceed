"""The primary seam: drive run_group and assert on the returned run record.

These tests say nothing about how run_group works inside. They compose the null
Group from the repository's own YAML overlays, run it, and check the record and
the on-disk artefacts — the contract every later ticket extends beneath this
same seam.
"""

import json
from pathlib import Path

from ceed_core.run_record import RUN_RECORD_FILENAME, RunRecord

from ceed_core import ParamEfficiencyMode, run_hash
from ceed_student import run_group


def test_null_group_runs_end_to_end_and_returns_a_record(null_config, tmp_path):
    record = run_group(null_config, tmp_path)

    assert record.group_code == "NULL"
    assert record.seed == 0
    assert record.param_efficiency is ParamEfficiencyMode.LORA
    assert record.layer_mapping.kind == "proportional"
    assert record.config_hash == run_hash(null_config)


def test_the_record_is_written_under_the_config_hash(null_config, tmp_path):
    record = run_group(null_config, tmp_path)

    run_dir = tmp_path / record.config_hash
    assert (run_dir / RUN_RECORD_FILENAME).is_file()
    assert RunRecord.read(run_dir) == record


def test_metrics_are_written_to_jsonl_on_disk(null_config, tmp_path):
    record = run_group(null_config, tmp_path)

    metrics_path = Path(record.metrics_path)
    assert metrics_path.is_file()
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert lines, "the run must emit at least one metric record"
    assert all("event" in line for line in lines)


def test_a_tracking_service_is_optional_and_its_absence_is_not_an_error(null_config, tmp_path):
    # No tracker supplied: the run still succeeds and disk still holds the metrics.
    record = run_group(null_config, tmp_path, tracker=None)
    assert Path(record.metrics_path).is_file()


def test_a_tracking_service_when_present_mirrors_every_metric(null_config, tmp_path):
    mirrored = []
    record = run_group(null_config, tmp_path, tracker=mirrored.append)

    disk_lines = Path(record.metrics_path).read_text().splitlines()
    assert len(mirrored) == len(disk_lines)


def test_the_config_hash_is_stable_across_runs(null_config, tmp_path):
    first = run_group(null_config, tmp_path / "a")
    second = run_group(null_config, tmp_path / "b")
    assert first.config_hash == second.config_hash


def test_rerunning_into_the_same_directory_replaces_rather_than_appends(null_config, tmp_path):
    # The run directory is a pure function of the config, so a second run of the
    # same Group reuses it. The metrics file is the source of truth and must not
    # silently accumulate a duplicate stream.
    first = run_group(null_config, tmp_path)
    first_lines = Path(first.metrics_path).read_text().splitlines()

    second = run_group(null_config, tmp_path)
    second_lines = Path(second.metrics_path).read_text().splitlines()

    assert Path(first.metrics_path) == Path(second.metrics_path)
    assert len(second_lines) == len(first_lines)
    assert RunRecord.read(tmp_path / second.config_hash) == second
