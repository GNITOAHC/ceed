"""Configuration composition, validation, and the two hashes.

The properties under test are the ones ADR-0001 and the ticket rest on: a
configuration hashes independently of field order and formatting; any meaningful
change moves the run hash; and the extraction fingerprint moves only for
extraction-relevant changes, so a Group's seed does not force a fresh store while
its layer set does.
"""

import pytest
from pydantic import ValidationError

from ceed_core import (
    ExtractionConfig,
    GroupConfig,
    LayerMapping,
    ParamEfficiencyMode,
    StudentConfig,
    extraction_fingerprint,
    resolve_group_config,
    run_hash,
)


def a_config(**overrides):
    """Build a valid GroupConfig directly, with field overrides for hash tests."""
    fields = {
        "group_code": "B0",
        "seed": 0,
        "param_efficiency": ParamEfficiencyMode.LORA,
        "layer_mapping": LayerMapping(kind="proportional", pairs=((9, 12), (29, 39))),
        "student": StudentConfig(model="gemma-4-E4B-it", dtype="float16"),
        "extraction": ExtractionConfig(
            teacher_model="gemma-4-26B-A4B-it",
            model_revision="main",
            layers=(9, 19, 29),
            ablation="mean-of-active",
            combine_weight="effective",
        ),
    }
    fields.update(overrides)
    return GroupConfig(**fields)


def test_overlays_compose_into_a_valid_null_group(null_config):
    assert null_config.group_code == "NULL"
    assert null_config.corpus is None
    assert null_config.auxiliary_signals == ()
    assert null_config.student.model == "google/gemma-4-e4b-it"


# Two YAML files with the same content but different key order, indentation
# style, and comments. Composing each must produce the same run hash.
_YAML_ORDERED = """\
group_code: B0
seed: 3
param_efficiency: lora
layer_mapping:
  kind: proportional
  pairs:
    - [9, 12]
    - [29, 39]
student:
  model: gemma-4-E4B-it
  dtype: float16
extraction:
  teacher_model: gemma-4-26B-A4B-it
  model_revision: main
  layers: [9, 19, 29]
  ablation: mean-of-active
  combine_weight: effective
  thinking_enabled: false
"""

_YAML_REORDERED = """\
# same configuration, keys shuffled and reformatted
extraction:
  combine_weight: effective
  ablation: mean-of-active
  thinking_enabled: false
  model_revision: main
  teacher_model: gemma-4-26B-A4B-it
  layers: [9, 19, 29]
student:
  dtype: float16
  model: gemma-4-E4B-it
param_efficiency: lora
seed:    3
layer_mapping:
  pairs: [[9, 12], [29, 39]]
  kind: proportional
group_code: B0
"""


def test_yaml_field_order_and_formatting_do_not_change_the_hash(tmp_path):
    ordered = tmp_path / "ordered.yaml"
    reordered = tmp_path / "reordered.yaml"
    ordered.write_text(_YAML_ORDERED)
    reordered.write_text(_YAML_REORDERED)

    assert run_hash(resolve_group_config([ordered])) == run_hash(resolve_group_config([reordered]))


def test_a_meaningful_change_changes_the_run_hash():
    assert run_hash(a_config()) != run_hash(a_config(seed=1))


def test_the_seed_does_not_change_the_extraction_fingerprint():
    # Seed is part of the run identity but irrelevant to what was extracted.
    assert extraction_fingerprint(a_config()) == extraction_fingerprint(a_config(seed=1))
    assert run_hash(a_config()) != run_hash(a_config(seed=1))


def test_an_extraction_change_changes_both_hashes():
    other = a_config(
        extraction=ExtractionConfig(
            teacher_model="gemma-4-26B-A4B-it",
            model_revision="main",
            layers=(1, 2, 3),
            ablation="mean-of-active",
            combine_weight="effective",
        )
    )
    assert extraction_fingerprint(a_config()) != extraction_fingerprint(other)
    assert run_hash(a_config()) != run_hash(other)


def test_an_unknown_field_is_rejected_at_load_time():
    with pytest.raises(ValidationError, match="typo_field"):
        GroupConfig.model_validate({**a_config().model_dump(), "typo_field": 1})


def test_a_negative_seed_is_rejected():
    with pytest.raises(ValidationError, match="seed"):
        a_config(seed=-1)


def test_an_unknown_param_efficiency_mode_is_rejected():
    with pytest.raises(ValidationError):
        GroupConfig.model_validate({**a_config().model_dump(), "param_efficiency": "quantized"})


def test_an_unknown_layer_mapping_kind_is_rejected():
    with pytest.raises(ValidationError, match="mapping kind"):
        LayerMapping(kind="diagonal", pairs=((0, 0),))
