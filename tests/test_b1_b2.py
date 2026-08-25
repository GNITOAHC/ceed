"""B1 and B2 through run_group: trained, then evaluated, with provenance recorded.

The fast tier drives B1 and B2 through the spine with a stand-in trainer and
evaluator, so the whole path — training provenance, the frozen checkpoint, the
parameter-efficiency mode, then accuracy — is asserted without an 8B Student or a
GPU. What distinguishes B1 from B2 (the logit-distillation weight) is a config
property, checked directly. The real LoRA fine-tune of B2 on the Volta box is a
gpu-tier test.
"""

import json
from pathlib import Path

import pytest

from ceed_core import ParamEfficiencyMode, RunRecord
from ceed_eval import harness_version
from ceed_student import EvaluationOutcome, TrainingOutcome, run_group


class StubTrainer:
    """A trainer that reports a fixed outcome without building a model."""

    def __init__(self, mode=ParamEfficiencyMode.LORA, checkpoint_dir="/tmp/ckpt"):
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.trained: list = []

    def train(self, config) -> TrainingOutcome:
        assert config.training is not None
        self.trained.append(config.group_code)
        return TrainingOutcome(
            param_efficiency=self.mode,
            lora_rank=4 if self.mode is ParamEfficiencyMode.LORA else None,
            steps=config.training.steps,
            steps_run=config.training.steps,
            batch_size=config.training.batch_size,
            examples_seen=config.training.steps * config.training.batch_size,
            epochs=2.0,
            resumed=False,
            initial_loss=3.0,
            final_loss=0.7,
            backbone_metrics={"cross_entropy": 0.6, "kd": 0.1},
            checkpoint_dir=self.checkpoint_dir,
            total_parameters=1000,
            trainable_parameters=900,
        )


class StubEvaluator:
    def __init__(self, accuracies):
        self.accuracies = accuracies

    def evaluate(self, config) -> EvaluationOutcome:
        return EvaluationOutcome(
            accuracies=self.accuracies,
            harness_version=harness_version(),
            decoding=config.evaluation.decoding,
        )


# -- config composition ------------------------------------------------------


def test_b1_is_supervised_only_and_b2_distils(b1_config, b2_config):
    assert b1_config.group_code == "B1"
    assert b2_config.group_code == "B2"
    # The shared backbone is one implementation; B1 and B2 differ only in weight.
    assert b1_config.training.backbone.kd_weight == 0.0
    assert b2_config.training.backbone.kd_weight == 1.0
    assert not b1_config.auxiliary_signals
    assert not b2_config.auxiliary_signals


def test_both_declare_a_corpus_training_and_evaluation(b1_config, b2_config):
    for config in (b1_config, b2_config):
        assert config.corpus is not None
        assert config.training is not None
        assert config.training.steps > 0
        assert config.evaluation is not None


# -- through run_group -------------------------------------------------------


def test_b2_trains_then_evaluates_and_records_both(b2_config, tmp_path):
    trainer = StubTrainer()
    record = run_group(
        b2_config,
        tmp_path,
        trainer=trainer,
        evaluator=StubEvaluator({"docvqa": 0.7, "gqa": 0.6, "chartqa": 0.4}),
    )

    assert trainer.trained == ["B2"]
    # Accuracy and the training decomposition both land on the record.
    assert record.metrics["docvqa"] == 0.7
    assert record.metrics["train.final_loss"] == 0.7
    assert record.metrics["train.cross_entropy"] == 0.6
    assert record.checkpoint_dir == "/tmp/ckpt"
    assert record.harness_version == harness_version()


def test_the_parameter_efficiency_mode_is_recorded_unambiguously(b2_config, tmp_path):
    record = run_group(b2_config, tmp_path, trainer=StubTrainer(ParamEfficiencyMode.LORA))
    # base.yaml sets lora as the development default; the record states it.
    assert record.param_efficiency is ParamEfficiencyMode.LORA
    # A LoRA run's adapter rank reaches the durable record (ADR-0005).
    assert record.metrics["train.lora_rank"] == 4.0
    assert RunRecord.read(tmp_path / record.config_hash) == record


def test_the_record_reflects_the_mode_that_actually_trained_not_config_intent(b2_config, tmp_path):
    # The record's mode is sourced from the training outcome, not the config, so
    # it can never claim a mode that did not train — the failure ADR-0005 guards.
    record = run_group(b2_config, tmp_path, trainer=StubTrainer(ParamEfficiencyMode.FULL))
    assert record.param_efficiency is ParamEfficiencyMode.FULL
    assert "train.lora_rank" not in record.metrics


def test_training_emits_a_trained_event_to_the_metrics_jsonl(b1_config, tmp_path):
    record = run_group(b1_config, tmp_path, trainer=StubTrainer())
    lines = [json.loads(line) for line in Path(record.metrics_path).read_text().splitlines()]
    trained = [line for line in lines if line["event"] == "run_group.trained"]
    assert len(trained) == 1
    assert trained[0]["steps"] == b1_config.training.steps
    assert trained[0]["checkpoint_dir"] == "/tmp/ckpt"


def test_a_group_declaring_training_without_a_trainer_records_no_checkpoint(b2_config, tmp_path):
    # Symmetry with evaluation: declaring training but supplying no trainer still
    # produces a valid, checkpoint-free record rather than failing.
    record = run_group(b2_config, tmp_path, trainer=None, evaluator=None)
    assert record.checkpoint_dir is None
    assert "train.final_loss" not in record.metrics


def test_b1_and_b2_share_a_checkpoint_only_within_their_own_run(b1_config, b2_config, tmp_path):
    # Two different Groups hash to two different run directories, so neither can
    # accidentally read the other's frozen checkpoint.
    r1 = run_group(b1_config, tmp_path, trainer=StubTrainer())
    r2 = run_group(b2_config, tmp_path, trainer=StubTrainer())
    assert r1.config_hash != r2.config_hash


# -- the real LoRA fine-tune of B2 on a GPU (criterion 6) -------------------


@pytest.mark.gpu
def test_b2_lora_completes_at_toy_scale_on_the_volta_box(b2_config, tmp_path):
    """A real LoRA fine-tune of B2 runs a handful of steps end to end on a GPU.

    Criterion 6: a LoRA run of B2 completes on the Volta box at toy scale. This
    needs the real Student weights and a CUDA device, and reads teacher artefacts
    from a store, so it is a gpu-tier test wired once the real Student builder and
    store-backed dataloader land. Skipped until then.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    pytest.skip("real Student builder and store-backed dataloader are a gpu-tier follow-on")
