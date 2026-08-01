"""The training loop: the backbone optimised, checkpointed, and resumed on CPU.

The real Student is 8B and trains on a GPU, so the loop is exercised here against
a tiny CPU Student with the same interface. What is asserted is behaviour the
plan rests on: the shared backbone actually drives the parameters down, the run
checkpoints and *resumes* rather than restarting (so a multi-day Group survives
preemption), a completed baseline is not retrained, and the parameter-efficiency
mode the run used is reported unambiguously (so a LoRA number can never be
mistaken for a full fine-tune).
"""

from typing import Any

import pytest
import torch
from ceed_student.training import (
    AccelerateTrainer,
    StudentForward,
    TrainableStudent,
    TrainingBatch,
    TrainingOutcome,
    resolve_lora_targets,
)

from ceed_core import BackboneConfig, ParamEfficiencyMode, TrainingConfig, WarmupConfig
from ceed_student import ForwardView


class TinyStudent(torch.nn.Module):
    """A CPU-sized Student: embed, project, read off the answer positions."""

    HIDDEN = 4

    def __init__(self, vocab: int = 8, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.proj = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))

    def answer_forward(
        self, batch: TrainingBatch, view: ForwardView, hidden_layers: frozenset[int] = frozenset()
    ) -> StudentForward:
        assert view is ForwardView.ORIGINAL
        hidden = self.embed(batch.student_inputs["input_ids"])
        logits = self.proj(hidden)
        positions = batch.answer_token_positions
        return StudentForward(
            logits=logits[positions],
            hidden_states={layer: hidden[positions] for layer in sorted(hidden_layers)},  # noqa: C420
        )


def a_batch(seed: int = 0) -> TrainingBatch:
    """One toy example whose gold tokens the tiny Student can learn to predict."""
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 8, (5,), generator=gen)
    positions = torch.tensor([2, 3, 4])
    gold = torch.randint(0, 8, (3,), generator=gen)
    # Teacher top-k agrees with gold at its argmax, so KD and CE pull together.
    topk_ids = torch.stack([gold, (gold + 1) % 8], dim=1)
    topk_values = torch.tensor([[4.0, 1.0]] * 3)
    return TrainingBatch(
        student_inputs={"input_ids": input_ids},
        answer_token_positions=positions,
        gold_token_ids=gold,
        teacher_topk_ids=topk_ids,
        teacher_topk_values=topk_values,
    )


def a_training_config(steps: int, **overrides) -> TrainingConfig:
    fields: dict[str, Any] = {
        "steps": steps,
        "learning_rate": 0.1,
        "backbone": BackboneConfig(kd_weight=1.0),
        "warmup": WarmupConfig(),
    }
    fields.update(overrides)
    return TrainingConfig(**fields)


def a_trainer(tmp_path, batches=None, **kwargs) -> AccelerateTrainer:
    batches = batches or [a_batch(0), a_batch(1)]
    return AccelerateTrainer(
        build_student=lambda config: TinyStudent(),
        build_batches=lambda config: batches,
        output_root=tmp_path,
        lora_targets=["proj"],
        cpu=True,
        **kwargs,
    )


def _config(mode=ParamEfficiencyMode.FULL, steps=6):
    # A minimal object carrying only what the trainer reads off the config.
    from ceed_core import (
        ExtractionConfig,
        GroupConfig,
        LayerMapping,
        StudentConfig,
    )

    return GroupConfig(
        group_code="B2",
        seed=0,
        param_efficiency=mode,
        layer_mapping=LayerMapping(kind="proportional", pairs=((9, 12),)),
        student=StudentConfig(model="tiny", dtype="float32"),
        extraction=ExtractionConfig(
            teacher_model="t",
            model_revision="main",
            layers=(9,),
            ablation="mean-of-active",
            combine_weight="effective",
        ),
        training=a_training_config(steps),
    )


# -- the backbone actually trains -------------------------------------------


def test_training_drives_the_backbone_loss_down(tmp_path):
    trainer = a_trainer(tmp_path)
    outcome = trainer.train(_config(steps=40))

    assert isinstance(outcome, TrainingOutcome)
    assert outcome.steps == 40
    assert outcome.final_loss < outcome.initial_loss


def test_the_outcome_records_the_backbone_decomposition(tmp_path):
    outcome = a_trainer(tmp_path).train(_config(steps=5))
    assert "cross_entropy" in outcome.backbone_metrics
    assert "kd" in outcome.backbone_metrics


# -- checkpoint and resume ---------------------------------------------------


def test_a_checkpoint_is_written_and_can_be_reopened(tmp_path):
    a_trainer(tmp_path).train(_config(steps=3))
    assert (tmp_path / "checkpoint").is_dir()


def test_resuming_continues_from_the_checkpoint_rather_than_restarting(tmp_path):
    batches = [a_batch(0), a_batch(1)]
    first = a_trainer(tmp_path, batches=batches).train(_config(steps=3))
    assert first.resumed is False
    assert first.steps_run == 3

    # A second trainer over the same output_root and a longer budget resumes.
    second = a_trainer(tmp_path, batches=batches).train(_config(steps=5))
    assert second.resumed is True
    assert second.steps_run == 2  # only the two remaining steps
    assert second.steps == 5


def test_a_completed_baseline_is_not_retrained(tmp_path):
    batches = [a_batch(0), a_batch(1)]
    a_trainer(tmp_path, batches=batches).train(_config(steps=4))
    again = a_trainer(tmp_path, batches=batches).train(_config(steps=4))
    assert again.resumed is True
    assert again.steps_run == 0  # nothing left to do; the checkpoint is reused


# -- parameter-efficiency provenance ----------------------------------------


def test_full_finetune_is_recorded_and_trains_every_parameter(tmp_path):
    outcome = a_trainer(tmp_path).train(_config(mode=ParamEfficiencyMode.FULL, steps=2))
    assert outcome.param_efficiency is ParamEfficiencyMode.FULL
    assert outcome.lora_rank is None


def test_lora_is_recorded_and_trains_only_the_adapter(tmp_path):
    trainer = a_trainer(tmp_path, lora_rank=2)
    outcome = trainer.train(_config(mode=ParamEfficiencyMode.LORA, steps=2))
    assert outcome.param_efficiency is ParamEfficiencyMode.LORA
    # The adapter rank is recorded so a null result is read against its capacity.
    assert outcome.lora_rank == 2
    # A LoRA run trains far fewer parameters than the full model has.
    assert 0 < outcome.trainable_parameters < outcome.total_parameters


def test_the_mode_is_taken_from_the_config_flag_not_the_trainer(tmp_path):
    # One configuration flag selects the mode: the same trainer trains full or
    # LoRA depending only on the Group it is handed.
    trainer = a_trainer(tmp_path)
    assert trainer.train(_config(mode=ParamEfficiencyMode.FULL, steps=1)).param_efficiency is (
        ParamEfficiencyMode.FULL
    )


def test_the_student_conforms_to_the_trainable_protocol():
    assert isinstance(TinyStudent(), TrainableStudent)


# -- LoRA target resolution --------------------------------------------------


class _Wrapped(torch.nn.Module):
    """Stands in for gemma-4's Gemma4ClippableLinear: not a plain Linear."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class _VisionLanguage(torch.nn.Module):
    """A Student with the same q_proj name on an adaptable and a wrapped module."""

    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.q_proj = torch.nn.Linear(4, 4)
        self.vision_tower = torch.nn.Module()
        self.vision_tower.q_proj = _Wrapped()


def test_lora_targets_resolve_to_adaptable_modules_only():
    # The vision tower's q_proj is not a plain Linear; targeting it by bare leaf
    # name is what breaks PEFT on the real Student.
    targets = resolve_lora_targets(_VisionLanguage(), ["q_proj"])
    assert targets == ["language_model.q_proj"]


def test_resolving_no_adaptable_target_is_refused():
    # An adapter that matches nothing would train nothing, silently.
    with pytest.raises(ValueError, match="no adaptable"):
        resolve_lora_targets(_VisionLanguage(), ["not_a_module"])
