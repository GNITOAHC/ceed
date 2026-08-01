"""The Student training loop: the shared backbone, checkpointed and resumable.

The loop is built on ``accelerate`` so a multi-day Group survives preemption: it
``save_state``s the model, optimiser and RNG on a cadence and, on restart,
``load_state``s them and continues from the step it reached rather than from
zero. A baseline that has already reached its step budget is therefore never
retrained — the frozen checkpoint is reused.

Parameter efficiency is one flag. ``full`` optimises every Student parameter;
``lora`` wraps the Student with a PEFT adapter and optimises only that. The mode
is carried into the :class:`TrainingOutcome` so the run record can state it
unambiguously — a LoRA number must never be mistaken for a full fine-tune
(ADR-0005).

The objective is the shared backbone alone (:func:`ceed_student.backbone`),
assembled through :func:`ceed_student.auxiliary.training_loss` with an empty set
of auxiliary terms and the Group's warm-up schedule. B1 and B2 attach no
auxiliary signals; the schedule is wired and honoured so that the first Group to
add a signal changes only what fills that empty list, never this loop.

The heavy dependencies — ``accelerate`` and ``peft`` — and the real Student are
injected or imported lazily, so importing this module and running the loop
against a tiny CPU Student stays cheap and needs no GPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from ceed_core import GroupConfig, ParamEfficiencyMode
from ceed_student.auxiliary import (
    AuxiliarySignal,
    ForwardView,
    WarmupSchedule,
    aux_terms,
    required_forward_views,
    required_student_layers,
    training_loss,
)
from ceed_student.backbone import backbone_loss

CHECKPOINT_DIRNAME = "checkpoint"
PROGRESS_FILENAME = "progress.json"
PROBES_DIRNAME = "probes"


@dataclass
class TrainingBatch:
    """One step's inputs: the Student's inputs and the teacher's cached targets.

    Attributes:
        student_inputs: The model-specific tensors the Student is run on (e.g.
            ``input_ids`` and pixel values), moved to the device by the trainer.
        answer_token_positions: The output positions that are answer tokens —
            the positions the backbone is scored at.
        gold_token_ids: The gold token id at each answer position.
        teacher_topk_ids: The teacher's top-k logit ids at each answer position.
        teacher_topk_values: The teacher's top-k logit values at those ids.
        coupling_strength: The teacher-measured coupling strength per answer
            token, used to gate auxiliary losses; ``None`` when no signal needs
            it (B1, B2).
        artefacts: Any further per-token artefacts an auxiliary signal reads;
            empty for the backbone-only Groups.
    """

    student_inputs: dict[str, Tensor]
    answer_token_positions: Tensor
    gold_token_ids: Tensor
    teacher_topk_ids: Tensor
    teacher_topk_values: Tensor
    coupling_strength: Tensor | None = None
    artefacts: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class StudentForward:
    """What one forward view of one batch yields, scoped to the answer tokens.

    Everything a Group's objective reads comes from here, and it is all sliced to
    the answer tokens before it leaves the Student, because every cached teacher
    artefact is (ADR-0001) — so a signal never has to know where the answer sat
    in the sequence.

    Attributes:
        logits: The Student's next-token logits at the answer positions,
            ``[answer_tokens, vocab]``.
        hidden_states: The residual-stream state at each *requested* student
            layer, ``[answer_tokens, hidden]``. Only the layers the Group's
            signals asked for are present; a backbone-only Group gets none, and
            the Student need not compute them.
    """

    logits: Tensor
    hidden_states: Mapping[int, Tensor] = field(default_factory=dict)


@runtime_checkable
class TrainableStudent(Protocol):
    """A Student the loop can train: a torch module that answers per forward view.

    The concrete Student is a ``torch.nn.Module`` (so it has parameters the
    optimiser and ``accelerate`` handle); this protocol adds the one method the
    loop calls — running the Student under a forward view and returning what the
    objective reads at the answer tokens.
    """

    def answer_forward(
        self, batch: TrainingBatch, view: ForwardView, hidden_layers: frozenset[int]
    ) -> StudentForward:
        """Run ``batch`` under ``view``, returning answer-token logits and hiddens.

        Args:
            batch: The step's batch.
            view: The forward view to run under.
            hidden_layers: The student layers whose hidden states the Group's
                signals need; empty for a backbone-only Group.
        """
        ...


@dataclass(frozen=True)
class SignalContext:
    """Everything an auxiliary signal is given for one step.

    Attributes:
        batch: The step's batch, carrying the teacher's cached artefacts.
        forwards: The Student's forward per view. A signal reads only views it
            declared, so the trainer never runs a forward no Group asked for.
    """

    batch: TrainingBatch
    forwards: Mapping[ForwardView, StudentForward]

    @property
    def original(self) -> StudentForward:
        """The original (un-intervened) forward, which every Group runs."""
        return self.forwards[ForwardView.ORIGINAL]

    def artefact(self, kind: str) -> Tensor:
        """Return a cached teacher artefact for this batch's answer tokens.

        Args:
            kind: The artefact kind, as declared by the signal's
                ``required_kinds``.

        Returns:
            The artefact, ``[answer_tokens, ...]``.

        Raises:
            KeyError: If the batch does not carry it — which means the
                dataloader was not asked for a kind some signal requires, and is
                a wiring bug rather than a data problem.
        """
        if kind not in self.batch.artefacts:
            raise KeyError(
                f"this batch carries no {kind!r} artefact; it holds {sorted(self.batch.artefacts)}"
            )
        return self.batch.artefacts[kind]


@dataclass(frozen=True)
class TrainingOutcome:
    """What a training run reports back.

    Attributes:
        param_efficiency: The mode the run actually trained under, recorded so a
            LoRA number can never reach the Phase 1 table as a full fine-tune.
        lora_rank: The LoRA adapter rank when the run was LoRA, else ``None`` —
            so a null result is read against the adapter capacity behind it
            (ADR-0005).
        steps: The step budget targeted (the config's ``steps``).
        steps_run: How many steps this invocation actually ran — zero when a
            completed checkpoint was reused.
        resumed: Whether the run continued from an existing checkpoint.
        initial_loss: The backbone loss at this invocation's first step (or the
            restored final loss if nothing ran).
        final_loss: The backbone loss at the last step trained.
        backbone_metrics: The final cross-entropy and distillation components.
        checkpoint_dir: Where the (frozen, reusable) checkpoint was written.
        total_parameters: The Student's total parameter count.
        trainable_parameters: How many parameters the optimiser updated,
            excluding probes — a probe is thrown away, so counting it here would
            overstate what was trained into the deployed Student.
        signal_names: The auxiliary signals that actually trained, in order. The
            single independent variable of the whole comparison, so it is
            recorded rather than inferred from the Group code.
        probe_parameters: How many probe parameters were optimised and then
            discarded; zero for a Group with no probing signal.
        auxiliary_metrics: Each signal's final scalar contribution, keyed by
            signal name.
    """

    param_efficiency: ParamEfficiencyMode
    lora_rank: int | None
    steps: int
    steps_run: int
    resumed: bool
    initial_loss: float
    final_loss: float
    backbone_metrics: dict[str, float]
    checkpoint_dir: str
    total_parameters: int
    trainable_parameters: int
    signal_names: tuple[str, ...] = ()
    probe_parameters: int = 0
    auxiliary_metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Trainer(Protocol):
    """Something that can train a resolved Group and report the outcome.

    Injected into :func:`ceed_student.run.run_group` exactly as the evaluator
    is, so the spine stays free of ``accelerate``, ``peft`` and the real Student
    at import time and remains exercisable on CPU.
    """

    def train(self, config: GroupConfig) -> TrainingOutcome:
        """Train ``config``'s Student and return the outcome."""
        ...


def _move_batch(batch: TrainingBatch, device: Any) -> TrainingBatch:
    """Return ``batch`` with every tensor moved to ``device``."""
    return replace(
        batch,
        student_inputs={k: v.to(device) for k, v in batch.student_inputs.items()},
        answer_token_positions=batch.answer_token_positions.to(device),
        gold_token_ids=batch.gold_token_ids.to(device),
        teacher_topk_ids=batch.teacher_topk_ids.to(device),
        teacher_topk_values=batch.teacher_topk_values.to(device),
        coupling_strength=None
        if batch.coupling_strength is None
        else batch.coupling_strength.to(device),
        artefacts={k: v.to(device) for k, v in batch.artefacts.items()},
    )


def resolve_lora_targets(model: torch.nn.Module, leaf_names: Sequence[str]) -> list[str]:
    """Return the full paths of adaptable modules matching ``leaf_names``.

    PEFT matches target modules by name *suffix*, which is ambiguous on a
    vision-language model: gemma-4 has a ``q_proj`` in the language model (a
    plain ``nn.Linear``, adaptable) and another in the vision tower (wrapped in
    ``Gemma4ClippableLinear``, which PEFT cannot adapt). Passing the bare leaf
    name therefore matches both and fails.

    Resolving to full paths of the modules PEFT actually supports keeps the
    adapter on the language path — which is also the right scope for CEED, since
    the distillation target is the reasoning path and a frozen vision tower is
    the standard VLM fine-tuning choice.

    Args:
        model: The Student to scan.
        leaf_names: The module leaf names to adapt, e.g. ``q_proj``.

    Returns:
        The full module paths to hand PEFT, in model order.

    Raises:
        ValueError: If no adaptable module matches, which would otherwise
            produce an adapter that trains nothing.
    """
    wanted = set(leaf_names)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in wanted
    ]
    if not targets:
        raise ValueError(
            f"no adaptable nn.Linear modules named {sorted(wanted)} in this Student; "
            "LoRA would train nothing"
        )
    return targets


def _apply_param_efficiency(
    student: torch.nn.Module,
    mode: ParamEfficiencyMode,
    lora_targets: Sequence[str],
    lora_rank: int,
    lora_alpha: int,
) -> torch.nn.Module:
    """Return the module to optimise for ``mode``.

    ``full`` returns the Student unchanged (every parameter trains). ``lora``
    wraps it in a PEFT LoRA adapter over the adaptable modules matching
    ``lora_targets`` and returns the wrapper (only the adapter trains).
    """
    if mode is ParamEfficiencyMode.FULL:
        return student
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=resolve_lora_targets(student, lora_targets),
        lora_dropout=0.0,
    )
    return get_peft_model(student, config)  # type: ignore[arg-type]


class AccelerateTrainer:
    """Trains a Student on the shared backbone with checkpointing and resume."""

    def __init__(
        self,
        build_student: Callable[[GroupConfig], TrainableStudent],
        build_batches: Callable[[GroupConfig], Sequence[TrainingBatch]],
        output_root: Path,
        build_signals: Callable[[GroupConfig], Sequence[AuxiliarySignal]] | None = None,
        lora_targets: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_rank: int = 4,
        lora_alpha: int = 8,
        cpu: bool = False,
    ) -> None:
        """Configure the trainer.

        The parameter-efficiency mode is *not* a trainer setting: it is read from
        ``config.param_efficiency`` at :meth:`train` time, so the single
        configuration flag selects it and the trainer cannot silently disagree
        with the Group it is running (ADR-0005). The same is true of the
        auxiliary signals: the factory is handed the Group and returns what that
        Group declares, so the loop has no per-Group branch.

        Args:
            build_student: Builds the Student for a Group. Injected so the real
                8B Student never reaches this module at import time.
            build_batches: Builds the Group's training batches (Student inputs
                plus the teacher's cached targets read from the store).
            output_root: The directory the checkpoint is written under.
            build_signals: Builds the Group's auxiliary signals. Omitted for the
                backbone-only Groups, which declare none.
            lora_targets: The module names LoRA adapts when in LoRA mode.
            lora_rank: The LoRA adapter rank; recorded so a null result can be
                read against the adapter capacity that produced it (ADR-0005).
            lora_alpha: The LoRA scaling factor.
            cpu: Force CPU execution (used by the fast test tier).
        """
        self.build_student = build_student
        self.build_batches = build_batches
        self.build_signals = build_signals
        self.output_root = output_root
        self.lora_targets = lora_targets
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.cpu = cpu

    def _checkpoint_dir(self) -> Path:
        return self.output_root / CHECKPOINT_DIRNAME

    def _read_progress(self) -> dict[str, Any] | None:
        path = self._checkpoint_dir() / PROGRESS_FILENAME
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def _write_progress(self, completed: int, metrics: dict[str, float], params: dict[str, int]):
        checkpoint = self._checkpoint_dir()
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / PROGRESS_FILENAME).write_text(
            json.dumps({"completed_steps": completed, **metrics, **params})
        )

    def _save_adapter(
        self, accelerator: Any, model: Any, mode: ParamEfficiencyMode
    ) -> None:  # pragma: no cover - PEFT save path needs a real adapter
        """Write the trained LoRA adapter beside the checkpoint, if this was a LoRA run.

        ``accelerate``'s state is what resumption reads; the adapter directory is
        what *evaluation* reads, so a trained Group can be scored without
        reconstructing the optimiser. A full fine-tune needs neither — its
        weights are in the state itself.
        """
        if mode is not ParamEfficiencyMode.LORA:
            return
        unwrapped = accelerator.unwrap_model(model)
        save = getattr(unwrapped, "save_pretrained", None)
        if save is None:
            return
        save(str(self._checkpoint_dir() / "adapter"))

    def _save_probes(self, probes: torch.nn.Module) -> None:
        """Write the probes beside the checkpoint, kept out of the deployed Student.

        A probe is deletable by construction: it is a parameter of the auxiliary
        signal, never of the Student, so nothing has to be stripped from the
        Student to remove it. Saving them separately keeps them available for
        analysis while making it structurally impossible for one to reach the
        adapter a served model is loaded with.
        """
        if not list(probes.parameters()):
            return
        directory = self._checkpoint_dir() / PROBES_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(probes.state_dict(), directory / "probes.pt")

    def train(self, config: GroupConfig) -> TrainingOutcome:
        """Train the Group's Student on the backbone, resuming if a checkpoint exists.

        Args:
            config: A resolved Group whose ``training`` declares the budget and
                schedule.

        Returns:
            The training outcome, including the parameter-efficiency mode used
            and the frozen checkpoint's location.

        Raises:
            ValueError: If the Group declares no training.
        """
        if config.training is None:
            raise ValueError(f"Group {config.group_code} declares no training")
        training = config.training
        # One configuration flag selects the mode; the trainer reads it here
        # rather than carrying its own, so the recorded mode is the mode that
        # actually trained (ADR-0005).
        mode = config.param_efficiency

        from accelerate import Accelerator

        accelerator = Accelerator(cpu=self.cpu)

        student = self.build_student(config)
        total_parameters = sum(p.numel() for p in student.parameters())  # type: ignore[attr-defined]
        model = _apply_param_efficiency(
            student,  # type: ignore[arg-type]
            mode,
            self.lora_targets,
            self.lora_rank,
            self.lora_alpha,
        )
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        signals = list(self.build_signals(config)) if self.build_signals is not None else []
        # Probes are optimised with the Student and thrown away with the signal,
        # so they are held apart from it: the Student's own module tree is never
        # touched and the deployed model is architecturally identical (story 47).
        probes = torch.nn.ModuleList([s for s in signals if isinstance(s, torch.nn.Module)])
        probe_parameters = sum(p.numel() for p in probes.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad]
            + [p for p in probes.parameters() if p.requires_grad],
            lr=training.learning_rate,
        )
        model, probes, optimizer = accelerator.prepare(model, probes, optimizer)

        batches = self.build_batches(config)
        schedule = WarmupSchedule(training.warmup.backbone_only_steps, training.warmup.ramp_steps)
        hidden_layers = required_student_layers(signals)
        views = sorted(required_forward_views(signals))
        params = {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        }

        prior = self._read_progress()
        completed = 0
        resumed = False
        restored = {"loss": 0.0, "cross_entropy": 0.0, "kd": 0.0}
        if prior is not None:
            accelerator.load_state(str(self._checkpoint_dir()))
            completed = int(prior["completed_steps"])
            resumed = completed > 0
            restored = {k: float(prior.get(k, 0.0)) for k in restored}

        initial_loss = restored["loss"]
        last = dict(restored)
        auxiliary: dict[str, float] = {}

        model.train()
        probes.train()
        for step in range(completed, training.steps):
            batch = _move_batch(batches[step % len(batches)], accelerator.device)
            forwards = {view: model.answer_forward(batch, view, hidden_layers) for view in views}
            context = SignalContext(batch=batch, forwards=forwards)
            backbone = backbone_loss(
                context.original.logits,
                batch.gold_token_ids,
                batch.teacher_topk_ids,
                batch.teacher_topk_values,
                kd_weight=training.backbone.kd_weight,
                temperature=training.backbone.kd_temperature,
            )
            terms = aux_terms(signals, context)
            loss = training_loss(backbone, terms, step=step, schedule=schedule)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            last = {
                "loss": float(loss.detach()),
                "cross_entropy": float(backbone.cross_entropy.detach()),
                "kd": float(backbone.kd.detach()),
            }
            auxiliary = {term.name: float(term.value.detach()) for term in terms}
            if step == completed:
                initial_loss = last["loss"]
            if training.checkpoint_every and (step + 1) % training.checkpoint_every == 0:
                accelerator.save_state(str(self._checkpoint_dir()))
                self._write_progress(step + 1, last, params)

        steps_run = training.steps - completed
        if steps_run > 0:
            accelerator.save_state(str(self._checkpoint_dir()))
            self._write_progress(training.steps, last, params)
            self._save_adapter(accelerator, model, mode)
            self._save_probes(accelerator.unwrap_model(probes))

        return TrainingOutcome(
            param_efficiency=mode,
            lora_rank=self.lora_rank if mode is ParamEfficiencyMode.LORA else None,
            steps=training.steps,
            steps_run=steps_run,
            resumed=resumed,
            initial_loss=initial_loss,
            final_loss=last["loss"],
            backbone_metrics={"cross_entropy": last["cross_entropy"], "kd": last["kd"]},
            checkpoint_dir=str(self._checkpoint_dir()),
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            signal_names=tuple(signal.name for signal in signals),
            probe_parameters=probe_parameters,
            auxiliary_metrics=auxiliary,
        )
