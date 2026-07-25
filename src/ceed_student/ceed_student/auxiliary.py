"""The auxiliary-signal contract that every Group above the backbone plugs into.

A Group is the shared backbone (:mod:`ceed_student.backbone`) plus zero or more
auxiliary signals. Each signal declares two things up front — the artefact kinds
it reads from the store, and the forward views it needs the Student run under —
and contributes a per-token loss. The trainer gates that loss by the teacher's
per-token coupling strength (supervision concentrates where the teacher shows
evidence-computation coupling) and scales it by a warm-up schedule that holds the
auxiliary terms out until the backbone has settled.

B1 and B2 attach no signals, so for them this module only supplies the
``ORIGINAL``-only forward-view requirement and a no-op step loss. The contract is
built in full here so that adding E1's Mode probing or E2's selectivity matching
is a new signal component and a YAML overlay, never a new training path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from ceed_core import StoreMetadata
from ceed_student.backbone import BackboneLoss


class ForwardView(StrEnum):
    """A view of one example the Student is run under within a training step.

    Selectivity signals need the Student's answer under a *relevant* and a
    *control* intervention alongside the original, all within one step; a
    backbone-only Group needs only the original. Naming the views as a closed
    set is what lets a signal declare which it requires.
    """

    ORIGINAL = "original"
    RELEVANT_INTERVENED = "relevant_intervened"
    CONTROL_INTERVENED = "control_intervened"


@runtime_checkable
class AuxiliarySignal(Protocol):
    """One supervision signal stacked on top of the backbone.

    A signal is swappable: it declares its store and forward-view requirements
    and produces a per-token loss, and the trainer does the rest (gating,
    scheduling, summation). Adding a Group is adding one of these plus a YAML
    overlay.
    """

    name: str
    weight: float

    def required_kinds(self) -> frozenset[str]:
        """Return the artefact kinds this signal reads from the store."""
        ...

    def required_views(self) -> frozenset[ForwardView]:
        """Return the forward views this signal needs the Student run under."""
        ...


@dataclass(frozen=True)
class AuxTerm:
    """One auxiliary signal's realised contribution for a step.

    Attributes:
        name: The signal that produced this term.
        weight: The signal's scalar weight.
        value: The signal's scalar loss for the step, already gated per token.
    """

    name: str
    weight: float
    value: Tensor


def required_artefact_kinds(signals: Sequence[AuxiliarySignal]) -> frozenset[str]:
    """Return every artefact kind required across ``signals`` (their union)."""
    kinds: frozenset[str] = frozenset()
    for signal in signals:
        kinds |= signal.required_kinds()
    return kinds


def required_forward_views(signals: Sequence[AuxiliarySignal]) -> frozenset[ForwardView]:
    """Return every forward view the step must run, always including the original.

    The backbone runs the original forward unconditionally, so it is present
    even when no signal asks for a view.
    """
    views: frozenset[ForwardView] = frozenset({ForwardView.ORIGINAL})
    for signal in signals:
        views |= signal.required_views()
    return views


def verify_store_supports(metadata: StoreMetadata, signals: Sequence[AuxiliarySignal]) -> None:
    """Fail fast unless the store supplies every artefact kind the signals need.

    Args:
        metadata: The artifact store's metadata, declaring the kinds it holds.
        signals: The Group's auxiliary signals.

    Raises:
        MissingArtifactKindError: If any required kind is absent, naming what is
            missing and what the store holds.
    """
    metadata.require_kinds(sorted(required_artefact_kinds(signals)))


@dataclass(frozen=True)
class WarmupSchedule:
    """The schedule that eases auxiliary terms in after the backbone settles.

    The auxiliary scale is zero for ``backbone_only_steps`` steps, then ramps
    linearly to one over ``ramp_steps`` further steps, then stays at one. A zero
    ramp switches the auxiliary terms on at full scale the moment the
    backbone-only window ends.

    Attributes:
        backbone_only_steps: Steps during which the auxiliary scale is zero.
        ramp_steps: Steps over which it ramps from zero to one.
    """

    backbone_only_steps: int
    ramp_steps: int

    def scale(self, step: int) -> float:
        """Return the auxiliary scale in ``[0, 1]`` for a zero-based step index."""
        elapsed = step - self.backbone_only_steps
        if elapsed < 0:
            return 0.0
        if self.ramp_steps == 0 or elapsed >= self.ramp_steps:
            return 1.0
        return elapsed / self.ramp_steps


def coupling_gate(
    token_loss: Float[Tensor, " tokens"],
    coupling_strength: Float[Tensor, " tokens"],
    threshold: float = 0.0,
) -> Float[Tensor, " tokens"]:
    """Zero the auxiliary loss on tokens whose coupling does not exceed ``threshold``.

    Supervision is concentrated where the teacher shows evidence-computation
    coupling: a token whose coupling strength is at or below the threshold
    contributes nothing, so a per-token loss is selected down to the tokens the
    teacher's measurement actually supports.

    Args:
        token_loss: The per-token auxiliary loss.
        coupling_strength: The teacher-measured coupling strength per token.
        threshold: The strength a token must exceed to contribute.

    Returns:
        The per-token loss with sub-threshold tokens zeroed.
    """
    keep = coupling_strength > threshold
    return token_loss * keep


def training_loss(
    backbone: BackboneLoss,
    aux_terms: Sequence[AuxTerm],
    step: int,
    schedule: WarmupSchedule,
) -> Tensor:
    """Assemble the scalar step loss from the backbone and scheduled auxiliaries.

    The backbone always contributes in full; each auxiliary term is scaled by
    its weight and the warm-up scale for the step. With no auxiliary terms the
    step loss is exactly the backbone total, so B1 and B2 optimise the backbone
    alone.

    Args:
        backbone: The backbone loss for the step.
        aux_terms: The realised auxiliary contributions for the step.
        step: The zero-based step index, for the warm-up scale.
        schedule: The auxiliary warm-up schedule.

    Returns:
        The scalar loss to optimise.
    """
    total = backbone.total
    scale = schedule.scale(step)
    for term in aux_terms:
        total = total + scale * term.weight * term.value
    return total
