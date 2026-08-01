r"""The auxiliary signals that complete the baseline Group set: B3, B4, and B5.

Each is a component plus a YAML overlay and nothing else — no new training path,
no branch in the loop — which is the property the plan needs, because every
experimental Group is the same shape and there will be nine more of them.

Between them they exercise three quite different kinds of signal, which is why
these three complete the baselines rather than merely adding to them:

* **B3** reads a cached *teacher hidden state* and needs a layer mapping to know
  where in the Student to put it. It is the standard hidden-state KD comparator.
* **B4** reads a cached *scalar per token* and reweights the backbone's own
  distillation term. It reproduces VA-OPD (arXiv:2605.21924), the plan's
  critical external baseline; what is faithful and what deviates is recorded in
  docs/adr/0007-va-opd-reproduced-off-policy-on-gold-answers.md.
* **B5** reads a cached *per-expert vector* and trains a removable probe to
  predict it. It is the routing-signal control that E1 is measured against: B5
  distils what the router *says*, E1 distils what ablation *measures*, and the
  gap between them is the whole thesis.

**Probes are deletable.** B3's projections and B5's probes are parameters of the
signal, never of the Student. They are optimised alongside it and then thrown
away, so the deployed Student is architecturally identical to the base model and
the zero-added-inference-cost claim is literally true rather than approximately
so.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional

from ceed_core import ArtifactStore, GroupConfig, LayerMapping, SignalOptions, VectorSpec
from ceed_student.auxiliary import AuxiliarySignal, ForwardView, verify_store_supports
from ceed_student.backbone import topk_kd_per_token
from ceed_student.training import SignalContext

# The artefact kinds these signals read. Named here rather than imported from
# ceed_teacher so that training never depends on the teacher package.
HIDDEN_STATES = "hidden_states"
COMBINE_WEIGHTS = "combine_weights"
VISUAL_ADVANTAGE = "visual_advantage"

HIDDEN_STATE_PROJECTION = "hidden_state_projection"
VISUAL_ADVANTAGE_REWEIGHTING = "visual_advantage_reweighting"
COMBINE_WEIGHT_PROBE = "combine_weight_probe"


class _MappedSignal(torch.nn.Module):
    """Shared plumbing for a signal that reads a layer-stacked teacher artefact.

    Both B3 and B5 supervise the Student at the student layers a
    :class:`~ceed_core.LayerMapping` names, from a teacher artefact stacked over
    whatever layers the store happens to hold. Resolving a teacher layer to its
    index in that stack is done here, against the store's own metadata, so
    neither signal carries a convention about extraction order.
    """

    def __init__(self, name: str, weight: float, mapping: LayerMapping, spec: VectorSpec) -> None:
        super().__init__()
        self.name = name
        self.weight = weight
        self.mapping = mapping
        # Fails here, at Group construction, if the store lacks a mapped layer —
        # not at hour six of training.
        self._layer_index = {
            teacher: spec.layer_index(teacher) for teacher in mapping.teacher_layers
        }

    def required_views(self) -> frozenset[ForwardView]:
        """These signals read the original forward only."""
        return frozenset({ForwardView.ORIGINAL})

    def required_student_layers(self) -> frozenset[int]:
        """The student layers the mapping places supervision at."""
        return frozenset(self.mapping.student_layers)

    def _student_hidden(
        self, context: SignalContext, teacher_layer: int, head: torch.nn.Module
    ) -> Tensor:
        """Return the Student's state at the mapped layer, in the head's precision.

        The Student runs in fp16 (Volta has no bf16, ADR-0006) but a probe is
        small enough to keep in fp32, so its input is widened rather than the
        probe narrowed. That is deliberate: the auxiliary loss and the gradients
        flowing back through it are then computed at full precision, where a
        half-precision head would be optimising against its own rounding.
        """
        hidden = context.original.hidden_states[self.mapping.student_layer_for(teacher_layer)]
        return hidden.to(next(head.parameters()).dtype)


class HiddenStateProjectionSignal(_MappedSignal):
    """B3: match the teacher's hidden state through a learned projection.

    For each mapped layer a linear projection carries the Student's residual
    state into the teacher's width and is matched to the teacher's cached state
    by mean squared error. The projection is necessary because the two models
    are different widths (2560 against 2816) and there is no reason their
    residual bases would align even if they were not.

    **Answer tokens only** (plan amendment A5). Textbook hidden-state KD
    supervises the whole sequence; that would be ~270 GB of cache, and — the
    reason that matters more — it would give B3 roughly fifty times the
    supervised positions of E1, so a B3-versus-E1 gap would measure volume
    rather than signal. The deviation from published hidden-KD results is real
    and is the price of the comparison being honest.
    """

    def __init__(
        self,
        weight: float,
        mapping: LayerMapping,
        spec: VectorSpec,
        student_hidden_size: int,
    ) -> None:
        """Build one projection per mapped layer.

        Args:
            weight: The signal's scalar loss weight.
            mapping: The Teacher-to-Student layer correspondence.
            spec: The store's spec for the hidden-state artefact, giving both
                the teacher width and which layers were cached.
            student_hidden_size: The Student's residual width.
        """
        super().__init__(HIDDEN_STATE_PROJECTION, weight, mapping, spec)
        teacher_hidden_size = spec.shape[-1]
        self.projections = torch.nn.ModuleDict(
            {
                str(teacher): torch.nn.Linear(student_hidden_size, teacher_hidden_size)
                for teacher in mapping.teacher_layers
            }
        )

    def required_kinds(self) -> frozenset[str]:
        """B3 reads the teacher's cached hidden states."""
        return frozenset({HIDDEN_STATES})

    def token_loss(self, context: SignalContext) -> Float[Tensor, " tokens"]:
        """Return the mean squared projection error at each answer token."""
        teacher = context.artefact(HIDDEN_STATES)  # [tokens, cached_layers, hidden]
        losses = []
        for teacher_layer in self.mapping.teacher_layers:
            projection = self.projections[str(teacher_layer)]
            predicted = projection(self._student_hidden(context, teacher_layer, projection))
            target = teacher[:, self._layer_index[teacher_layer], :].to(predicted.dtype)
            losses.append((predicted - target).pow(2).mean(dim=-1))
        return torch.stack(losses).mean(dim=0)


class CombineWeightProbeSignal(_MappedSignal):
    """B5: a removable probe predicts the teacher's effective combine weights.

    The Student is dense and has no router, so there is nothing in it to match a
    combine weight against directly. Instead a probe reads the Student's
    residual state at the mapped layer and predicts the teacher's distribution
    over experts; the Student is trained through the probe, so what the probe
    needs must be represented in the residual stream.

    The target is the **effective** combine weight — the softmax top-k weight
    times the per-expert scale, i.e. the number actually multiplying an expert's
    output (plan amendment A3), which is also the strongest form of the null
    hypothesis Phase 0.1 tests. Most of its 128 entries are exactly zero (only
    the top 8 route), so the loss is a cross-entropy against the normalised
    weight rather than a KL: ``-sum(p * log q)`` is defined where ``p`` is zero
    and ``p * log p`` is not. The two differ by the teacher's entropy, which is
    constant with respect to the Student.

    This is the *routing* comparator. It distils what the router asserts, which
    is exactly the quantity CEED's premise says fails to predict what ablation
    measures — so E1 minus B5 is the plan's internal test of that premise, and
    if the premise is wrong B5 is the fallback design.
    """

    def __init__(
        self,
        weight: float,
        mapping: LayerMapping,
        spec: VectorSpec,
        student_hidden_size: int,
    ) -> None:
        """Build one probe per mapped layer.

        Args:
            weight: The signal's scalar loss weight.
            mapping: The Teacher-to-Student layer correspondence.
            spec: The store's spec for the combine-weight artefact, giving both
                the expert count and which layers were cached.
            student_hidden_size: The Student's residual width.
        """
        super().__init__(COMBINE_WEIGHT_PROBE, weight, mapping, spec)
        n_experts = spec.shape[-1]
        self.probes = torch.nn.ModuleDict(
            {
                str(teacher): torch.nn.Linear(student_hidden_size, n_experts)
                for teacher in mapping.teacher_layers
            }
        )

    def required_kinds(self) -> frozenset[str]:
        """B5 reads the teacher's cached effective combine weights."""
        return frozenset({COMBINE_WEIGHTS})

    def token_loss(self, context: SignalContext) -> Float[Tensor, " tokens"]:
        """Return the probe's cross-entropy against the router at each answer token."""
        teacher = context.artefact(COMBINE_WEIGHTS)  # [tokens, cached_layers, experts]
        losses = []
        for teacher_layer in self.mapping.teacher_layers:
            probe = self.probes[str(teacher_layer)]
            log_q = functional.log_softmax(
                probe(self._student_hidden(context, teacher_layer, probe)), -1
            )
            weights = teacher[:, self._layer_index[teacher_layer], :].to(log_q.dtype)
            # A token whose weights are all zero carries no routing to distil;
            # normalising it would divide by zero, so it contributes nothing.
            total = weights.sum(dim=-1, keepdim=True)
            safe = total.clamp(min=torch.finfo(log_q.dtype).tiny)
            target = torch.where(total > 0, weights / safe, weights)
            losses.append(-(target * log_q).sum(dim=-1))
        return torch.stack(losses).mean(dim=0)


def va_group_weights(
    visual_advantage: Float[Tensor, " tokens"],
    top_fraction: float = 0.2,
    high_weight: float = 0.5,
) -> Float[Tensor, " tokens"]:
    r"""Return VA-OPD's per-token loss weights: high-advantage tokens up-weighted.

    The paper splits a rollout's tokens by *rank* — the top ``p_v`` fraction by
    visual advantage form the high-VA group :math:`V`, the rest the low-VA group
    :math:`L` — and weights the two groups' mean losses by :math:`\lambda` and
    :math:`1 - \lambda`:

    .. math:: L^{(k)}_{group} = \lambda \frac{1}{|V|} \sum_{t \in V} KL_t
              + (1 - \lambda) \frac{1}{|L|} \sum_{t \in L} KL_t

    The weights returned here are scaled so that their *mean* against the
    per-token losses reproduces that expression exactly, which is what lets the
    signal return a per-token loss like every other signal instead of a
    pre-reduced scalar. Their mean is one, so the reweighting redistributes the
    loss across tokens without changing its scale.

    Two cases the paper does not meet, because its rollouts are long and CEED's
    gold answers are often a single token:

    * fewer tokens than the split needs — the high group always takes at least
      one token, and when it takes all of them there is no low group, so the
      weights are uniform;
    * **no visual advantage anywhere** — if the teacher never benefited from the
      detail, ranking by advantage ranks noise, so the weights are uniform. This
      is the honest reading of a zero signal, and it makes B4 fall back to B2 on
      exactly the examples where VA-OPD's premise does not hold.

    Args:
        visual_advantage: Each answer token's cached visual advantage.
        top_fraction: The share of tokens forming the high-VA group (``p_v``).
        high_weight: The share of the loss that group carries (``lambda``).

    Returns:
        One non-negative weight per answer token, averaging one.
    """
    n_tokens = int(visual_advantage.shape[0])
    weights = torch.ones_like(visual_advantage)
    if n_tokens == 0 or float(visual_advantage.max()) <= 0.0:
        return weights

    n_high = min(n_tokens, max(1, int(n_tokens * top_fraction)))
    n_low = n_tokens - n_high
    if n_low == 0:
        return weights

    high = torch.topk(visual_advantage, n_high).indices
    weights = weights * ((1.0 - high_weight) * n_tokens / n_low)
    weights[high] = high_weight * n_tokens / n_high
    return weights


class VisualAdvantageSignal(torch.nn.Module):
    """B4: VA-OPD's reweighting of the distillation loss towards visual tokens.

    The teacher scored every gold answer token twice, once seeing the image and
    once seeing a degraded copy of it; the difference — the visual advantage —
    is cached per token. This signal ranks the answer tokens by that advantage
    and redistributes the distillation loss towards the top of the ranking,
    which is the paper's claim: a small minority of tokens carries the visual
    supervision, and averaging over all of them dilutes it away.

    The divergence being reweighted is the backbone's own
    (:func:`~ceed_student.backbone.topk_kd_per_token`), not a new one. That is
    deliberate: B4 must differ from B2 in the weighting and in nothing else, or
    the comparison measures two changes at once.

    The signal has no parameters — it is pure reweighting — so unlike B3 and B5
    it adds nothing to the optimiser.
    """

    def __init__(self, weight: float, options: SignalOptions, temperature: float) -> None:
        """Configure the reweighting.

        Args:
            weight: The signal's scalar loss weight.
            options: The VA-OPD constants (``p_v`` and ``lambda``).
            temperature: The distillation temperature, taken from the Group's
                backbone so the reweighted divergence is the same one the
                backbone computes.
        """
        super().__init__()
        self.name = VISUAL_ADVANTAGE_REWEIGHTING
        self.weight = weight
        self.top_fraction = options.va_top_fraction
        self.high_weight = options.va_high_weight
        self.temperature = temperature

    def required_kinds(self) -> frozenset[str]:
        """B4 reads the teacher's cached per-token visual advantage."""
        return frozenset({VISUAL_ADVANTAGE})

    def required_views(self) -> frozenset[ForwardView]:
        """The degradation happened on the teacher's side; the Student sees the original."""
        return frozenset({ForwardView.ORIGINAL})

    def required_student_layers(self) -> frozenset[int]:
        """B4 reads no hidden states."""
        return frozenset()

    def token_loss(self, context: SignalContext) -> Float[Tensor, " tokens"]:
        """Return the distillation divergence, reweighted by visual advantage."""
        batch = context.batch
        advantage = context.artefact(VISUAL_ADVANTAGE).reshape(-1).float()
        divergence = topk_kd_per_token(
            context.original.logits,
            batch.teacher_topk_ids,
            batch.teacher_topk_values,
            self.temperature,
        )
        weights = va_group_weights(advantage, self.top_fraction, self.high_weight)
        return weights.to(divergence.dtype) * divergence


def signal_artefact_kinds(config: GroupConfig) -> list[str]:
    """Return the artefact kinds a Group's declared signals read, without building them.

    The dataloader needs this before the Student is loaded — the batches it
    builds must already carry what the signals will ask for — and constructing
    the signals early would mean sizing probes against a Student that does not
    exist yet.

    Args:
        config: The resolved Group.

    Returns:
        The artefact kinds, sorted; empty for a backbone-only Group.

    Raises:
        ValueError: If the Group declares a signal that has no component.
    """
    kinds: set[str] = set()
    for declared in config.auxiliary_signals:
        kinds |= set(_kinds_for(declared.name, config.group_code))
    return sorted(kinds)


def build_signals(
    config: GroupConfig,
    store: ArtifactStore | None,
    student_hidden_size: int,
) -> list[AuxiliarySignal]:
    """Build a Group's auxiliary signals, refusing any the store cannot feed.

    This is the registry the Group overlay's signal names resolve through, and
    the one place a Group's declared supervision is checked against the artefacts
    that actually exist. Verification happens here, before a model is built or a
    step is run, so a Group whose signal was never extracted fails immediately
    with a message naming the missing kind (story 27).

    Args:
        config: The resolved Group.
        store: The artifact store the Group reads, or ``None`` for a Group that
            needs no artefacts.
        student_hidden_size: The Student's residual width, which B3's
            projections and B5's probes are sized against.

    Returns:
        The Group's signals, in declaration order; empty for B0, B1 and B2.

    Raises:
        ValueError: If a signal name is not one of the implemented components,
            or if a signal needs a store and none was supplied.
        MissingArtifactKindError: If the store lacks a required artefact kind.
    """
    if not config.auxiliary_signals:
        return []

    mapping = config.layer_mapping
    temperature = config.training.backbone.kd_temperature if config.training else 1.0

    def spec_for(kind: str) -> VectorSpec:
        if store is None:
            raise ValueError(
                f"Group {config.group_code} declares a signal reading {kind!r} but no "
                "artifact store was supplied; run teacher extraction first"
            )
        return store.metadata.vector_kinds[kind]

    signals: list[AuxiliarySignal] = []
    for declared in config.auxiliary_signals:
        if store is not None:
            store.metadata.require_kinds(_kinds_for(declared.name, config.group_code))
        if declared.name == HIDDEN_STATE_PROJECTION:
            signals.append(
                HiddenStateProjectionSignal(
                    declared.weight, mapping, spec_for(HIDDEN_STATES), student_hidden_size
                )
            )
        elif declared.name == COMBINE_WEIGHT_PROBE:
            signals.append(
                CombineWeightProbeSignal(
                    declared.weight, mapping, spec_for(COMBINE_WEIGHTS), student_hidden_size
                )
            )
        elif declared.name == VISUAL_ADVANTAGE_REWEIGHTING:
            spec_for(VISUAL_ADVANTAGE)  # present-or-fail, though its shape is not needed
            signals.append(VisualAdvantageSignal(declared.weight, declared.options, temperature))
        else:
            raise ValueError(
                f"Group {config.group_code} declares unknown auxiliary signal "
                f"{declared.name!r}; known signals are {sorted(SIGNAL_KINDS)}"
            )

    if store is not None:
        verify_store_supports(store.metadata, signals)
    return signals


# What each signal name needs from the store, so an unbuildable Group is refused
# before anything is constructed.
SIGNAL_KINDS: dict[str, frozenset[str]] = {
    HIDDEN_STATE_PROJECTION: frozenset({HIDDEN_STATES}),
    COMBINE_WEIGHT_PROBE: frozenset({COMBINE_WEIGHTS}),
    VISUAL_ADVANTAGE_REWEIGHTING: frozenset({VISUAL_ADVANTAGE}),
}


def _kinds_for(name: str, group_code: str) -> Sequence[str]:
    if name not in SIGNAL_KINDS:
        raise ValueError(
            f"Group {group_code} declares unknown auxiliary signal {name!r}; "
            f"known signals are {sorted(SIGNAL_KINDS)}"
        )
    return sorted(SIGNAL_KINDS[name])
