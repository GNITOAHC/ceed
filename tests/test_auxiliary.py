"""The auxiliary-signal contract, the warm-up schedule, and coupling gating.

B1 and B2 attach zero auxiliary signals, so none of this fires during their
training — but the contract every later Group plugs into is built and tested
here. A signal declares the artefact kinds it needs (so a Group whose store
lacks them fails immediately, not at hour six) and the forward views it needs;
auxiliary terms are held out during a backbone-only window and then ramped; and
each term is gated per token by the teacher's coupling strength.
"""

import pytest
import torch
from ceed_student.backbone import BackboneLoss

from ceed_core import StoreMetadata, VectorSpec
from ceed_student import (
    AuxTerm,
    ForwardView,
    WarmupSchedule,
    coupling_gate,
    required_artefact_kinds,
    required_forward_views,
    training_loss,
    verify_store_supports,
)


class _FakeSignal:
    """A stand-in auxiliary signal declaring its requirements without a loss."""

    def __init__(self, name, kinds, views, weight=1.0):
        self.name = name
        self.weight = weight
        self._kinds = frozenset(kinds)
        self._views = frozenset(views)

    def required_kinds(self):
        return self._kinds

    def required_views(self):
        return self._views


# -- requirement aggregation -------------------------------------------------


def test_required_kinds_is_the_union_across_signals():
    signals = [
        _FakeSignal("a", {"combine_weights"}, {ForwardView.ORIGINAL}),
        _FakeSignal("b", {"hidden_states", "combine_weights"}, {ForwardView.ORIGINAL}),
    ]
    assert required_artefact_kinds(signals) == frozenset({"combine_weights", "hidden_states"})


def test_no_signals_require_no_kinds():
    assert required_artefact_kinds([]) == frozenset()


def test_original_view_is_always_required_even_with_no_signals():
    # The backbone always runs the original forward; a Group with no auxiliary
    # signals still needs it.
    assert required_forward_views([]) == frozenset({ForwardView.ORIGINAL})


def test_required_views_unions_signal_views_with_the_original():
    signals = [
        _FakeSignal(
            "sel",
            set(),
            {ForwardView.RELEVANT_INTERVENED, ForwardView.CONTROL_INTERVENED},
        ),
    ]
    assert required_forward_views(signals) == frozenset(
        {
            ForwardView.ORIGINAL,
            ForwardView.RELEVANT_INTERVENED,
            ForwardView.CONTROL_INTERVENED,
        }
    )


# -- fail fast when the store cannot supply a required kind ------------------


def _metadata(*kinds):
    return StoreMetadata(
        extraction_fingerprint="fp",
        vector_kinds={k: VectorSpec(dtype="float16", shape=(2,)) for k in kinds},
    )


def test_a_signal_whose_kind_is_absent_fails_fast():
    signal = _FakeSignal("needs-attr", {"attribution"}, {ForwardView.ORIGINAL})
    with pytest.raises(Exception, match="attribution"):
        verify_store_supports(_metadata("combine_weights"), [signal])


def test_verify_passes_when_every_required_kind_is_present():
    signal = _FakeSignal("ok", {"combine_weights"}, {ForwardView.ORIGINAL})
    verify_store_supports(_metadata("combine_weights", "hidden_states"), [signal])


def test_no_signals_need_nothing_from_the_store():
    verify_store_supports(_metadata(), [])


# -- warm-up schedule --------------------------------------------------------


def test_auxiliary_scale_is_zero_during_the_backbone_only_window():
    schedule = WarmupSchedule(backbone_only_steps=3, ramp_steps=4)
    assert schedule.scale(0) == 0.0
    assert schedule.scale(2) == 0.0


def test_auxiliary_scale_ramps_linearly_then_saturates_at_one():
    schedule = WarmupSchedule(backbone_only_steps=2, ramp_steps=4)
    assert schedule.scale(2) == 0.0
    assert schedule.scale(3) == pytest.approx(0.25)
    assert schedule.scale(4) == pytest.approx(0.5)
    assert schedule.scale(6) == pytest.approx(1.0)
    assert schedule.scale(100) == 1.0


def test_a_zero_ramp_switches_auxiliary_on_at_full_scale():
    schedule = WarmupSchedule(backbone_only_steps=5, ramp_steps=0)
    assert schedule.scale(4) == 0.0
    assert schedule.scale(5) == 1.0


def test_the_scale_never_decreases():
    schedule = WarmupSchedule(backbone_only_steps=1, ramp_steps=3)
    values = [schedule.scale(s) for s in range(8)]
    assert values == sorted(values)


# -- coupling gating ---------------------------------------------------------


def test_gating_zeroes_tokens_below_the_coupling_threshold():
    token_loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
    coupling = torch.tensor([0.9, 0.1, 0.7, 0.05])
    gated = coupling_gate(token_loss, coupling, threshold=0.5)
    assert torch.equal(gated, torch.tensor([1.0, 0.0, 3.0, 0.0]))


def test_a_zero_threshold_selects_every_token_with_positive_coupling():
    token_loss = torch.tensor([1.0, 1.0, 1.0])
    coupling = torch.tensor([0.0, 0.5, 1.0])
    gated = coupling_gate(token_loss, coupling, threshold=0.0)
    # coupling of exactly zero is not evidence of coupling, so it is dropped.
    assert torch.equal(gated, torch.tensor([0.0, 1.0, 1.0]))


# -- assembling the step loss ------------------------------------------------


def _backbone(value: float) -> BackboneLoss:
    t = torch.tensor(value)
    return BackboneLoss(total=t, cross_entropy=t, kd=torch.tensor(0.0))


def test_with_no_auxiliary_terms_the_step_loss_is_the_backbone():
    schedule = WarmupSchedule(backbone_only_steps=0, ramp_steps=0)
    loss = training_loss(_backbone(1.5), [], step=0, schedule=schedule)
    assert torch.equal(loss, torch.tensor(1.5))


def test_auxiliary_terms_are_scaled_by_the_warmup_and_their_weight():
    schedule = WarmupSchedule(backbone_only_steps=0, ramp_steps=2)
    terms = [AuxTerm(name="s", weight=2.0, value=torch.tensor(3.0))]
    # step 1 -> scale 0.5, so contribution is 0.5 * 2.0 * 3.0 = 3.0 on top of 1.0.
    loss = training_loss(_backbone(1.0), terms, step=1, schedule=schedule)
    assert loss == pytest.approx(4.0)


def test_auxiliary_terms_contribute_nothing_during_the_backbone_only_window():
    schedule = WarmupSchedule(backbone_only_steps=3, ramp_steps=2)
    terms = [AuxTerm(name="s", weight=5.0, value=torch.tensor(9.0))]
    loss = training_loss(_backbone(1.0), terms, step=0, schedule=schedule)
    assert loss == pytest.approx(1.0)
