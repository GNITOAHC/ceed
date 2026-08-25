"""The three auxiliary signals that complete the baselines: B3, B4, and B5.

These are the components whose bugs would produce plausible wrong science rather
than an exception — a hidden state read from the wrong layer of the store, a
reweighting that quietly does nothing, a probe trained against an unnormalised
target. So what is asserted here is the numbers, against the definitions they
come from: VA-OPD's published formula for B4, the store's own layer axis for B3
and B5.
"""

import pytest
import torch
from ceed_student.signals import (
    COMBINE_WEIGHTS,
    HIDDEN_STATES,
    VISUAL_ADVANTAGE,
    CombineWeightProbeSignal,
    HiddenStateProjectionSignal,
    VisualAdvantageSignal,
    build_signals,
    va_group_weights,
)
from ceed_student.training import SignalContext, StudentForward, TrainingBatch
from ceed_teacher.va_opd import degrade_image, visual_advantage

from ceed_core import (
    LayerMapping,
    MissingArtifactKindError,
    SignalOptions,
    VectorSpec,
)
from ceed_student import ForwardView

# The store caches more layers than any Group supervises (ADR-0001's
# over-caching), so a signal asking for teacher layer 19 must resolve it against
# the store rather than assume it is first.
CACHED_LAYERS = (4, 9, 14, 19, 24, 29)


def a_context(hidden=None, artefacts=None, logits=None, topk=None) -> SignalContext:
    """A one-example context with whatever the signal under test reads."""
    ids, values = topk or (torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2))
    batch = TrainingBatch(
        student_inputs={},
        answer_token_positions=torch.arange(ids.shape[0]),
        gold_token_ids=torch.zeros(ids.shape[0], dtype=torch.long),
        teacher_topk_ids=ids,
        teacher_topk_values=values,
        artefacts=artefacts or {},
    )
    forward = StudentForward(
        logits=logits if logits is not None else torch.zeros(ids.shape[0], 4),
        hidden_states=hidden or {},
    )
    return SignalContext(batch=batch, forwards={ForwardView.ORIGINAL: forward})


# -- B4: the VA-OPD reproduction ---------------------------------------------


def test_degradation_keeps_the_image_size_so_the_two_passes_align():
    """The teacher must emit the same number of image tokens for both passes.

    A different size would change the visual token count and the per-token
    advantage would then compare different positions — the failure that produces
    a wrong number rather than an error.
    """
    from PIL import Image

    image = Image.new("RGB", (128, 96), "white")
    assert degrade_image(image).size == image.size


def test_degradation_destroys_fine_detail_but_keeps_global_layout():
    import numpy as np
    from PIL import Image

    # A left half that is black and a right half of 1-pixel stripes: the halves
    # are the layout, the stripes are the detail the teacher would read text off.
    image = Image.new("RGB", (100, 100), "black")
    for x in range(50, 100, 2):
        for y in range(100):
            image.putpixel((x, y), (255, 255, 255))

    before = np.asarray(image.convert("L"), dtype=float)
    after = np.asarray(degrade_image(image).convert("L"), dtype=float)

    # Away from the boundary the two halves still differ, so layout survived.
    assert after[:, :40].mean() < 20
    assert after[:, 60:].mean() > 100
    # But the stripes are gone: their alternation collapses to near-flat grey.
    assert before[:, 60:].std() > 100
    assert after[:, 60:].std() < 20


def test_a_tiny_image_still_degrades_rather_than_vanishing():
    from PIL import Image

    assert degrade_image(Image.new("RGB", (4, 4), "white")).size == (4, 4)


def test_an_out_of_range_degradation_fraction_is_refused():
    from PIL import Image

    with pytest.raises(ValueError, match="fraction"):
        degrade_image(Image.new("RGB", (8, 8)), fraction=0.0)


def test_visual_advantage_is_rectified_at_zero():
    # Tokens the teacher predicts *better* without the detail carry no visual
    # supervision; the paper's max(., 0) discards them rather than negating them.
    original = torch.tensor([-1.0, -3.0, -2.0])
    degraded = torch.tensor([-4.0, -1.0, -2.0])
    assert torch.equal(visual_advantage(original, degraded), torch.tensor([3.0, 0.0, 0.0]))


def test_the_group_weights_reproduce_the_papers_grouped_loss_exactly():
    r"""mean(w * KL) is VA-OPD's L_group, term for term.

    The paper writes the loss as a weighted sum of two group *means*. Returning
    per-token weights instead lets the signal look like every other signal; this
    pins the two to be the same number, which is what "reproduction" has to mean.
    """
    va = torch.tensor([0.0, 0.0, 0.0, 5.0, 1.0])
    divergence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = va_group_weights(va, top_fraction=0.2, high_weight=0.5)

    high = divergence[[3]].mean()  # the top 20% by advantage: one token
    low = divergence[[0, 1, 2, 4]].mean()
    assert (weights * divergence).mean() == pytest.approx(0.5 * high + 0.5 * low)


def test_the_group_weights_average_to_one_so_the_loss_scale_is_unchanged():
    # B4 redistributes the distillation loss across tokens; it must not also
    # rescale it, or B4-versus-B2 would confound reweighting with a bigger term.
    va = torch.tensor([0.1, 0.0, 3.0, 0.4, 2.0, 0.0, 0.7, 1.1, 0.2, 0.9])
    assert va_group_weights(va).mean() == pytest.approx(1.0)


def test_high_advantage_tokens_are_weighted_above_low_advantage_ones():
    va = torch.tensor([0.0, 0.0, 0.0, 0.0, 9.0])
    weights = va_group_weights(va, top_fraction=0.2, high_weight=0.5)
    assert weights[4] > weights[0]


def test_no_visual_advantage_anywhere_means_no_reweighting():
    """VA-OPD's premise not holding on an example must not become noise.

    With every advantage zero the ranking is arbitrary, so up-weighting its top
    would concentrate the loss on tokens chosen by tie-breaking order. B4 falls
    back to B2 on those examples instead.
    """
    weights = va_group_weights(torch.zeros(7))
    assert torch.equal(weights, torch.ones(7))


def test_a_single_answer_token_is_not_split_into_groups():
    # Most VQA answers are one or two tokens; a split with an empty low group
    # would silently rescale the loss by lambda.
    assert torch.equal(va_group_weights(torch.tensor([2.0])), torch.ones(1))


def _b4_case():
    """Logits, a cached teacher top-k, and an advantage concentrated on one token."""
    logits = torch.tensor([[2.0, 0.5, 0.1, 0.0], [0.2, 3.0, 0.1, 0.0], [0.0, 0.0, 1.0, 2.0]])
    ids = torch.tensor([[0, 1], [1, 2], [3, 2]])
    values = torch.tensor([[4.0, 1.0], [3.0, 2.0], [5.0, 0.5]])
    advantage = torch.tensor([[0.0], [0.0], [4.0]])
    return logits, ids, values, advantage


def test_b4s_step_loss_is_exactly_the_papers_grouped_loss():
    """backbone + signal == mean(w * KL), term for term.

    The signal cannot return the reweighted loss itself: the backbone has already
    contributed mean(KL), so B4 would optimise mean(KL) + mean(w*KL) — twice B2's
    distillation weight, with the paper's 4:1 token ratio flattened to 2.15:1.
    Both runs complete and both produce a plausible number, which is exactly why
    this is pinned here.
    """
    from ceed_student.backbone import topk_kd_per_token

    logits, ids, values, advantage = _b4_case()
    signal = VisualAdvantageSignal(1.0, SignalOptions(), temperature=2.0)
    context = a_context(artefacts={VISUAL_ADVANTAGE: advantage}, logits=logits, topk=(ids, values))

    divergence = topk_kd_per_token(logits, ids, values, temperature=2.0)
    backbone_kd = divergence.mean()  # what the shared backbone contributes
    weights = va_group_weights(advantage.reshape(-1))

    step = backbone_kd + signal.token_loss(context).mean()
    assert step == pytest.approx(float((weights * divergence).mean()), rel=1e-6)


def test_b4_reweights_the_backbones_own_divergence_and_nothing_else():
    """B4 differs from B2 in the weighting alone.

    If the signal computed its own divergence, B4-versus-B2 would measure two
    changes at once — which is the failure the whole comparison is built to
    avoid.
    """
    from ceed_student.backbone import topk_kd_per_token

    logits, ids, values, advantage = _b4_case()
    signal = VisualAdvantageSignal(1.0, SignalOptions(), temperature=2.0)
    context = a_context(artefacts={VISUAL_ADVANTAGE: advantage}, logits=logits, topk=(ids, values))

    divergence = topk_kd_per_token(logits, ids, values, temperature=2.0)
    expected = (va_group_weights(advantage.reshape(-1)) - 1.0) * divergence
    assert torch.allclose(signal.token_loss(context), expected)


def test_b4_is_exactly_b2_on_an_example_with_no_visual_advantage():
    """The fallback must be a true no-op, not an approximate one.

    Where VA-OPD's premise does not hold the weights come back uniform, and the
    residual is then identically zero — so B4's objective on that example is B2's,
    to the bit.
    """
    logits, ids, values, _ = _b4_case()
    signal = VisualAdvantageSignal(1.0, SignalOptions(), temperature=2.0)
    context = a_context(
        artefacts={VISUAL_ADVANTAGE: torch.zeros(3, 1)}, logits=logits, topk=(ids, values)
    )
    assert torch.equal(signal.token_loss(context), torch.zeros(3))


def test_b4_needs_no_hidden_states_and_carries_no_parameters():
    signal = VisualAdvantageSignal(1.0, SignalOptions(), temperature=1.0)
    assert signal.required_student_layers() == frozenset()
    assert list(signal.parameters()) == []


# -- B3: hidden-state projection --------------------------------------------


def _hidden_spec(hidden_size: int = 3) -> VectorSpec:
    return VectorSpec(
        dtype="float32", shape=(len(CACHED_LAYERS), hidden_size), layers=CACHED_LAYERS
    )


def test_b3_reads_the_teacher_layer_the_store_says_it_is_at():
    """The store caches six layers; the mapping asks for one of them by name.

    Reading index 0 instead of the mapped layer's index would train perfectly
    happily against the wrong teacher state, and nothing downstream would notice.
    """
    mapping = LayerMapping(kind="proportional", pairs=((19, 0),))
    signal = HiddenStateProjectionSignal(1.0, mapping, _hidden_spec(), student_hidden_size=3)
    with torch.no_grad():  # make the projection the identity, so the loss is readable
        signal.projections["19"].weight.copy_(torch.eye(3))
        signal.projections["19"].bias.zero_()

    teacher = torch.zeros(1, len(CACHED_LAYERS), 3)
    teacher[0, CACHED_LAYERS.index(19)] = torch.tensor([1.0, 2.0, 3.0])
    teacher[0, 0] = torch.tensor([9.0, 9.0, 9.0])  # a decoy at the first cached layer

    context = a_context(
        hidden={0: torch.tensor([[1.0, 2.0, 3.0]])}, artefacts={HIDDEN_STATES: teacher}
    )
    assert signal.token_loss(context).item() == pytest.approx(0.0)


def test_b3_supervises_at_the_student_layer_the_mapping_names():
    mapping = LayerMapping(kind="proportional", pairs=((19, 26),))
    signal = HiddenStateProjectionSignal(1.0, mapping, _hidden_spec(), student_hidden_size=3)
    assert signal.required_student_layers() == frozenset({26})
    assert signal.required_kinds() == frozenset({HIDDEN_STATES})


def test_b3_is_refused_when_the_store_never_cached_the_mapped_layer():
    # Better here, at Group construction, than at hour six of a run.
    mapping = LayerMapping(kind="proportional", pairs=((17, 26),))
    with pytest.raises(MissingArtifactKindError, match="17"):
        HiddenStateProjectionSignal(1.0, mapping, _hidden_spec(), student_hidden_size=3)


def test_b3s_loss_falls_as_the_projection_learns_the_teacher_state():
    mapping = LayerMapping(kind="proportional", pairs=((9, 0),))
    signal = HiddenStateProjectionSignal(1.0, mapping, _hidden_spec(), student_hidden_size=3)
    teacher = torch.zeros(2, len(CACHED_LAYERS), 3)
    teacher[:, CACHED_LAYERS.index(9)] = torch.tensor([[1.0, -2.0, 0.5], [0.3, 0.1, -1.0]])
    context = a_context(
        hidden={0: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
        artefacts={HIDDEN_STATES: teacher},
    )

    optimizer = torch.optim.Adam(signal.parameters(), lr=0.1)
    first = float(signal.token_loss(context).mean().detach())
    for _ in range(50):
        optimizer.zero_grad()
        signal.token_loss(context).mean().backward()
        optimizer.step()
    assert float(signal.token_loss(context).mean().detach()) < first


# -- B5: the combine-weight probe -------------------------------------------


def _combine_spec(n_experts: int = 4) -> VectorSpec:
    return VectorSpec(dtype="float32", shape=(len(CACHED_LAYERS), n_experts), layers=CACHED_LAYERS)


def test_b5s_loss_is_minimised_when_the_probe_matches_the_normalised_router():
    """The target is the router's distribution, not its raw magnitudes.

    Effective combine weights do not sum to one (the per-expert scale sees to
    that), so a probe trained against them unnormalised would be learning the
    scale as well as the routing.
    """
    mapping = LayerMapping(kind="proportional", pairs=((9, 0),))
    signal = CombineWeightProbeSignal(1.0, mapping, _combine_spec(), student_hidden_size=2)

    weights = torch.zeros(1, len(CACHED_LAYERS), 4)
    weights[0, CACHED_LAYERS.index(9)] = torch.tensor([0.6, 0.0, 0.2, 0.0])  # top-2 of 4 routed
    context = a_context(
        hidden={0: torch.tensor([[1.0, 1.0]])}, artefacts={COMBINE_WEIGHTS: weights}
    )

    optimizer = torch.optim.Adam(signal.parameters(), lr=0.2)
    for _ in range(400):
        optimizer.zero_grad()
        signal.token_loss(context).mean().backward()
        optimizer.step()

    predicted = torch.softmax(signal.probes["9"](torch.tensor([[1.0, 1.0]])), dim=-1)
    assert torch.allclose(predicted, torch.tensor([[0.75, 0.0, 0.25, 0.0]]), atol=0.02)


def test_a_token_with_no_routing_at_all_contributes_nothing():
    # An all-zero combine-weight row has no distribution to normalise; dividing
    # by its sum would make the loss NaN and poison the whole run.
    mapping = LayerMapping(kind="proportional", pairs=((9, 0),))
    signal = CombineWeightProbeSignal(1.0, mapping, _combine_spec(), student_hidden_size=2)
    context = a_context(
        hidden={0: torch.tensor([[1.0, 1.0]])},
        artefacts={COMBINE_WEIGHTS: torch.zeros(1, len(CACHED_LAYERS), 4)},
    )
    loss = signal.token_loss(context)
    assert torch.isfinite(loss).all()
    assert loss.item() == pytest.approx(0.0)


# -- the registry: a Group's declared signals, checked against the store -----


def a_store(tmp_path, **kinds):
    """A real (empty) artifact store declaring the given artefact kinds."""
    from ceed_core import ArtifactStore, StoreMetadata

    return ArtifactStore.create(
        tmp_path / "store",
        StoreMetadata(extraction_fingerprint="fp", vector_kinds=kinds),
    )


def test_a_backbone_only_group_builds_no_signals(b2_config):
    assert build_signals(b2_config, None, student_hidden_size=8) == []


def test_a_group_whose_signal_kind_was_never_extracted_fails_immediately(b5_config, tmp_path):
    store = a_store(tmp_path, hidden_states=_hidden_spec())
    with pytest.raises(MissingArtifactKindError, match="combine_weights"):
        build_signals(b5_config, store, student_hidden_size=8)


def test_an_unknown_signal_name_is_refused_by_name(b5_config, tmp_path):
    from ceed_core import AuxiliarySignalConfig

    config = b5_config.model_copy(
        update={"auxiliary_signals": (AuxiliarySignalConfig(name="telepathy"),)}
    )
    store = a_store(tmp_path, combine_weights=_combine_spec())
    with pytest.raises(ValueError, match="telepathy"):
        build_signals(config, store, student_hidden_size=8)


def test_a_signal_needing_a_store_refuses_to_build_without_one(b3_config):
    with pytest.raises(ValueError, match="artifact store"):
        build_signals(b3_config, None, student_hidden_size=8)


def test_each_baseline_group_builds_exactly_its_declared_signal(
    b3_config, b4_config, b5_config, tmp_path
):
    store = a_store(
        tmp_path,
        hidden_states=_hidden_spec(),
        combine_weights=_combine_spec(),
        visual_advantage=VectorSpec(dtype="float32", shape=(1,)),
    )
    for config, expected in (
        (b3_config, "hidden_state_projection"),
        (b4_config, "visual_advantage_reweighting"),
        (b5_config, "combine_weight_probe"),
    ):
        signals = build_signals(config, store, student_hidden_size=8)
        assert [signal.name for signal in signals] == [expected]


# -- precision: an fp16 Student, an fp32 probe -------------------------------


def test_a_probe_stays_in_fp32_over_an_fp16_student():
    """The Student runs in fp16 on Volta; the probe must not follow it down.

    A half-precision head optimises against its own rounding, and the two dtypes
    meeting in a matmul is a hard error rather than a silent one — so the
    Student's state is widened to the probe's precision, not the reverse.
    """
    mapping = LayerMapping(kind="proportional", pairs=((9, 0),))
    for signal, kind, artefact in (
        (
            HiddenStateProjectionSignal(1.0, mapping, _hidden_spec(), student_hidden_size=3),
            HIDDEN_STATES,
            torch.zeros(1, len(CACHED_LAYERS), 3),
        ),
        (
            CombineWeightProbeSignal(1.0, mapping, _combine_spec(), student_hidden_size=3),
            COMBINE_WEIGHTS,
            torch.zeros(1, len(CACHED_LAYERS), 4),
        ),
    ):
        context = a_context(
            hidden={0: torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)},
            artefacts={kind: artefact},
        )
        loss = signal.token_loss(context)
        assert loss.dtype is torch.float32
        assert torch.isfinite(loss).all()
