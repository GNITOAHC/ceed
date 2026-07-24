"""The secondary seam: numerical invariants of the CEA ablation engine.

These properties are not observable in a run record; they live at the layer
boundary, and their violation is exactly what would produce plausible-looking
wrong science. They run against the synthetic hybrid teacher on CPU in
milliseconds.
"""

import jaxtyping
import pytest
import torch

from ceed_teacher import (
    SyntheticConfig,
    SyntheticHybridTeacher,
    attribution_vector,
    replace_expert_output,
    select_probed_experts,
)


def a_teacher(seed: int = 0) -> SyntheticHybridTeacher:
    teacher = SyntheticHybridTeacher(SyntheticConfig(seed=seed))
    teacher.run(torch.tensor([3, 1, 4, 1, 5, 9]))
    return teacher


def test_replacing_an_expert_with_its_own_output_is_a_bit_exact_no_op():
    teacher = a_teacher()
    layer, token = 1, 2
    baseline = teacher.ffn_output(layer, token)
    weight = teacher.effective_combine_weights(layer, token)[0]
    output = teacher.expert_outputs(layer, token)[0]

    unchanged = replace_expert_output(baseline, weight, output, output)
    assert torch.equal(unchanged, baseline)


def test_shared_dense_plus_weighted_experts_equals_the_ffn_output():
    teacher = a_teacher()
    layer, token = 1, 3
    shared = teacher.shared_dense_output(layer, token)
    weights = teacher.effective_combine_weights(layer, token)
    outputs = teacher.expert_outputs(layer, token)
    activated = teacher.activated_experts(layer, token)

    routed = (weights[activated].unsqueeze(-1) * outputs[activated]).sum(dim=0)
    assert torch.allclose(shared + routed, teacher.ffn_output(layer, token), atol=1e-6)


def test_ablating_a_token_leaves_earlier_tokens_bit_identical():
    teacher = a_teacher()
    layer, token = 0, 3
    baseline_ffn = teacher.ffn_output(layer, token)

    # Two variants resumed together: the unchanged FFN, and a nudged one.
    variants = torch.stack([baseline_ffn, baseline_ffn + 1.0])
    full = teacher.resume_full(layer, token, variants)  # [2, seq, vocab]
    reference, nudged = full[0], full[1]

    # Nudging this token cannot change any earlier token's logits.
    assert torch.equal(nudged[:token], reference[:token])
    assert not torch.equal(nudged[token], reference[token])
    # Resuming with the unchanged FFN reproduces the baseline at this token.
    baseline_logits = teacher.resume(layer, token, baseline_ffn.unsqueeze(0))
    assert torch.equal(baseline_logits[0], reference[token])


def test_mean_of_active_replacement_preserves_residual_norm_within_tolerance():
    teacher = a_teacher()
    mean_changes = []
    zero_changes = []
    for layer in range(teacher.n_layers):
        for token in range(1, 5):
            weights = teacher.effective_combine_weights(layer, token)
            outputs = teacher.expert_outputs(layer, token)
            activated = teacher.activated_experts(layer, token)
            active_mean = outputs[activated].mean(dim=0)
            ffn = teacher.ffn_output(layer, token)
            # The residual stream is the post-attention state plus the FFN output;
            # ADR-0003 is a claim about this norm, not the FFN output's alone.
            residual = teacher.post_attention_output(layer, token) + ffn
            norm = residual.norm()
            for expert in activated.tolist():
                weight, output = weights[expert], outputs[expert]
                mean_residual = teacher.post_attention_output(layer, token) + replace_expert_output(
                    ffn, weight, output, active_mean
                )
                zero_residual = teacher.post_attention_output(layer, token) + replace_expert_output(
                    ffn, weight, output, torch.zeros_like(active_mean)
                )
                mean_changes.append(float((mean_residual.norm() - norm) / norm))
                zero_changes.append(float((zero_residual.norm() - norm) / norm))

    mean_changes_t = torch.tensor(mean_changes)
    zero_changes_t = torch.tensor(zero_changes)
    # Per case, mean replacement keeps the residual-stream norm within a tight band.
    assert mean_changes_t.abs().max() < 0.05
    # Systematically, zeroing shrinks the residual norm while mean does not
    # (ADR-0003): the average signed change is markedly more negative for zeroing.
    assert zero_changes_t.mean() < mean_changes_t.mean()
    assert zero_changes_t.mean() < 0


def test_near_miss_experts_are_selected_by_effective_combine_weight():
    teacher = a_teacher()
    layer, token = 1, 3
    activated, near_miss = select_probed_experts(teacher, layer, token, n_near_miss=2)

    weights = teacher.effective_combine_weights(layer, token)
    activated_set = set(activated.tolist())
    remaining = [e for e in range(teacher.n_experts) if e not in activated_set]
    expected = sorted(remaining, key=lambda e: float(weights[e]), reverse=True)[:2]
    assert near_miss.tolist() == expected
    # Every activated expert outranks every near-miss expert by effective weight.
    assert weights[activated].min() >= weights[near_miss].max()


def test_attribution_vector_covers_activated_then_near_miss_experts():
    teacher = a_teacher()
    result = attribution_vector(teacher, layer=1, token=2, n_near_miss=3)

    assert result.n_activated == teacher.top_k
    assert result.experts.shape[0] == teacher.top_k + 3
    assert result.attribution.shape[0] == result.experts.shape[0]
    # Activated experts lead, near-miss experts follow.
    assert result.experts[: result.n_activated].tolist() == teacher.activated_experts(1, 2).tolist()


def test_attribution_shape_annotations_are_enforced_under_pytest():
    # resume expects [variants, hidden]; a bare [hidden] vector is the wrong rank,
    # and the beartype import hook installed by the root conftest must reject it.
    teacher = a_teacher()
    with pytest.raises(jaxtyping.TypeCheckError):
        teacher.resume(0, 2, torch.zeros(teacher.hidden_size))


def test_attribution_is_finite_for_a_well_formed_forward():
    teacher = SyntheticHybridTeacher(SyntheticConfig(seed=1))
    teacher.run(torch.tensor([2, 7, 1, 8, 2, 8]))
    result = attribution_vector(teacher, layer=0, token=3, n_near_miss=0)
    assert torch.isfinite(result.attribution).all()
    assert result.experts.shape[0] == teacher.top_k
