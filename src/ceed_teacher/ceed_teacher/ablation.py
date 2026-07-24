"""The Causal Expert Attribution engine.

For one answer token and one hybrid layer, this measures how much each probed
expert's specific output matters to the answer token's gold log-probability, by
replacing that expert's output with the mean of the token's activated experts and
reading the change (see docs/adr/0003).

The measurement is defined uniformly over two kinds of probed expert:

* an **activated** expert, which the router selected, is replaced in place;
* a **near-miss** expert, which the router did not select but which ranks highest
  by effective combine weight among the rest, is first routed in at its effective
  combine weight and then replaced,

so that near-miss experts contribute an attribution on the same footing as
activated ones and the measurement does not inherit the router's blind spots. The
shared dense path is never touched.

All probed experts for a ``(token, layer)`` are evaluated in a single batched
resume, so attribution costs roughly one partial forward rather than one per
expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from ceed_teacher.protocol import HybridForward


@dataclass(frozen=True)
class AttributionResult:
    """A Causal Expert Attribution vector and the experts it ranges over.

    Attributes:
        experts: The probed expert ids, activated experts first, then near-miss
            experts in descending effective-combine-weight order.
        attribution: The attribution of each probed expert, aligned with
            ``experts``: the drop in the gold answer token's log-probability when that
            expert's output is replaced by the mean of the activated experts'
            outputs.
        n_activated: How many leading entries of ``experts`` are activated
            experts; the remainder are near-miss experts.
    """

    experts: Int[Tensor, " probed_experts"]
    attribution: Float[Tensor, " probed_experts"]
    n_activated: int


def select_probed_experts(
    fwd: HybridForward, layer: int, token: int, n_near_miss: int
) -> tuple[Int[Tensor, " top_k"], Int[Tensor, " n_near_miss"]]:
    """Return the activated experts and the near-miss experts for this token.

    Activated experts are the router's own selection. Near-miss experts are the
    ``n_near_miss`` non-activated experts with the highest effective combine
    weight — ranked by that weight, never by raw router logit.

    Args:
        fwd: The completed forward to query.
        layer: The hybrid layer.
        token: The answer-token position.
        n_near_miss: How many near-miss experts to include.

    Returns:
        A pair ``(activated, near_miss)`` of expert-id tensors.
    """
    activated = fwd.activated_experts(layer, token)
    weights = fwd.effective_combine_weights(layer, token)
    is_activated = torch.zeros_like(weights, dtype=torch.bool)
    is_activated[activated] = True
    masked = weights.masked_fill(is_activated, float("-inf"))
    near_miss = torch.topk(masked, n_near_miss).indices
    return activated, near_miss


def replace_expert_output(
    baseline_ffn: Float[Tensor, " hidden"],
    weight: Tensor,
    expert_output: Float[Tensor, " hidden"],
    replacement: Float[Tensor, " hidden"],
) -> Float[Tensor, " hidden"]:
    """Return ``baseline_ffn`` with one expert's output swapped for ``replacement``.

    The expert contributes ``weight * expert_output`` to the feed-forward output;
    this returns the feed-forward output with that contribution changed to
    ``weight * replacement``. When ``replacement`` is the expert's own output the
    result is bit-identical to ``baseline_ffn``, because the delta is exactly
    zero rather than a subtract-then-add.

    Args:
        baseline_ffn: The feed-forward output the expert already contributes to.
        weight: The expert's effective combine weight.
        expert_output: The expert's current output.
        replacement: The output to substitute.

    Returns:
        The adjusted feed-forward output.
    """
    return baseline_ffn + weight * (replacement - expert_output)


def attribution_vector(
    fwd: HybridForward, layer: int, token: int, n_near_miss: int = 2
) -> AttributionResult:
    """Compute the Causal Expert Attribution vector for one answer token and layer.

    Args:
        fwd: The completed teacher-forced forward to attribute.
        layer: The hybrid layer to attribute at.
        token: The answer-token position to attribute.
        n_near_miss: How many near-miss experts to include beyond the activated
            ones.

    Returns:
        The attribution over the activated experts followed by the near-miss
        experts.
    """
    activated, near_miss = select_probed_experts(fwd, layer, token, n_near_miss)
    probed = torch.cat([activated, near_miss])

    weights = fwd.effective_combine_weights(layer, token)
    expert_out = fwd.expert_outputs(layer, token)
    active_mean = expert_out[activated].mean(dim=0)
    baseline_ffn = fwd.ffn_output(layer, token)
    activated_set = set(activated.tolist())

    # One base and one mean-replaced variant per probed expert, evaluated together.
    variants: list[Tensor] = []
    for expert in probed.tolist():
        weight = weights[expert]
        output = expert_out[expert]
        if expert in activated_set:
            base = baseline_ffn
        else:
            base = replace_expert_output(baseline_ffn, weight, torch.zeros_like(output), output)
        replaced = replace_expert_output(base, weight, output, active_mean)
        variants.append(base)
        variants.append(replaced)

    logits = fwd.resume(layer, token, torch.stack(variants))  # [2 * probed, vocab]
    gold = fwd.gold_token_id(token)
    log_probs = torch.log_softmax(logits, dim=-1)[:, gold]  # [2 * probed]
    base_lp = log_probs[0::2]
    replaced_lp = log_probs[1::2]
    return AttributionResult(
        experts=probed,
        attribution=base_lp - replaced_lp,
        n_activated=int(activated.shape[0]),
    )
