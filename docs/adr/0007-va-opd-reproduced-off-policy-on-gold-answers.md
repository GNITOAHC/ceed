---
status: accepted
---

# B4 reproduces VA-OPD's visual advantage and token reweighting, off-policy on gold answers

B4 is the plan's **critical external baseline**: if CEED's region-intervention signals do not beat a published method that already uses counterfactual image conditions, the contribution is much smaller than claimed. So B4 was built against the source paper — *Visual-Advantage On-Policy Distillation for Vision-Language Models* (arXiv:2605.21924) — rather than against the plan's one-line description of it ("global degradation, token reweighting"), and this records exactly which parts are the paper's and which are ours.

Reading the paper changed the implementation in three ways the plan's description would not have produced: the degradation is a specific recipe (10% bilinear down, nearest up) rather than "blur"; the advantage is **rectified at zero**, which discards tokens the teacher predicts better without the detail; and the reweighting is by **rank** (the top `p_v` fraction) rather than by the advantage's magnitude.

## Reproduced exactly

| Element | The paper | Here |
| --- | --- | --- |
| Degradation | downsample to 10% of spatial resolution (bilinear), upsample back (nearest) | `ceed_teacher.va_opd.degrade_image`, same factor and same interpolation pair |
| Visual advantage | `a_t = max(log p_T(y_t \| v, q, y_<t) − log p_T(y_t \| ṽ, q, y_<t), 0)` | `ceed_teacher.va_opd.visual_advantage`, cached per answer token |
| Grouping | rank tokens by `a_t`; the top `p_v = 0.2` form the high-VA group | `ceed_student.signals.va_group_weights`, `va_top_fraction` |
| Group weighting | `λ · mean_V(KL) + (1 − λ) · mean_L(KL)`, `λ = 0.5` | the same expression, expressed as per-token weights whose mean reproduces it identically |

Image dimensions are preserved through the degradation, as the paper requires: the two teacher passes must emit the same number of visual tokens or the per-token advantage compares different positions. Extraction checks this per example and refuses rather than caching a misaligned advantage.

## Deviations, and why

**Off-policy on gold answers, not on-policy on student rollouts.** The paper scores *student-generated rollouts* with the teacher during training. CEED cannot: ADR-0001 forbids loading the teacher during student training (every teacher measurement is precomputed), ADR-0002 fixes supervision to the teacher-forced gold answer, and amendment A9 forbids sampling. So B4's advantage is measured on the gold answer, cached once, and read from the store like every other signal.

This is the deviation that most limits what B4 can claim, and it is stated in any reported comparison. It is also unavoidable *within CEED*, and the alternative is worse: an on-policy B4 would differ from every other Group in its supervised token set as well as its signal, and the plan's whole design rests on the Groups differing only in the auxiliary signal.

**The rollout-level reweighting degenerates.** The paper's first granularity z-scores trajectory-averaged advantage across `K` sibling rollouts and softmaxes it into per-rollout weights. Under teacher forcing there is exactly one sequence per example, so `K = 1`: the z-score is undefined and the softmax is `1.0`. B4 therefore implements the token-level granularity only, and says so. This is a consequence of the point above, not an independent choice.

**The divergence is the shared backbone's, not a new one.** The paper's `KL_t` is a reverse KL, which is the natural choice for on-policy distillation; ours is the backbone's forward KL over the teacher's cached top-k support. Matching the paper's direction here would have made B4 differ from B2 in *two* ways — reweighting and divergence — and the delta would no longer isolate the reweighting. B4 reweights `ceed_student.backbone.topk_kd_per_token`, the identical function B2 averages.

Because every Group's step loss is `backbone + Σ signal`, and the backbone has already contributed `mean(KL)`, the signal returns the **residual** `(w_t − 1)·KL_t` rather than `w_t·KL_t`. The sum is then exactly `L_group`. Returning the reweighted loss itself would optimise `mean(KL) + mean(w·KL)`: twice B2's distillation weight, and a high-to-low token ratio of 2.15 against the paper's 4. Both variants train and both report a plausible number, so `tests/test_signals.py::test_b4s_step_loss_is_exactly_the_papers_grouped_loss` pins the identity rather than the implementation.

**Uniform weights when the split is degenerate.** CEED's gold answers are frequently one or two tokens, and the paper's rollouts are long, so two cases arise that it never meets. When the high-VA group would take every token there is no low group, and when *every* advantage is zero the ranking ranks noise. Both fall back to uniform weights, so B4 reduces to B2 on those examples rather than concentrating the loss on a token chosen by tie-breaking order.

## Consequences

- B4's numbers are a reproduction of VA-OPD's *signal* inside CEED's training frame, not a reproduction of the paper's results, and must be reported that way. A B4-versus-published-VA-OPD comparison is not available and is not claimed.
- `visual_advantage` roughly doubles extraction cost for the examples it covers: a second full teacher forward per example. It is opt-in (`--with-visual-advantage`) for that reason.
- If E2 beats B4 on grounding selectivity, the honest reading is that region-level interventions beat *this* adaptation of global degradation. Whether they beat on-policy VA-OPD is a separate question the plan does not answer.
