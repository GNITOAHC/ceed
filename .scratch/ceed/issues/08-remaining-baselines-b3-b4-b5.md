# 08 — Remaining baselines: B3, B4, and B5

**What to build:** The baseline group completes. Three more Groups train and report numbers, and between them they prove the auxiliary-signal contract holds for three quite different kinds of signal — a cached per-expert vector, a cached hidden state under a layer mapping, and a second teacher pass over degraded images.

B5 is the one to build first: it is the cheapest real auxiliary signal and it is what demonstrates that adding a Group is a component plus a YAML overlay rather than a new training script.

**Blocked by:** 06, 07

**Status:** ready-for-agent

- [x] **B5** distils the teacher's effective combine weights, and adding it required only a new auxiliary-signal component and a configuration overlay — `CombineWeightProbeSignal` in `ceed_student/signals.py` plus `configs/groups/b5.yaml`. A removable probe on the Student's residual state predicts the teacher's normalised combine-weight distribution; the target is the *effective* weight (A3), read off the real router by `ceed_teacher/router.py`. `test_router.py` pins that it is the scaled weight and not the router probability.
- [x] A layer mapping is a first-class object rather than an implicit convention, because a deliberately mismatched mapping is a control Group and Phase 2 reports variance across alternatives — `ceed_core/layer_mapping.py`: `LayerMapping` gains resolution (`student_layer_for`), validation (no duplicate teacher layer, non-negative, non-empty), and two constructors — `proportional_mapping` (B3's placeholder) and `mismatched_mapping` (C2's control, a derangement over the same student layers so no pair survives). `test_layer_mapping.py`.
- [x] **B3** performs hidden-state projection distillation under a proportional placeholder mapping, supervising answer tokens only — the same token set as every other Group (see ADR-0001 and plan amendment A5) — `HiddenStateProjectionSignal` plus `configs/groups/b3.yaml`. A learned projection per mapped layer carries 2560 to 2816 and is matched by MSE at the answer tokens.
- [x] **B4** reproduces VA-OPD: teacher passes over globally degraded images producing per-token reweighting. The reproduction is checked against the source paper rather than inferred from its description in the plan — the paper (arXiv:2605.21924) was read, and **ADR-0007** records line by line what is reproduced exactly (10% bilinear-down/nearest-up degradation, the rectified advantage, `p_v = 0.2`, `λ = 0.5`, the grouped loss) and what deviates (off-policy on gold answers; the rollout-level granularity degenerates at K=1; the divergence is the backbone's so that B4 differs from B2 in the weighting alone). Reading the paper changed three things the plan's one-line description would not have.
- [x] All three run through `run_group` and report accuracy — verified on the real 4×V100 box against the real 26B Teacher's artefacts and the real 8B Student. B3, B4 and B5 each trained, evaluated, and recorded their own `train.aux.<signal>` metric.
- [~] The baseline group is complete: B0 through B5 all have numbers, from frozen reusable checkpoints — **the machinery is complete and proven end to end; the full-corpus numbers are not run.** See below.

## What is not done

**B3, B4 and B5 have no full-corpus numbers.** Everything needed to produce them exists and has been exercised end to end on real weights at toy scale (3 examples, 3 steps), but the real runs need a full-corpus extraction with `--all` — a second store, since a store's schema is fixed at creation — plus three ~2000-step trainings and three 565-example evaluations. That is GPU-hours, not code.

The extraction is the long pole and is the one thing that cannot be skipped: `--with-visual-advantage` doubles the forward cost, and the B2 store already on disk holds only the top-k logits.

## Measured during this ticket

**`per_expert_scale` reorders less than A3 implies.** It spans 0.9805–1.0234 across all 30 layers, so over 300 (answer token, layer) pairs it changes the top-8 ordering in 26% of cases but the top-1 expert in only 1.3%. A3's premise stands — the three candidate definitions are genuinely not monotone transforms — but the divergence lives in the tail of the top-8. Recorded as a measured note in ADR-0003.

**VA-OPD's premise reproduces on real data.** Visual advantage is sharply concentrated: on the smoke examples one token carried 0.96, 2.69 and 9.70 nats respectively while the rest sat at ~0. That is the paper's central observation, holding on DocVQA under the teacher used here.

**B3's auxiliary head is ten times its adapter.** The projections are 21.6M parameters against a rank-4 adapter's 2.3M (B5's probes are 1.0M). A B3 null under LoRA is therefore even harder to attribute than ADR-0005 already warns; noted in `configs/groups/b3.yaml` and the running guide.

## Fixed in review

**B4 double-counted the distillation term.** The signal returned `w·KL` while the backbone had already contributed `mean(KL)`, so B4 optimised `mean(KL) + mean(w·KL)` — twice B2's distillation weight, with the paper's 4:1 high-to-low token ratio flattened to 2.15:1. It now returns the residual `(w − 1)·KL`, so the step loss is exactly VA-OPD's `L_group`, B4's backbone configuration stays identical to B2's, and B4 reduces to B2 exactly where the weights come back uniform. Pinned by `test_b4s_step_loss_is_exactly_the_papers_grouped_loss`.

**A new config field silently changed every trained Group's identity.** `TrainingConfig.coupling_threshold` was added for a gate no baseline uses; because the run hash covers the whole configuration, it changed B0, B1 and B2's hashes — including the B2 hash published in the provenance of the merged checkpoint. Removed (the gate keeps its own default), and `test_a_trained_baselines_run_hash_does_not_drift` now pins all three hashes so this cannot recur.

Also from review: the store schema had two definitions that had already diverged (the script's knew `visual_advantage`, `ceed_teacher`'s did not) — now one, `store_schema`, which both derive from; the router capture keyed by depth rather than by the layer's own index; `gold_logprobs` annotated an id tensor as `Float`; `build_signals` carried an if/elif cascade duplicating the registry it sat beside; and a parameter used `head`, which CONTEXT.md lists as a term to avoid for a probe.

## Follow-on

- The checked-in `layer_mapping` in `base.yaml` maps `(29, 39)`; the proportional rule derives `(29, 40)`. The pairs are data and are recorded on every run record, so the mapping a run used is checkable either way — but the two disagree. Left alone deliberately: changing `base.yaml` changes **every** config hash, including those of the already-trained B1 and B2 checkpoints and the published merged B2 model. The spec has B3 re-run under the probe-based mapping anyway, which is the natural moment to fix it.
- `batch_size` is still declared and not honoured — the loop consumes one example per step. It affects every Group equally, so B2-to-B5 comparisons stay valid, but the step budget does not mean what the config says.
