# CEED implementation

Labels: `ready-for-agent`

## Problem Statement

The CEED research plan (`ceed.md`, as amended by `docs/plan-amendments-v2.1.md`) describes a twelve-group controlled distillation comparison plus a four-step teacher characterisation study, and no code exists to run any of it. Today a researcher wanting to answer the plan's central question — does a MoE teacher's causally-verified division of computational labour transfer to a compute-matched dense student — cannot measure a single Causal Expert Attribution, cannot produce a Mode, cannot generate a relevant/control intervention pair, and cannot train or evaluate a single Group.

Three properties make this harder than "write the training script":

1. **The comparison is the contribution.** The twelve Groups must differ *only* in their auxiliary signal. Any accidental asymmetry — different supervised token sets, different corpora, different decoding — silently invalidates the result rather than breaking the run.
2. **The failure mode is wrong numbers, not exceptions.** An off-by-one between a cached answer token index and a student cross-entropy target, an ablation that perturbs the shared dense path, a control region that is not texture-matched: each produces a Phase 1 table that looks entirely reasonable and means nothing.
3. **The measurement may not survive contact with the teacher.** Because every teacher layer is a hybrid layer, expert ablation is measured against an intact shared dense path, and CEA magnitudes may sit at the noise floor. The researcher needs to know this in week one, not week six.

## Solution

A uv workspace of seven packages under `src/`, split along the pipeline's artifact-flow boundaries, in which the **artifact store** is the single seam between teacher-side measurement and student-side training, and a single `run_group` entry point takes a Group configuration and produces a run record containing that Group's numbers.

The researcher gets, in order:

- A **walking skeleton** that proves the whole chain end-to-end and produces B0 — a real zero-shot accuracy number — before any expensive teacher work begins. This is where the fp16-on-Volta risk is discovered if it exists.
- A **teacher extraction stage** that fills the artifact store with everything the baseline Groups need, over-caching effective combine weights at all 30 layers and hidden states at 6 candidate layers so that the eventual three-layer choice never blocks a baseline.
- A **student training stage** whose auxiliary signal is a swappable component, so adding E1's Mode probing or E2's selectivity matching is a new auxiliary signal, not a new training script.
- A **Phase 0 analysis stage** that reads the store with SQL and answers 0.0 through 0.4.
- An **intervention stage** producing relevant/control pairs, gated behind Phase 0.4's artefact validation.

The ordering is deliberately the researcher's: the entire baseline Group set (B0–B5) is deliverable before Phase 0 runs, then Phase 0, then the control and experimental Groups.

## User Stories

### Getting started and environment

1. As a researcher, I want `uv sync --extra v100` to produce a working environment on the Volta box, so that CUDA-capability mismatches surface at install time rather than at hour six of an extraction run.
2. As a researcher, I want one lockfile covering both the Volta and scale-out targets, so that the two environments cannot drift in any dependency other than torch.
3. As a researcher, I want to load the Student in fp16 on a V100 and get a coherent answer to one question, so that I learn whether Gemma-4 is numerically stable on Volta before committing to the extraction budget.
4. As a researcher, I want each package installable without its siblings' dependency trees, so that a `diffusers` pin can never move the `transformers` version teacher extraction depends on.
5. As a maintainer, I want ruff, mypy, and the fast test tier to run in CI on every change, so that the repo does not rot between research sprints.

### Corpus

6. As a researcher, I want DocVQA, GQA, and ChartQA normalised into one Example type with stable identifiers, so that every Group provably trains on the identical corpus.
7. As a researcher, I want a written corpus manifest recording exactly which examples and splits a run used, so that a Group trained in week two is comparable to one trained in week ten.
8. As a researcher, I want ChartQA marked as ineligible for interventions in the corpus itself, so that the exclusion is enforced by the data rather than remembered by the operator.
9. As a researcher, I want DocVQA examples to carry their OCR words and boxes, so that relevant regions and text-substitution counterfactuals are derivable without a second annotation pass.
10. As a researcher, I want GQA examples to carry the scene-graph object box tied to the answer, so that relevant regions are annotation-backed.
11. As a researcher, I want deterministic, seed-recorded splits, so that a held-out set is genuinely held out across all twelve Groups.

### Teacher measurement

12. As a researcher, I want teacher-forced forward passes over gold answers, so that cached artefacts align 1:1 with the Student's cross-entropy targets in every Group.
13. As a researcher, I want top-k teacher logits cached per answer token, so that the backbone's logit KD is identical across every Group and computed once.
14. As a researcher, I want effective combine weights cached for all 30 teacher layers, so that choosing the three CEA layers later never forces a re-extraction.
15. As a researcher, I want teacher hidden states cached at 6 candidate layers, so that B3 and the layer-mapping study can both proceed before the layer sweep concludes.
16. As a researcher, I want a free-generation pass producing a per-example correctness flag, so that supervision can be gated away from the teacher's own errors.
17. As a researcher, I want to ablate a single expert for a single answer token and measure the change in that token's log-probability, so that a Causal Expert Attribution exists at all.
18. As a researcher, I want all probed experts for one (token, layer) computed as a single batched set of variants, so that attribution extraction costs roughly one partial forward rather than eleven full ones.
19. As a researcher, I want near-miss experts included in the attribution vector, so that measurement does not inherit the router's blind spots.
20. As a researcher, I want the shared dense path left untouched during ablation, so that CEA measures expert contribution and not the layer's whole FFN.
21. As a researcher, I want each ablation to leave tokens before the ablated one bit-identical, so that the cached-KV assumption underpinning the cost model is verified rather than assumed.
22. As a researcher, I want extraction to resume from a partially written store, so that a preempted multi-day job does not restart from zero.
23. As a researcher, I want extraction to run across several workers writing independent shards, so that the four local GPUs are usable without write coordination.

### Artifact store

24. As a researcher, I want every artefact keyed by example and answer-token index, so that joining logits, combine weights, hidden states, and attributions needs no alignment logic.
25. As a researcher, I want the store's location determined by the extraction fingerprint, so that training a Group against artefacts from a different ablation definition is impossible rather than merely unlikely.
26. As a researcher, I want to query the store with SQL, so that Phase 0's correlations and mutual information are analysis rather than array plumbing.
27. As a researcher, I want a store to declare which artefact kinds it contains, so that starting a Group whose signal was never extracted fails immediately with a clear message.

### Interventions

28. As a researcher, I want a relevant region derived from dataset annotations, so that ECC results are annotation-backed.
29. As a researcher, I want a control region matched to its relevant region in size and texture, so that the invariance supervision means what it claims.
30. As a researcher, I want regions removed by inpainting rather than rectangle masking, so that the Student cannot learn to detect the intervention instead of losing the evidence.
31. As a researcher, I want text-substitution counterfactuals on document examples, so that the cleanest available causal semantics are exercised where they exist.
32. As a researcher, I want intervened images persisted and content-addressed, so that the teacher's and the Student's intervened passes see byte-identical inputs.
33. As a researcher, I want the intervention pipeline validated against Phase 0.4 before any E-Group trains, so that mask artefacts cannot poison the selectivity signal.

### Phase 0 analysis

34. As a researcher, I want the shared-dense versus sparse share of FFN output norm reported per layer, so that I learn on day one whether CEED's signal is structurally attenuated.
35. As a researcher, I want CEA magnitude and across-expert diversity swept over all 30 layers on a small subset, so that the three CEA layers are chosen by measurement rather than asserted.
36. As a researcher, I want the per-token Spearman correlation between effective combine weight and measured attribution, so that Phase 0.1's routing–attribution divergence has a number.
37. As a researcher, I want attribution vectors clustered per layer into Modes, so that the Student's probe has a stable supervision target immune to expert-ID permutation.
38. As a researcher, I want Mode assignment compared by mutual information against router-cluster and hidden-state-cluster baselines at equal cluster count, so that Phase 0.2's go/no-go is decidable.
39. As a researcher, I want Mode stability measured across subsets, paraphrases, and same-token/different-image pairs, so that I can tell whether Modes track function or token identity.
40. As a researcher, I want attribution reorganisation compared between relevant and control interventions, so that Phase 0.4's responsiveness criterion has a number.
41. As a researcher, I want each Phase 0 step to emit its go/no-go verdict against the plan's stated criterion, so that the gate is recorded rather than argued.

### Student training

42. As a researcher, I want the backbone — cross-entropy plus top-k logit KD — implemented once and shared by every Group, so that no Group can accidentally differ in it.
43. As a researcher, I want each auxiliary signal to be a swappable component, so that adding a new Group is a configuration change and a new component, not a new training script.
44. As a researcher, I want to run the Student on original and both intervened images within one training step, so that selectivity-profile matching is expressible.
45. As a researcher, I want auxiliary losses gated per token by teacher-measured coupling strength, so that supervision concentrates where the teacher shows evidence–computation coupling.
46. As a researcher, I want auxiliary terms introduced on a warm-up schedule after the backbone, so that they cannot destabilise early training.
47. As a researcher, I want probes to be deletable, so that the deployed Student is architecturally unchanged and the zero-added-inference-cost claim is literally true.
48. As a researcher, I want to run a Group under LoRA locally and under full fine-tuning remotely from one configuration, so that a smoke test and a headline run are the same code path.
49. As a researcher, I want the run record to state unambiguously whether a Group was LoRA or fully fine-tuned, so that a LoRA number can never reach the Phase 1 table.
50. As a researcher, I want training to checkpoint and resume, so that a multi-day Group survives preemption.
51. As a researcher, I want Baseline checkpoints frozen and reused, so that no Baseline is ever retrained across later comparisons.

### Evaluation

52. As a researcher, I want accuracy and hallucination numbers produced by `lmms-eval`, so that they are comparable to published baselines rather than to my own metric reimplementation.
53. As a researcher, I want greedy decoding enforced regardless of what the checkpoint's generation config ships, so that decoding variance does not swamp a predicted +0.5 point effect.
54. As a researcher, I want three seeds on the Groups whose deltas carry claims, so that "E1 ≫ C1" is falsifiable.
55. As a researcher, I want the Student's own grounding selectivity measured on held-out interventions, so that the plan's novel evaluation protocol produces a number.
56. As a researcher, I want the Phase 1 comparison table generated from the run records on disk, so that the decision logic in the plan is a computation over files I own rather than a dashboard reading.

### Reproducibility

57. As a researcher, I want every run identified by a hash of its canonical configuration, so that two runs claiming to be the same Group provably are.
58. As a researcher, I want metrics written to disk as the source of truth with the tracking service as an optional viewer, so that results survive the absence of an account or a network.
59. As a reviewer, I want to see which layer mapping a reported B3 or E1 used, so that the layer-mapping robustness claim is checkable.
60. As a researcher, I want frozen golden attribution outputs for a couple of fixed examples, so that a refactor cannot silently change ablation semantics.

## Implementation Decisions

Recorded ADRs are authoritative for the decisions they cover: `0001` artifact store scope and format, `0002` teacher-forced gold answers with thinking disabled, `0003` ablation definition and the shared dense path, `0004` the seven-package workspace, `0005` LoRA-versus-full-fine-tune, `0006` the cu126 torch pin. The following are the remaining decisions and interface commitments.

### Package responsibilities

- **`ceed_core`** — domain types, configuration schema, artifact store read/write, extraction fingerprinting. Depends on nothing in the project and on no model, dataset, or training framework. Every other package depends on it.
- **`ceed_data`** — dataset loaders normalising DocVQA, GQA, and ChartQA into one Example type; splits; corpus manifest. Carries per-dataset intervention eligibility as data.
- **`ceed_interventions`** — region selection, inpainting, control matching, text substitution. Content-addresses its outputs.
- **`ceed_teacher`** — hooked forward passes, artefact extraction, the CEA ablation engine, ECC recomputation, the correctness pass.
- **`ceed_modes`** — attribution clustering into Modes, and the Phase 0.0–0.4 analyses.
- **`ceed_student`** — Student wrapper, removable probes, the backbone, the auxiliary signal components, Group configurations, the training loop.
- **`ceed_eval`** — the `lmms-eval` adapter, the grounding-selectivity protocol, in-loop validation, Phase 1 table generation.

### The single entry point

`run_group` takes a fully-resolved Group configuration and returns a run record. Everything the plan calls a Group — B0 through E4, at any seed, under either parameter-efficiency mode — is one call. Data loading, store reads, training, evaluation, and metric emission all sit below it. This is the highest seam in the system and the primary place behaviour is tested.

### Configuration

Pydantic v2 models composed from YAML overlays: a base, a student layer, a Group overlay, and optionally a Phase 2 variant. Configuration serialises to canonical JSON; its hash is simultaneously the run identifier and, for the extraction subset of fields, the extraction fingerprint. Group definitions are validated at load, not at hour six.

### Auxiliary signal contract

Every Group is the shared backbone plus zero or more auxiliary signals. An auxiliary signal declares which artefact kinds it requires (so a missing extraction fails fast), which forward views it needs (original only, or original plus both intervened), and contributes a per-token loss that the trainer gates by coupling strength and scales by the warm-up schedule. B0 has none; E4 has three. Adding a Group means adding a component and a YAML overlay.

### Layer mapping

Represented as an explicit first-class object, not an implicit convention, because C2 is defined as a *deliberately mismatched* mapping and Phase 2 requires reporting variance across two alternatives. A proportional mapping is the initial placeholder that unblocks B3; the probe-based mapping replaces it once B2 has trained, and B3 is re-run.

### Tensor shapes

Cross-package tensors carry `jaxtyping` shape annotations naming their axes — attribution vectors as `(tokens, layers, probed_experts)`, ECC tensors as `(tokens, regions, layers, probed_experts)`, selectivity profiles as `(tokens, regions)`. Under pytest, `beartype` enforces them at runtime. This is the primary guard against the rank-confusion errors that would otherwise produce silently wrong science.

### Store schema

One row per `(example_id, answer_token_index)`, with artefact kinds as columns and fixed-size binary encoding for vectors. Parquet shards, one per extraction worker. A store-level metadata record declares the artefact kinds present, the extraction fingerprint, and the source corpus manifest.

## Testing Decisions

### What makes a good test here

A good test asserts a property of the pipeline's *output* — a number in a run record, a row in the store, a bit-exact invariant of an ablation — and says nothing about how that output was computed. In this codebase there is a second criterion that matters more than usual: **a good test can distinguish "wrong number" from "no number".** Tests that only prove the code runs are close to worthless, because everything here runs.

### Seams

Two, and the second exists only because the first cannot express it.

**Primary seam — `run_group`.** Behaviour is driven end-to-end: a tiny corpus, a tiny synthetic teacher, a tiny student, a handful of steps, asserting on the returned run record. This covers Group composition, the backbone, auxiliary signal wiring, the store contract, the training loop, evaluation, and metric emission. Adding an auxiliary signal or a Group should require no new seam.

**Secondary seam — the CEA ablation engine's numerical invariants.** Properties like "replacing an expert with its own output is bit-exact identity" and "the sum of expert contributions plus the shared dense path equals the full FFN output" are not observable in a run record; they are observable only at the layer boundary. This seam is justified precisely because these invariants are the ones whose violation produces plausible-looking wrong science.

### The tiny synthetic teacher

Both seams rest on a purpose-built fixture: a 2-layer, 8-expert, top-2, hidden-64 model with the *same structure* as the real teacher — hybrid layers with a shared dense path, and a router parameterised with input scale and per-expert scale. It runs on CPU in milliseconds. Without it, every meaningful test needs 49 GB and a GPU, and therefore will not be run.

### Tiers

- **Fast (default)** — CPU only, synthetic fixture, seconds. Both seams. Runs in CI and on every change.
- **`@pytest.mark.gpu`** — the real teacher and student on one or two examples, end-to-end extraction and a few training steps. Run manually and nightly.
- **Golden files** — frozen attribution output for two fixed examples. A golden diff means either a bug or a deliberate change to the extraction fingerprint, and the fingerprint says which.

### What gets tested where

At the primary seam: Group composition and the shared backbone; auxiliary signals declaring and receiving the artefact kinds they need; a Group whose required artefacts are absent failing immediately; the run record recording parameter-efficiency mode, seed, layer mapping, and configuration hash; warm-up scheduling; coupling-strength gating selecting the intended tokens; probes being removable with the resulting model architecturally identical to the base Student.

At the secondary seam: expert-with-itself identity; expert contributions plus shared dense path summing to the FFN output; ablation at token *t* leaving tokens before *t* bit-identical; mean-of-active replacement preserving residual norm within tolerance; near-miss selection ordering by effective combine weight rather than raw logit.

Store contract tests sit at the primary seam via round-trips: write, read, assert identical arrays and identical keys; assert that two different extraction configurations produce different store roots; assert that a cached answer-token index aligns with the Student's cross-entropy target index on a hand-checked example.

Intervention tests assert properties of the produced pairs — a control region matches its relevant region in area within tolerance and does not overlap it; an intervened image is content-addressed and reproducible; a document text-substitution changes only the intended span.

### Prior art

None — this is a greenfield repository. These conventions are the prior art for everything that follows.

## Out of Scope

- **Phase 2 in its entirety.** Matched-FLOPs re-runs, held-out generalisation, layer-mapping robustness, and inference verification are follow-on work. The layer mapping is built as a first-class object so Phase 2 does not require rework, but no Phase 2 experiment is delivered here.
- **Baseline B6** — the DIITO-style interchange-intervention comparator. The plan already marks it budget-permitting and expects it to be handicapped by the architecture gap.
- **Thinking mode.** Disabled everywhere; the Student is never trained to think.
- **ChartQA interventions.** ChartQA trains and evaluates but is never intervened on.
- **Running Phase 1 to completion locally.** The Volta box hosts development, teacher extraction, and smoke tests. Headline Groups require A100/H100-class hardware, and neither the cluster choice nor its launcher is decided here.
- **The specific inpainting model and the Mode clustering algorithm and cluster count.** Both are deferred until the VA-OPD paper has been read and Phase 0.0/0.2 data exists; the plan's range of 8–32 Modes is not narrowed in advance.
- **Publication artefacts.** Figures, tables for the paper, and the related-work positioning are downstream of results.

## Further Notes

**The two risks worth watching, in order.**

The first is fp16 stability on Volta. Both checkpoints ship bf16; sm70 has no bf16, and Gemma models have a reputation for fp16 overflow. The walking skeleton is deliberately shaped to hit this in the first days — loading the Student in fp16 and getting a coherent answer to one question is the point of B0, not an incidental step. If fp16 proves unstable, teacher extraction moves to fp32 at roughly double the memory and time.

The second is that the shared dense path attenuates CEA into the noise floor. Because every teacher layer is hybrid, ablating one of eight experts leaves the dense path rebuilding the residual stream for free. Phase 0.0 measures this before any extraction at scale. It is the single result most likely to invalidate the thesis, and it costs a sweep over 200 examples to learn.

**On the build order.** The researcher's stated ordering — baselines, then Phase 0, then controls and experimental Groups — appeared to conflict with the fact that B3 and B5 consume the three CEA layers, which Phase 0.0 chooses. It dissolves by over-caching: effective combine weights at all 30 layers and hidden states at 6 candidates cost ~14 GB against 294 GB free, making the three-layer choice a selection at training time rather than a re-extraction. The ordering therefore holds exactly as stated, and this is why ADR 0001 records over-caching as a deliberate consequence rather than an accident of convenience.

**On honest nulls.** The plan anticipates that E4 ≈ E3 is a likely outcome and intends to report coupling as a null result. That is only publishable if the null is interpretable, which is the whole reason ADR 0005 forbids LoRA from the headline table. The infrastructure's job is to make the plan's negative outcomes as trustworthy as its positive ones.
