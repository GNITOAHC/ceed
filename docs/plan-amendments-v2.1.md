# CEED plan amendments (v2 → v2.1)

Amendments to `ceed.md` forced by facts established while designing the implementation. Each is stated as a replacement for, or addition to, the numbered section it affects. They are recorded here rather than edited into `ceed.md` in place, because that document carefully tracks its own provenance (`[New in v2]`, `[v2 final]`) and the amendments should be visible as a distinct revision. Fold them in when convenient.

---

## A1 — §2.1: the teacher is a hybrid dense+sparse architecture

**Fact.** Every one of the teacher's 30 decoder layers contains an always-on shared dense MLP (2816→2112→2816) in parallel with the 128-expert top-8 mixture (2816→704→2816), summed into the residual stream via separate post-FFN layernorms. The plan is written as though the FFN is purely sparse.

**Amendment.** §2.1 must state that CEA is measured against an intact shared dense path, and that attribution magnitudes are therefore attenuated by construction: removing one of eight experts leaves the dense path rebuilding the residual stream for free. The paper cannot describe expert ablation as removing "the" computation at a layer.

**Consequence.** This is the largest single threat to the thesis and is measured first — see A4.

## A2 — §2.1: "the emitted token" → "the gold token under teacher forcing"

**Fact.** §2.1 defines attribution over the log-probability of the emitted token, implying free generation, while §5 defines the student backbone over gold answers, implying teacher forcing. These are different sequences and cannot both be the attributed one.

**Amendment.** Attribution is measured with the gold answer teacher-forced. A separate free-generation pass per example yields only a correctness flag, which implements the correctness gating already promised in §9. The limitation — attribution is measured on a distribution the teacher is not in at inference time — is stated explicitly.

## A3 — §2.1: "gating score" is the effective combine weight

**Fact.** The teacher's router carries both an input `scale` and a `per_expert_scale`. Raw router logits, softmax top-8 weights, and scale-adjusted weights are not monotone transforms of each other — `per_expert_scale` reorders experts. §6's Phase 0.1 Spearman correlation is undefined until one is chosen, and the choice moves the headline number.

**Amendment.** "Gating score" means the **effective combine weight**: the softmax top-8 weight multiplied by `per_expert_scale`, i.e. the quantity actually multiplying the expert's output. This is the strongest form of the null hypothesis and the hardest version of the claim to attack.

## A4 — §6: new Phase 0.0, ahead of Phase 0.1

**Amendment.** Insert a Phase 0.0 row into the Phase 0 table, run before anything else:

| Step | Procedure | Expected outcome | Failure signal → action |
|---|---|---|---|
| 0.0 | Sweep all 30 teacher layers on ~200 examples. Report (a) shared-dense vs sparse share of FFN output norm per layer, (b) CEA magnitude and across-expert diversity per layer. Fix the three CEA layers from the result. | Sparse path carries a material share of FFN output norm at some layers; CEA magnitudes are well above measurement noise there | Dense path dominates at all layers, or CEA magnitudes sit at noise → CEED's signal is structurally attenuated; reconsider before any extraction at scale |

This also replaces the plan's unspecified "early / middle / late" layer choice with a measured one, which pre-empts the certain reviewer question "why these three layers?" for a rounding error of compute.

## A5 — §6: B3 is answer-token scoped

**Fact.** Full-sequence hidden states for 3 layers over 40k examples at ~400 positions each are ~270 GB.

**Amendment.** B3's hidden-state projection KD supervises **answer tokens only**, as every other group does. Beyond the disk constraint, this is what makes §6's "matched auxiliary-loss budget" claim honest: with full-sequence supervision B3 would receive roughly fifty times the supervised positions of E1, confounding the B3–E1 comparison by volume rather than by signal type. The deviation from textbook hidden-state KD, and its consequence for comparability with published hidden-KD results, must be stated.

## A6 — §4: the student is 8B parameters

**Fact.** `gemma-4-E4B-it` is 8.0B actual parameters (16 GB bf16). "Effective 4B" describes active compute via Per-Layer Embeddings.

**Amendment.** §4's compute-matching claim survives unchanged — 4B active teacher versus 4B active student is the claim, and it holds. But the training-memory discussion must treat the student as an 8B model, and §7's schedule should reflect that full fine-tuning it is an A100/H100-class job.

## A7 — §4: thinking mode is out of scope for v1

**Amendment.** Strike "instruction-tuned with thinking mode, yielding rationale spans for analysis" as a v1 capability. The thinking gate is disabled for all extraction. Enabling it would require the student to be trained to think for the probes to have matching positions, roughly doubling scope for a benefit the plan describes only as being for analysis.

## A8 — §2.2 / §6: ChartQA is excluded from the intervention subset

**Fact.** DocVQA ships OCR words with boxes (and supports text-substitution counterfactuals); GQA ships scene-graph object boxes tied to the answer. ChartQA has no region annotations, so every ChartQA relevant region would come from the plan's weaker "teacher attribution verified by output effect" branch.

**Amendment.** DocVQA and GQA are the annotation-backed intervention core. ChartQA contributes to the training corpus and to accuracy evaluation but never enters the intervention subset in v1. This keeps every ECC result annotation-backed and, critically, means a weak Phase 0.4 relevant-versus-control gap can be diagnosed as an inpainting artefact rather than being confounded with region-identification error. ChartQA returns as a Phase 2 held-out generalisation set.

## A9 — §6: effect sizes require seeds

**Fact.** §6 predicts aggregate gains of +0.5–1.5 points. A single seed per group, or sampled decoding, puts the predicted effect inside the noise band, and "E1 ≫ C1" becomes unfalsifiable.

**Amendment.** Greedy decoding everywhere (overriding the checkpoints' shipped `do_sample: true`, temperature 1.0, top-p 0.95). Three seeds for the groups whose deltas carry claims — B2, B5, C1, E1, E3, E4 — and one seed elsewhere. Report error bars on those six.
