# CEED v2: Causal Expert–Evidence Distillation
### Revised research plan for distilling Gemma-4-26B-A4B-it into Gemma-4-E4B-it
*(v2: repositioned against attribution-KD precedents; adds detailed CEA/ECC definitions, a step-by-step experiment plan with group design and expected outcomes, and a parallel execution schedule.)*

---

## 1. Thesis

CEED transfers knowledge from a sparse MoE vision-language teacher (Gemma-4-26B-A4B-it, ~26B total / 4B active parameters) into a compact student (Gemma-4-E4B-it, effective 4B parameters). The pair is approximately matched in active inference compute, isolating the question:

> Does the knowledge stored in a MoE teacher's full parameter set — specifically its *causally verified* division of computational labor and the visual evidence driving it — transfer to a student running at the same active-compute budget, when that structure is made an explicit training signal?

CEED does **not** distill router probabilities. Router gating is a prediction made before computation, distorted by load balancing and frequency bias; measured attribution is what the computation actually did. This distinction is now independently documented (§3). CEED distills two causally grounded objects:

1. **Causal Expert Attribution (CEA):** per token, the measured output effect of virtually ablating each activated and near-activated teacher expert.
2. **Evidence–Computation Correspondence (ECC):** how the attribution profile reorganizes under targeted, region-level image interventions with matched controls.

All CEED machinery (probes, projection heads) is removed after training. The deployed E4B student is architecturally unchanged: zero added inference cost.

---

## 2. CEA and ECC: precise definitions

### 2.1 Causal Expert Attribution

For each answer token t and each selected teacher MoE layer m (early / middle / late):

1. Run the teacher normally; record the activated expert set, gating weights, and the log-probability of the emitted token.
2. For each expert e among the top-k activated experts **plus the next 2–3 near-miss experts** by gating score: re-run the forward pass from layer m with expert e's contribution removed (zeroed, or replaced by the mean expert output to preserve residual-stream statistics), holding all other tokens' routing fixed. For near-miss experts, additionally measure the *gain* from force-activating them once.
3. Attribution of expert e = change in output log-probability (optionally also a top-k distribution divergence).

The result is a short per-(token, layer) vector: a measured profile of which experts did the work. Near-miss experts are included deliberately: the dormant-expert literature shows causally decisive experts can sit just below the activation threshold, and measuring only activated experts would inherit the router's blind spots.

**Why attribution beats routing as a target.** (a) Empirical: router probabilities and token-usage frequency correlate only weakly with expert-wise ablated ΔNLL; frequently activated experts can be nearly removable, and removing some experts reduces loss (arXiv:2606.18304, Fig. 2; similarly STEP's loss-impact scoring critique of routing-frequency heuristics). (b) Semantic: attribution vectors are comparable across contexts; expert IDs are not — two tokens with similar attribution profiles received similar functional treatment, whereas shared expert IDs may reflect load-balancing accidents. (c) Denoising: an expert that fires but contributes nothing gets high routing weight and near-zero attribution; the student should not spend capacity encoding it.

**Functional modes.** Attribution vectors are clustered corpus-wide, per layer, into 8–32 modes. Labels (e.g., fine-grained visual extraction, numeric manipulation, surface realization) come from post-hoc analysis, never assumption. Modes are the supervision unit for the student probe: coarser and more stable than raw vectors, immune to expert-ID permutation.

### 2.2 Evidence–Computation Correspondence

CEA is static; ECC adds the causal link to the input.

**Interventions per example:** (i) relevant-region counterfactual — the answer-critical region removed by inpainting (preferred over masking; avoids rectangle artifacts), identified via annotations where available, else teacher attribution verified by the teacher's own output effect; (ii) matched irrelevant-region control — identical procedure, comparable size/texture, unrelated region; (iii) optionally for OCR/document tasks, text-substitution counterfactuals (change a printed number), which have the cleanest causal semantics.

**The ECC object:** recompute CEA under each intervention. Per token, a tensor over (region j, layer m, expert e): the change in expert e's measured contribution when evidence j is removed. Derived summaries:

- **Coupling strength** per (token, region): magnitude of attribution reorganization, paired with the output-level effect of the same intervention.
- **Selectivity profile** per token: the ordered vector of output effects across (relevant, control) — desired shape: large effect / near-zero effect.
- **Four-cell diagnosis** (upgrading the original RACD table): crossing *output change* with *attribution change* fixes the routing version's ambiguity — "attribution reorganized, output stable" is genuine internal compensation, not router noise. The strong-coupling cell (both change, relevant intervention only) selects tokens for the coupling loss; the control mask defines invariance.

---

## 3. Precedent map (post-check, v2)

**Claims that are taken (cite and differentiate):**

- Sparse-to-dense distillation: OneS (arXiv:2201.10890); MoE→dense pruning + KD (arXiv:2605.28207).
- Inactive-expert knowledge in KD: Every Expert Matters (arXiv:2502.12947).
- Counterfactual image conditions as VLM distillation signal: VA-OPD (arXiv:2605.21924, global degradation → scalar per-token reweighting); Vision-OPD (crop-conditioned privileged teacher).
- Router-to-router distillation (MoE students only): TGR-MoE (arXiv:2604.21330), B-Distill (AAAI 2026), PADD (arXiv:2606.10369).
- Counterfactual expert impact as an inference-time metric: CoR (arXiv:2604.14246) — never a training/transfer signal.
- **[New in v2] Input-attribution matching in KD:** AD-KD (arXiv:2305.10010) matches teacher/student input-token attribution maps; Temporal Saliency Distillation (arXiv:2601.04263) derives perturbation-based input saliency and trains the student to match it. **Consequence:** CEED's selectivity-profile loss is an *extension* of attribution-KD — to region-level visual evidence, with explicit relevance labels and invariance supervision — not a standalone contribution. Contribution list revised accordingly (§8).
- **[New in v2, final check] Causal-abstraction distillation:** DIITO — "Causal Distillation for Language Models" (Wu, Geiger et al., NAACL 2022; arXiv:2112.02505) — augments distillation with interchange intervention training so the student imitates the teacher's *causal computation process*, becoming a causal abstraction of the teacher. This owns the general idea of "distill causal structure, not just outputs and states," and even the phrase "causal distillation." **Consequences:** (a) DIITO and the IIT / causal-abstraction literature (Geiger et al.) must anchor the related-work section — CEED is positioned *within* this family as its extension to MoE module structure, multimodal input interventions, and the MoE→dense setting; (b) CEED's coupling-consistency term is framed as causal-abstraction-style supervision using input-evidence interventions and expert ablations rather than activation interchanges; (c) the paper avoids "causal distillation" as an unqualified label for the whole method. Mechanical differences preserved: DIITO intervenes on *activations* (swapping hidden states between inputs) in layer-aligned dense BERT pairs and matches counterfactual outputs; CEED intervenes on *inputs* (image regions) and *modules* (experts), matches attribution structure, and requires no layer-aligned activation swapping — which would be ill-defined across the MoE→PLE architecture gap.

**Claims that support CEED (cite as motivation):**

- Routing–attribution misalignment: attribution-guided MoE pruning (arXiv:2606.18304); STEP loss-impact scoring (OpenReview). Phase 0.1 becomes a replication-plus-extension on a multimodal MoE, lowering risk.
- Dormant experts / correlational routing: CoR (arXiv:2604.14246); Routing Distraction in multimodal MoE (arXiv:2604.08541).
- Per-token expert-ablation maps as interpretability (small vision MoE): MoE-MAE (arXiv:2509.10919) — analysis only, no transfer.
- Routing habits transfer implicitly through vanilla KD: Shadow-MoE (arXiv:2510.16968) — motivates the C1 control and the claim that *explicit* structure supervision must beat implicit leakage.

**Claims that remain open (CEED's novelty core, restated after final check):**

1. Module-level (expert) causal attribution as a **distillation target** — no precedent in any modality. (DIITO aligns hidden-state slices via interchange; it never attributes to or supervises expert/module structure.)
2. The **coupling** of input-evidence interventions with internal expert-attribution reorganization as a transfer signal — DIITO's interventions are activation swaps with no notion of input evidence; VA-OPD's input interventions carry no notion of internal structure. CEED occupies the intersection neither touches.
3. The **compute-matched multimodal MoE→router-free-dense** setting with removable representational probes.

Optional baseline B6 (budget-permitting): a DIITO-style interchange-intervention objective adapted to this pair, as the strongest causal-abstraction comparator. Expected to be handicapped by the teacher–student architecture gap (activation interchange requires a layer/width alignment that MoE→PLE lacks); if infeasible, this is documented as the reason input/module-level interventions were chosen over activation-level ones.

---

## 4. Teacher–student pair

Gemma-4-26B-A4B-it: open-weights multimodal MoE; router internals and expert outputs accessible, so virtual ablation is feasible offline; instruction-tuned with thinking mode, yielding rationale spans for analysis. Gemma-4-E4B-it: on-device sibling, effective 4B via Per-Layer Embeddings, shared tokenizer lineage (no cross-tokenizer alignment). Matched active compute makes gains attributable to transferred structure, not capacity. Caveat carried from v1: E4B's PLE/hybrid-attention architecture makes teacher→student layer mapping nontrivial; it is selected empirically (probe-based, under KD-only training) and robustness across two alternative mappings is reported.

---

## 5. Student objectives

Backbone for every group: supervised CE on answers + top-k logit KD (with log-sum-exp normalization bookkeeping). CEED adds three terms, introduced on a warm-up schedule (CE+KD first), with small weights (≈0.05–0.1 of CE) gated per token by teacher-measured coupling strength:

1. **Mode probing (removable):** linear probes on selected student layers predict per-token teacher mode assignments (soft targets); richer variant predicts a compressed attribution embedding. Deleted after training.
2. **Selectivity-profile matching (behavioral):** run the student on original + intervened images; match the *vector* of per-token output effects across regions to the teacher's with a robust loss plus a ranking constraint (relevant-region effect must exceed control-region effect by the teacher's margin). Framed as attribution-KD extended to region-level evidence with invariance supervision (§3).
3. **Coupling consistency:** on strong-coupling tokens only, the shift in the student's probed mode embedding between original and intervened runs must track the teacher's attribution shift. The component whose ablation must show super-additivity for the coupling story to hold.

---

## 6. Step-by-step experiment plan

### Phase 0 — Teacher characterization (go/no-go; standalone publishable)

Data: ~5k examples across DocVQA, ChartQA, one grounded-VQA set. No training.

| Step | Procedure | Expected outcome | Failure signal → action |
|---|---|---|---|
| 0.1 | Per-token Spearman correlation: gating score vs. measured attribution, three layers | Weak-to-moderate (ρ ≈ 0.3–0.6), replicating pruning-literature misalignment on a multimodal MoE | ρ > 0.9 everywhere → CEA ≈ routing; abandon pivot, fall back to router-signal variant (B5 design) with reduced claims |
| 0.2 | Mutual information: mode assignment vs. token-function category, compared against router-cluster MI and hidden-state-cluster MI at equal cluster count | CEA modes carry more compact task structure than both comparators | CEA MI ≤ router MI → attribution adds cost without information; stop |
| 0.3 | Mode stability: across data subsets, paraphrases, same-token/different-image pairs | Modes track function, not token identity | Modes predictable from token identity alone → restrict supervision to image-dependent tokens; re-test |
| 0.4 | ECC responsiveness: attribution reorganization, relevant vs. control mask; correlation of reorganization magnitude with teacher correctness | Relevant ≥ 2–3× control; magnitude predicts correctness/hallucination | No gap → intervention artifacts; fix inpainting pipeline before any E-group training |

Go criterion: 0.1 misalignment material, 0.2–0.3 pass, 0.4 passes after at most one pipeline iteration.

### Phase 1 — Controlled distillation comparison

All groups: identical corpus (30–50k examples; interventions on ~30%: one relevant + one matched control each), identical student, identical CE + logit-KD backbone, matched auxiliary-loss budget where applicable. Only the auxiliary signal varies.

| Group | Auxiliary signal | Role | Depends on |
|---|---|---|---|
| B0 | none (zero-shot E4B) | floor | nothing |
| B1 | none (SFT, no teacher) | isolates teacher value | corpus only |
| B2 | CE + logit KD only | **primary baseline** | teacher logits |
| B3 | + hidden-state projection KD | standard alternative | teacher hidden states |
| B4 | + VA-OPD reproduction (global degradation, token reweighting) | **critical external baseline** | teacher passes on degraded images |
| B5 | + router-score distillation (original RACD design) | internal ablation: routing vs. attribution | router logits (trivial extraction) |
| C1 | + probe on **shuffled** mode labels | control: generic auxiliary-task regularization | mode labels from Phase-0 pipeline |
| C2 | + probe with mismatched layer mapping | control: layer-correspondence sensitivity | mode labels |
| E1 | + CEA mode probing | experimental component 1 | CEA extraction |
| E2 | + selectivity-profile matching | experimental component 2 | intervention pipeline |
| E3 | E1 + E2 | additivity | both |
| E4 | E1 + E2 + coupling consistency (full CEED) | full method | ECC extraction |

**Expected outcomes and decision logic:**

- **E1 > B5 > B2** on task accuracy, with the E1–B5 gap concentrated on tokens with high Phase-0 routing–attribution divergence. Expected magnitude: +0.5–1.5 aggregate, +2–3 on OCR/reasoning-heavy subsets. E1 ≈ B5 → divergence exists but does not matter for transfer; publish as characterization + negative transfer finding.
- **E2 > B4** on grounding selectivity (student's own relevant-vs-control effect ratio, held-out interventions) and hallucination (POPE, HallusionBench), even at comparable accuracy. Expected: selectivity ratio +30–50% over B4; unsupported-claim rate −2 to −5 points. Rationale: global degradation cannot teach invariance to irrelevant regions; control masks can. E2 ≤ B4 across the board → region interventions not worth their cost; keep E1 only.
- **E1 ≫ C1** — non-negotiable. Shuffled-label parity means the gain is auxiliary-task regularization, not knowledge transfer. This is the first control reviewers will demand (and Shadow-MoE's implicit-leakage finding makes it doubly necessary).
- **E4 > E3** (coupling super-additivity): smallest, least certain effect. E4 ≈ E3 → ship E3; report coupling as a null result. The paper survives on E1/E2 + Phase 0.
- **Mechanistic link:** probes retrained post hoc on the final E4 student recover teacher modes at ≥60% top-1 (vs. near-chance on B2), and per-example decodability correlates with accuracy gains.

### Phase 2 — Fairness and robustness

1. **Matched-FLOPs re-run:** grant B2 the compute spent on ablations/interventions as extra vanilla-KD data. Expected: E4's edge shrinks by roughly a third but survives. If it vanishes → CEED is a data-generation method, not a signal method; reframe honestly.
2. **Held-out generalization:** gains persist (attenuated) on general VQA outside the training task families; task-local-only gains would indicate modes encode dataset style.
3. **Layer-mapping robustness:** repeat E1 under two alternative mappings; report variance.
4. **Inference verification:** byte-identical E4B architecture; zero latency delta.

---

## 7. Execution schedule and parallelism

The phase numbering describes *logical* dependency for the scientific argument, not a serial compute schedule. The actual dependency structure:

**Can start immediately, in parallel with Phase 0:**
- **B0–B4**: consume no CEA/ECC data — only the corpus, teacher logits, teacher hidden states (B3), and globally degraded teacher passes (B4). They are required comparators regardless of Phase 0's verdict, so running them early risks no wasted compute even if Phase 0 fails its go-criteria.
- **B5**: router-logit extraction is a trivial byproduct of the same teacher forward passes that generate B2's logits; no ablation studies needed.
- **Shared infrastructure**: the intervention-generation pipeline (inpainting, region selection, control matching) serves both Phase 0.4 and E2/B4 — build once, first.
- Corpus assembly, teacher logit caching, layer-mapping probe study (runs on B2's checkpoints as they train).

**Gated on Phase-0 *pipeline* (not its conclusions):**
- **C1, C2, E1**: need mode labels, i.e., the CEA extraction and clustering code — available as soon as Phase 0.1–0.2's extraction runs, before the analysis is interpreted.
- **E2**: needs the validated intervention pipeline (Phase 0.4's artifact check should pass first, or E2's signal may be poisoned by mask artifacts — this is the one genuine serialization point).

**Gated on Phase-0 *conclusions* (the true go/no-go):**
- **E3, E4** (full training runs) and the final experimental-group budget. If 0.1 or 0.2 fails, the E-track stops and the completed B-track becomes the comparison set for the fallback (router-signal) study.

Practical schedule: weeks 1–2 build shared pipelines + launch B0–B5; weeks 2–4 Phase-0 analysis while baselines train; weeks 4–5 launch C1/C2/E1 (and E2 once 0.4 passes); weeks 6–12 E3/E4 conditional on go; weeks 12–16 Phase 2. Baseline checkpoints are frozen and reused across all later comparisons, so no baseline is ever retrained.

---

## 8. Revised contributions and novelty statement

Contributions (v2, precedent-adjusted):

1. **Causal-attribution characterization of a production multimodal MoE** (Gemma-4-26B-A4B): routing–attribution divergence, functional-mode structure, and evidence-responsiveness — extending pruning-literature misalignment findings to the multimodal setting and to intervention dynamics.
2. **Causal expert attribution as a distillation target** — the first use of module-level measured attribution (rather than router scores, outputs, or input saliency) as transfer supervision.
3. **Evidence–computation coupling transfer**: distilling how internal functional structure reorganizes under relevant-vs-control visual interventions.
4. **Inference-free transfer** into a compute-matched, router-free on-device student via removable probes.
5. A **grounding-selectivity evaluation protocol** (relevant/control effect ratios on held-out interventions).

Explicitly *not* claimed (v2 change): perturbation-based attribution matching per se (AD-KD, TSD precede it); counterfactual image signals per se (VA-OPD); sparse-to-dense distillation per se.

> Defensible statement: *We show that a multimodal MoE's router is an unreliable witness to its own computation and distill what the model measurably does instead: per-token causal expert attributions and their reorganization under targeted visual-evidence interventions. Extending attribution-based distillation from input saliency to internal module structure, CEED transfers this evidence–computation correspondence into a router-free, compute-matched on-device student at zero added inference cost.*

---

## 9. Risks (carried from v1, unchanged in substance)

Ablation cost (mitigate: three layers, top-k+2 experts, activation caching, answer tokens only, validated first-order approximation); attribution noise (distill modes, not raw vectors; report Phase-0 reliability); region-identification errors (teacher-verified effect gaps; annotation-backed datasets for core results); PLE layer mapping (empirical selection + robustness report); teacher errors (correctness gating); auxiliary-loss interference (warm-up, small gated weights, early stopping on KD-only validation).

---

## References

- OneS — arXiv:2201.10890 · Every Expert Matters — arXiv:2502.12947 · MoE→dense pruning+KD — arXiv:2605.28207
- VA-OPD — arXiv:2605.21924 · Vision-OPD — 2026 · TGR-MoE — arXiv:2604.21330 · PADD — arXiv:2606.10369 · B-Distill — AAAI 2026
- Shadow-MoE — arXiv:2510.16968 · CoR / dormant experts — arXiv:2604.14246 · Routing Distraction — arXiv:2604.08541
- Attribution-guided MoE pruning — arXiv:2606.18304 · STEP — OpenReview · MoE-MAE ablation maps — arXiv:2509.10919
- **[v2]** AD-KD — arXiv:2305.10010 · **[v2]** Temporal Saliency Distillation — arXiv:2601.04263
- **[v2 final]** DIITO, Causal Distillation for Language Models — arXiv:2112.02505, NAACL 2022 · IIT / causal abstraction — Geiger et al. 2021–2024
