# CEED

CEED (Causal Expert–Evidence Distillation) transfers a sparse mixture-of-experts vision-language teacher's measured division of computational labour — and the visual evidence driving it — into a compact dense student running at the same active-compute budget.

This glossary fixes the vocabulary used across the research plan, the code, and the paper. It is a glossary only; implementation decisions live in `docs/adr/`.

## Language

### Models

**Teacher**:
`gemma-4-26B-A4B-it`. The sparse MoE vision-language model whose internal structure is measured and distilled.
_Avoid_: source model, large model

**Student**:
`gemma-4-E4B-it`. The dense, router-free model that receives the distilled signal and is the deployed artefact.
_Avoid_: target model, small model, dense model

**Hybrid layer**:
A teacher decoder layer, in which a shared dense path and a sparse expert mixture both contribute to the residual stream. Every one of the teacher's layers is hybrid.
_Avoid_: MoE layer (misleading — it implies the FFN is purely sparse)

**Shared dense path**:
The always-on dense MLP inside a hybrid layer, which fires for every token regardless of routing. It is never an ablation target and is never called an expert.
_Avoid_: shared expert, always-on expert, dense expert

**Expert**:
One of the teacher's 128 per-layer routed feed-forward blocks. Only experts are ablation targets.

**Layer mapping**:
The chosen correspondence between teacher layers and student layers, used to place probes and hidden-state supervision. Distinct from the teacher's own layer indices.

### Attribution

**Causal Expert Attribution (CEA)**:
The measured change in an answer token's log-probability when a single expert's contribution is replaced, holding all other tokens' routing fixed. Attribution is measured, never predicted.
_Avoid_: expert importance, expert saliency, expert score

**Attribution vector**:
The per-(answer token, layer) vector of CEA values across the probed experts.

**Probed expert**:
An expert included in the attribution vector for a token: the activated experts plus the near-miss experts.

**Near-miss expert**:
An expert ranked just below the activation threshold by effective combine weight, included in the attribution vector so that measurement does not inherit the router's blind spots.
_Avoid_: dormant expert (that term names a phenomenon in the literature, not a selection rule), inactive expert

**Effective combine weight**:
The weight actually multiplying an expert's output in the residual stream. This is what "gating score" means throughout CEED, and it is the comparator against which attribution's divergence is claimed.
_Avoid_: gating weight, router probability, routing score (all ambiguous between raw logit, softmax weight, and scaled weight)

**Routing–attribution divergence**:
The extent to which effective combine weight fails to predict CEA. The premise the whole method rests on.

**Mode**:
A cluster of attribution vectors, computed per layer over the corpus, serving as the supervision unit for the student's probe. Modes are coarser and more stable than raw attribution vectors, and are invariant to expert-ID permutation. Labels for modes are assigned by post-hoc analysis, never by assumption.
_Avoid_: expert cluster, functional class, expert role

### Evidence

**Intervention**:
A targeted edit to an input image that removes or alters a region, used to probe how attribution reorganises. Always applied in relevant/control pairs.
_Avoid_: perturbation, augmentation, corruption

**Relevant region**:
The answer-critical region of an image, identified from dataset annotations wherever possible.
_Avoid_: evidence region, salient region, ground-truth region

**Control region**:
A region matched to a relevant region in size and texture but unrelated to the answer. Its purpose is to define invariance, not to be a second relevant region.
_Avoid_: irrelevant region, distractor, negative region

**Text-substitution counterfactual**:
An intervention that alters printed text in a document image rather than removing a region. The cleanest causal semantics available, and applicable only to document tasks.

**Evidence–Computation Correspondence (ECC)**:
How a token's attribution profile reorganises under a given intervention, paired with the output-level effect of that same intervention.
_Avoid_: grounding signal, causal correspondence

**Coupling strength**:
The magnitude of attribution reorganisation for a (token, region) pair. Used to gate which tokens receive auxiliary supervision.

**Selectivity profile**:
The per-token ordered vector of output effects across the relevant and control interventions. The desired shape is a large relevant effect and a near-zero control effect.
_Avoid_: sensitivity profile, grounding score

**Four-cell diagnosis**:
The classification of a (token, intervention) pair by crossing whether the output changed with whether the attribution changed. The cell where both change under the relevant intervention selects tokens for coupling supervision.

**Strong-coupling token**:
An answer token falling in the both-changed cell of the four-cell diagnosis under its relevant intervention.

### Experiments

**Group**:
One arm of the controlled distillation comparison, identified by its code (B0–B5, C1–C2, E1–E4). All groups share a corpus, a student, and a backbone; only the auxiliary signal differs.
_Avoid_: run, experiment, condition, arm

**Backbone**:
The supervision every group shares: cross-entropy on gold answers plus top-k logit knowledge distillation.

**Auxiliary signal**:
The per-group supervision added on top of the backbone. The single independent variable of the comparison.

**Probe**:
A removable head trained on selected student layers to predict teacher mode assignments. Probes are deleted after training; the deployed student is architecturally unchanged.
_Avoid_: adapter, head, classifier

**Answer token**:
A token of the gold answer under teacher forcing. Every cached teacher artefact and every auxiliary loss is scoped to answer tokens.
_Avoid_: target token, output token, generated token (the teacher does not generate the attributed sequence)

**Correctness flag**:
A per-example boolean recording whether the teacher answers correctly under free generation, used to gate supervision away from the teacher's own errors.

**Artifact store**:
The on-disk record of everything measured from the teacher. It is the seam between teacher-side measurement and student-side training; the teacher is never loaded during student training.

**Extraction fingerprint**:
The content hash of the extraction configuration — layer set, ablation definition, combine-weight definition, thinking gate, model revision — which identifies an artifact store and makes artefacts from different definitions impossible to mix.

**Grounding selectivity**:
The evaluation protocol measuring a model's own relevant-versus-control effect ratio on held-out interventions. Reported for the student, mirroring the teacher's selectivity profile.
