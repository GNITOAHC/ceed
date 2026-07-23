# 13 — Phase 0.4: ECC responsiveness

**What to build:** The last Phase 0 criterion, and the true go/no-go. A researcher can state whether the teacher's internal computation actually reorganises when answer-critical evidence is removed — and reorganises materially more than when unrelated evidence is removed.

This is also the one genuine serialisation point in the plan: if the relevant-versus-control gap is weak because of inpainting artefacts rather than because of the teacher, any Group trained on selectivity would be learning noise. The intervention pipeline must be validated here before ticket 18 starts.

**Blocked by:** 09, 12

**Status:** ready-for-agent

- [ ] Attribution is recomputed under each intervention, producing the ECC object per token over region, layer, and expert
- [ ] Attribution reorganisation is compared between relevant and control interventions, against the plan's expectation that relevant exceeds control by a factor of two to three
- [ ] Reorganisation magnitude is correlated against teacher correctness, testing whether it predicts correctness and hallucination
- [ ] The go/no-go verdict is recorded against the plan's criterion
- [ ] A weak gap is diagnosed as an intervention artefact and the pipeline fixed before any experimental Group trains — which is possible precisely because every intervention here is annotation-backed (see plan amendment A8)
- [ ] **Phase 0 is complete.** All five verdicts — 0.0 through 0.4 — are recorded together with the evidence behind each
