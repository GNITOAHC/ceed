# 17 — The grounding-selectivity evaluation protocol

**What to build:** The plan's own novel evaluation contribution. A researcher can measure any Student's relevant-versus-control effect ratio on held-out interventions — the same shape of measurement made on the teacher in Phase 0.4, now made on the Student.

This is the metric the central E2-versus-B4 claim rests on: that global degradation cannot teach invariance to irrelevant regions but matched control regions can. Without it that claim has no number.

**Blocked by:** 05, 12

**Status:** ready-for-agent

- [ ] A held-out set of intervention pairs is reserved and never seen in training
- [ ] For a given Student, the per-token output effect of the relevant intervention and of the control intervention are measured and reported as a ratio
- [ ] The protocol runs against any Student — untrained, baseline, or experimental — so B0 through E4 are all comparable on it
- [ ] Hallucination benchmarks are wired into the evaluation harness alongside accuracy, so the grounding claim is supported by both a bespoke and a standard measure
- [ ] The protocol is documented well enough to be reproduced from the paper alone
