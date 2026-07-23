# 08 — Remaining baselines: B3, B4, and B5

**What to build:** The baseline group completes. Three more Groups train and report numbers, and between them they prove the auxiliary-signal contract holds for three quite different kinds of signal — a cached per-expert vector, a cached hidden state under a layer mapping, and a second teacher pass over degraded images.

B5 is the one to build first: it is the cheapest real auxiliary signal and it is what demonstrates that adding a Group is a component plus a YAML overlay rather than a new training script.

**Blocked by:** 06, 07

**Status:** ready-for-agent

- [ ] **B5** distils the teacher's effective combine weights, and adding it required only a new auxiliary-signal component and a configuration overlay
- [ ] A layer mapping is a first-class object rather than an implicit convention, because a deliberately mismatched mapping is a control Group and Phase 2 reports variance across alternatives
- [ ] **B3** performs hidden-state projection distillation under a proportional placeholder mapping, supervising answer tokens only — the same token set as every other Group (see ADR-0001 and plan amendment A5)
- [ ] **B4** reproduces VA-OPD: teacher passes over globally degraded images producing per-token reweighting. The reproduction is checked against the source paper rather than inferred from its description in the plan
- [ ] All three run through `run_group` and report accuracy
- [ ] The baseline group is complete: B0 through B5 all have numbers, from frozen reusable checkpoints
