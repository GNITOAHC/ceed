# 07 — B1 and B2: the training loop and shared backbone

**What to build:** A researcher can train the Student. Groups B1 (supervised fine-tuning, no teacher) and B2 (the primary baseline: cross-entropy plus top-k logit distillation) both run to completion and report numbers, and the machinery that every later Group plugs into exists.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] A training loop built on `accelerate` handles checkpointing and resumption, so a multi-day Group survives preemption
- [ ] The backbone — cross-entropy on gold answers plus top-k logit distillation — is implemented once and shared, so no Group can accidentally differ in it
- [ ] The auxiliary-signal contract exists: a signal declares which artefact kinds it requires and which forward views it needs, and contributes a per-token loss that the trainer gates and schedules. B1 and B2 attach zero signals
- [ ] Auxiliary terms are introduced on a warm-up schedule after the backbone, so they cannot destabilise early training
- [ ] LoRA and full fine-tuning are selected by one configuration flag, and the run record states unambiguously which was used, so a LoRA number can never reach the Phase 1 table (see ADR-0005)
- [ ] A LoRA run of B2 completes on the Volta box at toy scale
- [ ] B1 and B2 both run through `run_group` and report accuracy through the evaluation harness
- [ ] Baseline checkpoints are frozen and reusable, so no baseline is ever retrained for a later comparison
