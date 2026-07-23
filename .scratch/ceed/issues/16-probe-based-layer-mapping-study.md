# 16 — Probe-based layer mapping study, and the B3 re-run

**What to build:** The placeholder mapping gets replaced by a measured one. The Student's architecture makes teacher-to-student layer correspondence genuinely non-obvious — 30 mixture-of-experts layers against 42 dense layers, with per-layer embeddings and shared key-value layers — so the mapping is selected empirically from trained baseline checkpoints rather than assumed.

This resolves the deadlock noted during planning: the mapping study needs a trained baseline, and the baselines needed a mapping. The proportional placeholder broke it; this ticket closes it.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Probe transfer is measured across candidate teacher-to-student layer correspondences using trained B2 checkpoints, under distillation-only training as the plan specifies
- [ ] The selected mapping is recorded as a first-class layer mapping object, alongside the proportional placeholder it replaces
- [ ] B3 is re-run under the selected mapping, and both B3 results are retained with their mapping named
- [ ] Two alternative mappings are retained as named objects, so the Phase 2 robustness report needs no rework
- [ ] Any Group already trained under the placeholder is identifiable from its run record
