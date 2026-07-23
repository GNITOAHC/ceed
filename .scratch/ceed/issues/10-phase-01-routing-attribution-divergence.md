# 10 — Phase 0.1: routing–attribution divergence

**What to build:** The premise the whole method rests on gets a number. A researcher can state, with evidence, how badly the teacher's router predicts what its experts measurably did.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] Per-token Spearman correlation between effective combine weight and measured attribution is computed for each of the three chosen layers
- [ ] The comparator is the effective combine weight — the quantity actually multiplying the expert's output — not a raw router logit, so the result is the strongest form of the null hypothesis (see plan amendment A3)
- [ ] The result is reported against the plan's expectation of weak-to-moderate correlation, and its go/no-go verdict is recorded rather than argued
- [ ] The distribution of divergence across tokens is retained, not just its aggregate, because later comparisons predict gains concentrated on high-divergence tokens
- [ ] The analysis runs as a query over the artifact store rather than as bespoke array code
