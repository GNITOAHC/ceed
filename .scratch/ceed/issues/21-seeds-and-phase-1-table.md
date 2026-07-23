# 21 — Seeds and the Phase 1 comparison table

**What to build:** The result. Every Group's numbers are collected into the comparison table, with error bars on the Groups whose deltas carry claims, and the plan's decision logic evaluated against it rather than eyeballed.

The plan predicts aggregate gains of half a point to one and a half points. At one seed per Group that is inside the noise band, and the non-negotiable claim — that E1 substantially exceeds C1 — would be unfalsifiable.

**Blocked by:** 08, 15, 19, 20

**Status:** ready-for-agent

- [ ] Three seeds are run for the Groups whose deltas carry claims — B2, B5, C1, E1, E3, E4 — and one seed elsewhere (see plan amendment A9)
- [ ] All headline runs are full fine-tunes; the table generation refuses to include any run whose record says LoRA
- [ ] The comparison table is generated from the run records on disk, so the result is a computation over files rather than a dashboard reading
- [ ] Error bars are reported for the multi-seed Groups
- [ ] The plan's decision logic is evaluated explicitly and its verdicts recorded: E1 against B5 against B2, with the gap examined on high-divergence tokens; E2 against B4 on grounding selectivity and hallucination; E1 against C1; E4 against E3
- [ ] Where an expected effect fails to appear, the plan's stated fallback is applied and the null reported honestly
- [ ] Every cell in the table is traceable to a run record and its configuration hash
