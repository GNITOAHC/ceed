# 09 — Real-teacher CEA and the Phase 0.0 sweep

**What to build:** The ablation engine is bound to the real teacher, and the first question CEED must answer gets answered: does the sparse path carry enough of the computation for expert attribution to be measurable at all?

Every teacher layer is a hybrid layer, so ablating one of eight experts leaves the shared dense path rebuilding the residual stream for free. If the dense path dominates, every measured change in log-probability is small and the whole method is computed over near-noise. **This is the single result most likely to invalidate the thesis, and it costs a sweep over ~200 examples to learn.**

**Blocked by:** 03, 06

**Status:** ready-for-agent

- [ ] The real teacher implements the hybrid-layer protocol, so the ablation engine works against it unchanged — and the invariants from ticket 03 hold on the real model for a couple of examples under the gpu test tier
- [ ] A sweep over all 30 teacher layers on ~200 examples reports, per layer, the shared dense path's share of feed-forward output norm
- [ ] The same sweep reports, per layer, CEA magnitude and across-expert diversity, with an explicit comparison against measurement noise
- [ ] The three CEA layers are chosen from the sweep result and recorded, replacing the plan's unspecified early/middle/late (see plan amendment A4)
- [ ] The sweep is reportable as a Phase 0 figure, so "why these three layers?" is answered by measurement
- [ ] Attribution vectors for the chosen layers are extracted into the artifact store under a fingerprint that captures the ablation definition
- [ ] Golden attribution output is frozen for two fixed examples, so a later refactor cannot silently change ablation semantics
- [ ] If the dense path dominates at all layers or CEA magnitudes sit at noise, the finding is recorded and escalated before any extraction at scale
