# 19 — E3: additivity of Mode probing and selectivity matching

**What to build:** The Group that tests whether CEED's two components add. E3 is E1 and E2 composed, and if the auxiliary-signal contract is right it should be a configuration overlay attaching two existing components rather than any new machinery.

**Blocked by:** 14, 18

**Status:** ready-for-agent

- [ ] E3 attaches both the Mode probing and selectivity-matching components, with the auxiliary-loss budget matched against E1 and E2 individually
- [ ] Composing the two required no new training-loop code — if it did, the contract needs revisiting and that is worth recording
- [ ] The two components' warm-up schedules and per-token gating compose without one starving the other, verified rather than assumed
- [ ] E3 runs through `run_group`, reports accuracy and grounding selectivity
- [ ] The E3-versus-E1 and E3-versus-E2 deltas are computed, since additivity is what this Group exists to measure
