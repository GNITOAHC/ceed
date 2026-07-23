# 20 — ECC extraction and E4: coupling consistency

**What to build:** Full CEED. On the tokens where the teacher shows genuine evidence–computation coupling, the shift in the Student's probed Mode embedding between the original and intervened runs is trained to track the teacher's attribution shift.

The plan is candid that this is the smallest and least certain effect, and that E4 ≈ E3 is a likely outcome to be reported as a null. That null is only publishable if it is interpretable, which is why headline runs must be full fine-tunes rather than LoRA (see ADR-0005).

**Blocked by:** 13, 18

**Status:** ready-for-agent

- [ ] The four-cell diagnosis is computed per token and intervention, crossing whether the output changed with whether the attribution changed
- [ ] Strong-coupling tokens — the both-changed cell under the relevant intervention — are identified and stored, and the control mask defines where invariance is expected
- [ ] Coupling strength per token and region is derived and used to gate auxiliary supervision across all experimental Groups, not only this one
- [ ] The coupling-consistency loss applies only on strong-coupling tokens, matching the Student's probed embedding shift to the teacher's attribution shift
- [ ] E4 runs through `run_group`, reports accuracy and grounding selectivity
- [ ] Probes retrained post hoc on the final E4 Student recover teacher Modes, and per-example decodability is correlated against accuracy gains — the plan's mechanistic link
- [ ] If E4 ≈ E3, the null is recorded with the evidence that makes it interpretable: full fine-tuning, matched budget, matched token set
