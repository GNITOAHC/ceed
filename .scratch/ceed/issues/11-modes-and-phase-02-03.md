# 11 — Modes, and Phase 0.2 / 0.3

**What to build:** Attribution vectors become Modes — the stable, expert-ID-permutation-invariant supervision unit the Student's probe will target — and the two Phase 0 criteria that decide whether Modes carry information worth distilling get their numbers.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] Attribution vectors are clustered per layer over the corpus into Modes, with the algorithm and cluster count chosen from the data rather than fixed in advance (the plan's range is 8–32)
- [ ] Mode assignments are written to the artifact store keyed the same way as every other artefact, so the Student's probe target is a join away
- [ ] **Phase 0.2:** mutual information between Mode assignment and token-function category is compared against a router-cluster baseline and a hidden-state-cluster baseline at equal cluster count, with the verdict recorded
- [ ] **Phase 0.3:** Mode stability is measured across data subsets, across paraphrases, and across same-token/different-image pairs, with the verdict recorded
- [ ] If Modes turn out predictable from token identity alone, that is reported and the plan's remedy — restricting supervision to image-dependent tokens — is costed
- [ ] Mode labels come from post-hoc analysis of what the clusters contain, never from assumption
