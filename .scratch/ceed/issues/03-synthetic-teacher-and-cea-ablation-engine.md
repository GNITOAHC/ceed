# 03 — Synthetic teacher fixture and the CEA ablation engine

**What to build:** A researcher can compute a Causal Expert Attribution vector and trust it, because the ablation engine's numerical invariants are verified on every test run — on CPU, in milliseconds, without loading 49 GB of weights. This is the highest-risk logic in the repo and it has no dependency on the corpus, the artifact store, or the real teacher.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] A hybrid-layer protocol expresses what the ablation engine needs from a model: activated experts, effective combine weights, per-expert outputs, the shared dense path's output, and the ability to resume a forward pass from a given layer for a single token position
- [ ] A tiny synthetic fixture implements it: 2 layers, 8 experts, top-2, hidden size 64, with a shared dense path in every layer and a router parameterised by both an input scale and a per-expert scale — structurally the same as the real teacher, small enough to run on CPU in milliseconds
- [ ] The ablation engine computes an attribution vector for a given answer token and layer, over the activated experts plus the near-miss experts
- [ ] Ablation replaces an expert's output with the mean of that token's activated expert outputs; the shared dense path is never modified (see ADR-0003)
- [ ] Near-miss experts are selected by effective combine weight — softmax weight multiplied by per-expert scale — not by raw router logit
- [ ] All probed experts for one (token, layer) are evaluated as a single batched set of variants rather than as separate forward passes
- [ ] The secondary test seam is established, with these invariants passing: replacing an expert with its own output is a bit-exact no-op; the sum of expert contributions plus the shared dense path equals the full feed-forward output; ablating at a token leaves all earlier tokens bit-identical; mean-of-active replacement preserves residual norm within tolerance
- [ ] Attribution vectors carry shape annotations naming their axes, enforced at runtime under pytest
