---
status: accepted
---

# Expert ablation replaces an expert with the mean of that token's active experts, leaving the shared dense path intact

Every teacher layer is a **hybrid layer**: an always-on shared dense MLP (2816→2112→2816) runs in parallel with the 128-expert top-8 mixture (2816→704→2816), summed into the residual stream. The research plan does not mention this, and it changes what "ablating an expert" can mean. We ablate by replacing expert *e*'s output with the mean of that token's eight activated expert outputs, and we never touch the shared dense path. "Gating score" is defined throughout as the **effective combine weight** — softmax top-8 weight multiplied by the layer's `per_expert_scale`.

## Considered Options

**Zeroing the expert** is the most literal removal and is what the attribution-guided pruning literature does, which would make Phase 0.1 an exact replication. It was rejected because it shrinks the residual norm, confounding attribution with a pure magnitude effect — and the plan's own §2.1 already prefers mean replacement for exactly this reason.

**Leave-one-out with renormalised top-k weights** best answers "what if the router had never picked *e*", but it perturbs every other expert's weight simultaneously, so per-expert attributions stop being independently interpretable and the attribution vector loses its meaning as a profile.

**Treating the shared dense path as a 129th expert** would directly quantify the dense-versus-routed division of labour and produce richer modes. It was rejected because it puts a non-expert inside a contribution whose novelty claim is specifically *module-level expert* attribution.

For the combine-weight definition, raw router logits and bare softmax weights were both rejected: `per_expert_scale` reorders experts, so those are not monotone transforms of one another, and the effective combine weight is the **strongest** form of the null hypothesis. Demonstrating divergence against the weight that actually multiplies the expert output is far harder to attack than divergence against a raw logit.

## Consequences

- CEA magnitudes are attenuated by construction: removing one of eight experts leaves the dense path rebuilding the residual stream for free. If the dense path dominates the layer's FFN output norm, every measured ΔlogP is small and the routing–attribution correlation is computed over near-noise.
- This is therefore the single largest threat to the thesis, and it is cheap to measure. **Phase 0.0** is added ahead of everything else in Phase 0: sweep all 30 layers on ~200 examples and report the dense-versus-sparse share of FFN output norm alongside attribution magnitude and diversity per layer. The three CEA layers are then chosen from that sweep rather than asserted, which also answers "why these three layers?" for a rounding error of compute.
- Changing any part of this definition changes the extraction fingerprint and invalidates every cached CEA artefact and every golden file.
