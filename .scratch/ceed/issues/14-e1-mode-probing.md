# 14 — E1: Mode probing with removable probes

**What to build:** The first experimental Group. Linear probes on selected Student layers are trained to predict the teacher's per-token Mode assignments, and then deleted — so the deployed Student is byte-for-byte the architecture it started as and the zero-added-inference-cost claim is literally true.

**Blocked by:** 07, 11

**Status:** ready-for-agent

- [ ] Probes attach to Student layers selected by the layer mapping and predict per-token Mode assignments as soft targets
- [ ] The probing loss is gated per token and introduced on the warm-up schedule, at the small weight relative to cross-entropy that the plan specifies
- [ ] Probes are removable: after removal the Student's architecture is identical to the base checkpoint, verified rather than asserted
- [ ] E1 runs through `run_group` and reports accuracy
- [ ] The run record states which layer mapping placed the probes, so the robustness claim is checkable
- [ ] Adding E1 required a new auxiliary-signal component and a configuration overlay, and no change to the training loop
