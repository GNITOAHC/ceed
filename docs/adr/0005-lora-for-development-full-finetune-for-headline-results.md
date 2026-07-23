---
status: accepted
---

# LoRA is a development affordance only; every reported Phase 1 number comes from full fine-tuning

Student training runs through a custom `accelerate` loop with parameter-efficiency behind a config flag. Development and smoke runs on the local 4×V100 box use LoRA; **no number that appears in the Phase 1 comparison table may come from a LoRA run.**

The student is 8.0B actual parameters (the "effective 4B" in the plan refers to active compute via Per-Layer Embeddings, not parameter count). Full fine-tuning arithmetic is 16 GB fp16 weights + 16 GB gradients + 32 GB fp32 master + 64 GB Adam state = 128 GB, exactly the total VRAM of the local box with nothing left for activations. LoRA is therefore the only thing that fits locally, and full fine-tuning locally would require ZeRO-3 with CPU optimizer offload at a speed that would destroy development velocity.

The reason the split is a rule rather than a convenience is interpretive. The plan already anticipates that E4 ≈ E3 is a likely outcome and plans to report coupling as a null result. Under LoRA, that null result is unattributable: it could mean coupling does not transfer, or it could mean a rank-*r* adapter lacked the capacity to hold expert-structure supervision. Since the null results are among the paper's honest contributions, they have to be interpretable, and that requires full fine-tuning.

## Considered Options

`Trainer` with a custom `compute_loss` was rejected for the training loop. The E2 and E4 objectives require several forward passes per example — student on original, relevant-intervened, and control-intervened images — plus per-token gating of auxiliary losses by teacher-measured coupling strength. That multi-view, gated structure *is* the research contribution and should not be squeezed through a one-batch-one-forward abstraction.

## Consequences

- Phase 1 cannot complete on the local hardware; the box is for development, teacher-side extraction, and smoke tests. Real training runs require A100/H100-class hardware.
- The two paths must be kept honest: a LoRA run and a full fine-tune of the same group must be distinguishable in the run record, not just by config inspection.
