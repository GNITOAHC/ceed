---
status: accepted
---

# `torch` is pinned to 2.9.1+cu126 on the development box, expressed as a uv conflicting extra

CUDA 13.0 dropped support for Volta (compute capability 7.0), and the development box is 4×Tesla V100 (sm70) on driver 580.159.03. The `cu128` and `cu130` wheel indexes and the default PyPI `torch` therefore do **not** contain sm70 kernels. Only the `cu126` index does, and its newest build is `torch==2.9.1+cu126`.

Rather than pin every machine to a CUDA 12.6 build, or maintain two independent lockfiles that could drift in any of the other forty dependencies, the workspace declares uv conflicting extras: `[v100]` resolves torch from the `cu126` index, `[modern]` resolves a newer torch from `cu128` for the scale-out hardware. One lockfile carries both, and `uv sync --extra v100` selects the local one.

Python is pinned to 3.13 rather than the 3.14 present on the box. `torch` ships cp314 wheels, but the long tail this project depends on — `diffusers`, `deepspeed`, `opencv`, `lmms-eval` — has materially thinner 3.14 coverage, and nothing here benefits from 3.14.

## Consequences

- Volta has no bf16 and no FlashAttention-2. Both checkpoints ship as bf16 and must be cast, so fp16 numerical stability on Gemma-4 is an open risk to be settled empirically at the first end-to-end run, before any expensive teacher work. If fp16 proves unstable, the fallback is fp32 for teacher extraction, which roughly doubles extraction memory and time.
- The local environment is capped at torch 2.9.1 and cannot use kernels introduced later; scale-out runs will use a different torch, so anything depending on torch version behaviour must be pinned in config, not inferred.
