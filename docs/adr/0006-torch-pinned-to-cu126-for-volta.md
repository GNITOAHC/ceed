---
status: accepted
---

# `torch` is pinned to 2.9.1+cu126 on the development box, expressed as a uv conflicting extra

CUDA 13.0 dropped support for Volta (compute capability 7.0), and the development box is 4×Tesla V100 (sm70) on driver 580.159.03. The `cu128` and `cu130` wheel indexes and the default PyPI `torch` therefore do **not** contain sm70 kernels. Only the `cu126` index does, and its newest build is `torch==2.9.1+cu126`.

Rather than pin every machine to a CUDA 12.6 build, or maintain two independent lockfiles that could drift in any of the other forty dependencies, the workspace declares two conflicting **dependency groups**: `v100` resolves torch from the `cu126` index, `modern` resolves a newer torch from `cu128` for the scale-out hardware. One lockfile carries both.

Groups rather than optional-dependencies, because uv has no `default-extras` setting (verified against uv 0.11.29). With extras, a bare `uv sync` selects neither target, and `accelerate` and `transformers` then pull torch transitively from PyPI — which ships no sm70 kernels. The result is an environment that installs cleanly, imports cleanly, and fails only when a kernel is finally launched on the GPU. `default-groups = ["dev", "v100"]` makes the Volta build the default, so the local box is correct without a flag and the scale-out target is the explicit opt-in (`--no-default-groups --group dev --group modern`, which is what CI uses).

Python is pinned to 3.13 rather than the 3.14 present on the box. `torch` ships cp314 wheels, but the long tail this project depends on — `diffusers`, `deepspeed`, `opencv`, `lmms-eval` — has materially thinner 3.14 coverage, and nothing here benefits from 3.14.

## Consequences

- Verified on the box: `torch 2.9.1+cu126` reports `sm_70` in its architecture list and runs an fp16 matmul across all four devices. A PyPI-resolved torch would not.
- Volta has no bf16 and no FlashAttention-2. Both checkpoints ship as bf16 and must be cast, so fp16 numerical stability on Gemma-4 was an open risk to be settled empirically at the first end-to-end run, before any expensive teacher work. If fp16 proved unstable, the fallback would have been fp32 for teacher extraction, roughly doubling extraction memory and time.
  - **Resolved for the Student (2026-07-25, ticket 05 / B0).** `google/gemma-4-e4b-it` loads in fp16 on a V100 and is numerically stable: logits stay finite on both the text and the vision path, and answers are coherent (greedy "4" for "2 + 2"; "42" read from a document image). The fp32 fallback is not needed for the Student, and B0 runs in fp16. This is exercised by the gpu-tier test `test_student_loads_in_fp16_and_answers_coherently`.
  - **Still open for teacher extraction.** The teacher is a larger sparse-MoE checkpoint whose ablation resumes partial forwards; its fp16 stability is confirmed at the first extraction run, not by B0. The fp32 fallback and its ~2× cost remain the plan of record should teacher extraction prove unstable.
- The lockfile still contains a PyPI `torch` entry. That is the resolution for the `--no-default-groups` fork only; it is unreachable from either supported sync command.
- The local environment is capped at torch 2.9.1 and cannot use kernels introduced later; scale-out runs will use a different torch, so anything depending on torch version behaviour must be pinned in config, not inferred.
