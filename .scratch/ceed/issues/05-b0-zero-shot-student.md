# 05 — B0: the zero-shot Student

**What to build:** The first real number. A researcher runs Group B0 and gets zero-shot accuracy for the Student on all three datasets, produced by a standard evaluation harness under enforced greedy decoding.

This ticket also carries the project's first serious technical risk. Volta has no bf16 and both checkpoints ship as bf16, so the Student must be cast to fp16 — and Gemma models have a reputation for fp16 overflow. Do the DocVQA path first and confirm coherent output before building out the rest, so that instability surfaces on day one rather than after the harness is finished.

**Blocked by:** 02, 04

**Status:** ready-for-agent

- [ ] The Student loads in fp16 on a single V100 and produces coherent answers to DocVQA questions — verified before the remaining criteria are attempted _(blocked: `gemma-4-E4B-it` weights are not in the local cache; the load+greedy path and a gpu-tier coherence test are implemented and skip until the weights are present)_
- [ ] If fp16 proves numerically unstable, the finding is recorded with evidence and the fallback path (fp32 for extraction) is costed before proceeding _(cannot be exercised until the criterion above runs)_
- [x] An evaluation adapter drives the Student through `lmms-eval`, so accuracy is comparable to published baselines rather than to a bespoke metric reimplementation
- [x] Decoding is greedy, enforced in code, overriding whatever the checkpoint's shipped generation configuration says (see plan amendment A9)
- [x] Group B0 runs through `run_group` and reports accuracy on DocVQA, GQA, and ChartQA
- [x] The run record captures the harness version and the decoding settings actually used
