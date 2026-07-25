# 06 — Artifact store and teacher extraction

**What to build:** A researcher can run teacher extraction over the corpus and end up with a queryable, fingerprinted store containing everything the baseline Groups need — and can then train Groups without ever loading the teacher again.

**Blocked by:** 02, 04, 05

**Status:** ready-for-agent

- [x] Teacher forward passes are teacher-forced over gold answers with the thinking gate disabled (see ADR-0002) _(the extraction engine reads teacher-forced artefacts and runs only the correctness pass freely; verified on the synthetic teacher. Disabling the thinking gate is a build_source obligation for the real hooked teacher — the gpu-tier follow-on below — since the synthetic teacher has no gate.)_
- [x] The store holds one row per example and answer-token index, with artefact kinds as columns and fixed-size binary encoding for vectors
- [x] Cached artefacts include top-k logits, effective combine weights for all 30 layers, and hidden states for 6 candidate layers — the over-caching that lets the three-layer choice be a training-time selection rather than a re-extraction (see ADR-0001) _(layer sets are ExtractionSpec parameters — all layers / candidate layers — not hardcoded to 30/6)_
- [x] A separate free-generation pass produces a per-example correctness flag
- [x] The store's location is determined by the extraction fingerprint, so artefacts from different ablation or layer definitions cannot be mixed
- [x] A store-level metadata record declares which artefact kinds it contains, the fingerprint, and the source corpus manifest
- [x] Starting a Group whose required artefact kind is absent fails immediately with a clear message _(`StoreMetadata.require_kinds`; wiring it into `run_group` is a later training ticket)_
- [x] The store is queryable with SQL, so Phase 0 analyses are analysis rather than array plumbing
- [x] Extraction runs across several workers each writing independent shards, with no write coordination
- [x] Extraction resumes from a partially written store rather than restarting
- [x] A cached answer-token index is verified to align with the Student's cross-entropy target index on a hand-checked example
- [x] Store round-trips preserve arrays and keys exactly; two different extraction configurations produce different store roots

**Deferred (gpu-tier follow-on):** the real gemma-4-26b-a4b teacher is ~52 GB in fp16 and needs multi-GPU hooking, so wiring it as an `ExtractionSource` (module hooks on the MoE router/experts, thinking gate off) is a separate gpu-tier step, exactly as ticket 05 deferred the real lmms-eval run. The store and extraction engine are complete and verified on the structurally-identical `SyntheticHybridTeacher`.
