# 06 — Artifact store and teacher extraction

**What to build:** A researcher can run teacher extraction over the corpus and end up with a queryable, fingerprinted store containing everything the baseline Groups need — and can then train Groups without ever loading the teacher again.

**Blocked by:** 02, 04, 05

**Status:** ready-for-agent

- [ ] Teacher forward passes are teacher-forced over gold answers with the thinking gate disabled (see ADR-0002)
- [ ] The store holds one row per example and answer-token index, with artefact kinds as columns and fixed-size binary encoding for vectors
- [ ] Cached artefacts include top-k logits, effective combine weights for all 30 layers, and hidden states for 6 candidate layers — the over-caching that lets the three-layer choice be a training-time selection rather than a re-extraction (see ADR-0001)
- [ ] A separate free-generation pass produces a per-example correctness flag
- [ ] The store's location is determined by the extraction fingerprint, so artefacts from different ablation or layer definitions cannot be mixed
- [ ] A store-level metadata record declares which artefact kinds it contains, the fingerprint, and the source corpus manifest
- [ ] Starting a Group whose required artefact kind is absent fails immediately with a clear message
- [ ] The store is queryable with SQL, so Phase 0 analyses are analysis rather than array plumbing
- [ ] Extraction runs across several workers each writing independent shards, with no write coordination
- [ ] Extraction resumes from a partially written store rather than restarting
- [ ] A cached answer-token index is verified to align with the Student's cross-entropy target index on a hand-checked example
- [ ] Store round-trips preserve arrays and keys exactly; two different extraction configurations produce different store roots
