# 02 — The `run_group` spine

**What to build:** A researcher can describe a Group in YAML and run it with a single call, getting back a run record written to disk. Nothing real happens yet — no model, no corpus — but the whole spine exists and is the seam every later ticket tests through.

**Blocked by:** 01

**Status:** ready-for-agent

- [x] Group configuration is expressed as typed models composed from YAML overlays: a base, a student layer, a Group overlay, and an optional Phase 2 variant
- [x] Invalid configuration fails at load time with a clear message, not partway through a run
- [x] Configuration serialises to canonical JSON whose hash is the run identifier; the extraction-relevant subset of fields hashes separately to the extraction fingerprint (see ADR-0001)
- [x] Two configurations differing only in field order or formatting produce the same hash; two differing in any meaningful field produce different hashes
- [x] `run_group` accepts a resolved Group configuration and returns a run record
- [x] The run record states the Group code, seed, parameter-efficiency mode, layer mapping, and configuration hash
- [x] Metrics are written to JSONL on disk as the source of truth; the tracking service is optional and its absence is not an error
- [x] A null Group — no auxiliary signals, no model, no corpus — runs end to end and writes a valid run record
- [x] The primary test seam is established: tests drive `run_group` and assert on the returned run record, not on internals
