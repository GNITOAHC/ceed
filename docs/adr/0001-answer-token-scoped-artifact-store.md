---
status: accepted
---

# The artifact store is scoped to answer tokens, and is the only seam between teacher and student

Every teacher-side measurement — top-k logits, effective combine weights, hidden states, CEA, ECC, correctness flags — is precomputed into an on-disk artifact store keyed by `(example_id, answer_token_index)`, and the teacher is never loaded during student training. Crucially, **the store holds answer tokens only**, including for the hidden-state group B3, even though textbook hidden-state KD supervises the full sequence.

## Considered Options

Full-sequence hidden states were the obvious alternative and were rejected on two grounds. The first is mechanical: at ~400 positions per example (280 image soft tokens plus text) across 40k examples and 3 layers, full-sequence hidden states are ~270 GB against 294 GB of free disk, before any other artefact exists.

The second reason is the one that would still hold on a larger disk. Phase 1 claims that groups differ *only* in their auxiliary signal, and that auxiliary-loss budgets are matched. If B3 supervises 400 positions per example while E1 supervises ~8, the B3-versus-E1 comparison is confounded by supervision volume rather than by signal type, and the central comparison of the paper is no longer controlled. Answer-token scoping makes every group supervise the identical token set. It is a deviation from convention chosen deliberately, and the paper must say so.

A co-resident teacher during training was also rejected: a 49 GB fp16 teacher cannot share 4×V100 with a training 8B student, and it would re-run identical teacher forwards once per group across twelve groups.

## Consequences

- Reported B3 numbers are not directly comparable to published full-sequence hidden-KD results. This is a stated limitation, not an oversight.
- The store is Parquet shards read through DuckDB, one shard per extraction worker so parallel writes need no coordination, with fixed-size binary columns for vectors. Phase 0's analyses are then largely SQL rather than bespoke array code. At ~6 GB total the format was chosen for analysis ergonomics, not throughput.
- The store root is the extraction fingerprint (see the CONTEXT glossary), so artefacts produced under different ablation or layer definitions cannot be silently mixed.
- Because the store is the only seam, effective combine weights are cached for **all 30 teacher layers** and hidden states for **6 candidate layers**, at a combined cost of ~14 GB. This makes the eventual three-layer choice a selection at training time rather than a re-extraction, which is what lets the baseline groups be built before the Phase 0 layer sweep has run. CEA itself cannot be over-cached this way — ablating all 30 layers costs ten times the pilot — so the sweep remains a Phase 0 job on a 200-example subset.
