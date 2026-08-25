# CEED — running the baseline Groups

Distilling the measured division of computational labour from a sparse MoE
vision-language teacher (`gemma-4-26b-a4b-it`) into a compute-matched dense
student (`gemma-4-e4b-it`).

This README covers **B0–B5, end to end**. For anything beyond that — inference,
resuming, troubleshooting — see [`docs/running-groups.md`](docs/running-groups.md).

## The Groups

| Group | Objective | Trains | Reads from the store |
| --- | --- | --- | --- |
| **B0** | the Student as it ships, zero-shot | no | — |
| **B1** | cross-entropy on gold answers (`kd_weight: 0`) | yes | — |
| **B2** | B1 **+ top-k logit distillation** — the primary baseline | yes | `top_k_logit_*` |
| **B3** | B2 **+ hidden-state projection** | yes | `+ hidden_states` |
| **B4** | B2 **+ VA-OPD visual-advantage reweighting** | yes | `+ visual_advantage` |
| **B5** | B2 **+ router combine-weight probing** | yes | `+ combine_weights` |

B3–B5 differ from B2 in their auxiliary signal and nothing else. That is asserted
in `tests/test_b3_b4_b5.py`, not maintained by hand.

## Setup

```bash
uv sync
uv run hf auth whoami          # gemma repos are gated; a write token is needed for upload
```

Everything runs through `uv`. Never invoke `python` or `pytest` directly.

## 1. Verify the sources (~6 min)

```bash
uv run pytest -m slow tests/test_loaders_live.py
```

Do not skip this. It is the only thing that says the Hub still serves the fields
the loaders read.

## 2. Build the corpus (~1–2 h, ~1.3 GB)

```bash
uv run python scripts/build_corpus.py \
    --output data/corpus-all \
    --datasets docvqa gqa chartqa \
    --limit 0 --dataset-limit gqa=10000 --seed 0
```

`--limit 0` takes DocVQA and ChartQA whole; the override caps GQA, whose source
holds 943,000 questions. Result: **17,849 examples** (docvqa 5,349 / gqa 10,000 /
chartqa 2,500) split 80/10/10 → **train 14,278 / validation 1,830 / test 1,741**.

Which split each source is drawn from — `validation` for DocVQA, `train` for GQA,
`test` for ChartQA — is what the published sources actually serve, not what their
papers name. CEED's own 10% test split is held out on top.

## 3. Cache the Teacher's artefacts (~3 h, needs all 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/extract_teacher_artifacts.py \
    --corpus data/corpus-all --store data/store-v2 \
    --group b2 --split train --top-k 64 --skip-correctness --all
```

**One pass serves B2–B5.** All six Groups resolve to the same extraction
fingerprint, so `--group` only names whose config supplies the extraction subset;
`--all` is what decides the artefact kinds. B0 and B1 read nothing from the store.

It resumes — re-run the identical command after a preemption. A store's schema is
fixed at creation: to add an artefact kind later you need a new `--store`
directory, so always pass `--all`.

## 4. Train and score all six (~25 h sequential, ~11 h in four lanes)

```bash
./run_each_group.sh                    # all six, one GPU, one at a time
./run_each_group.sh b3 b4 b5           # a subset
```

Or four lanes across the four cards:

```bash
setsid nohup bash scripts/run_baselines.sh \
    data/corpus-all data/store-v2 runs logs 4800 \
    > logs/baselines-all.log 2>&1 < /dev/null &
```

Both are resumable: a Group at its step budget is reused rather than retrained,
one killed halfway continues from its last checkpoint. Re-running is the recovery.

`STEPS=4800` is optimiser steps, each consuming 8 examples by gradient
accumulation — **2.69 passes** over the 14,278-example train split. Raising it
continues the same run rather than starting a new one. What matters for the
comparison is only that every Group shares the value.

Knobs: `CORPUS`, `STORE`, `OUTPUT`, `LOGDIR`, `GPU`, `STEPS`.

## 5. Read the results

```bash
./collect_results.sh
```

Resolves each Group's `config_hash` from the corpus and budget, so the rows are
matched rather than guessed at, and exits non-zero if they are not one comparison
— mixed seeds, budgets, modes, or extraction fingerprints.

Three metrics, no aggregate: ANLS, exact match and relaxed accuracy do not
average into anything meaningful.

## 6. Merge and publish (optional)

```bash
./merge_and_upload.sh                  # merge + verify + model card, no upload
UPLOAD=1 ./merge_and_upload.sh         # ... and upload, private
```

B1–B5 only; B0 is the unmodified base model. Per Group: fold the adapter in
(fp32 on CPU), reload and re-score it the way a downstream user would, generate
`README.md` from the run record, upload, then delete to hold the disk peak at
15 GB instead of 75 GB (`KEEP=1` to keep them).

Knobs: `HF_OWNER`, `PUBLIC=1`, `KEEP=1`.

## Results on this corpus

`data/corpus-all` (fingerprint `01c253e0db94387c`), 4800 steps, LoRA rank 4,
scored on the full 1,830-example validation split:

| Group | DocVQA (ANLS) | GQA (exact) | ChartQA (relaxed) |
| --- | --- | --- | --- |
| B0 zero-shot | 0.7975 | 0.4528 | 0.5863 |
| **B1** no teacher | **0.8798** | **0.6959** | **0.7871** |
| B2 +logit KD | 0.8506 | 0.6191 | 0.5783 |
| B3 +hidden-state | 0.8501 | 0.6289 | 0.5341 |
| B4 +VA-OPD | 0.8538 | 0.6102 | 0.6185 |
| B5 +combine-weight | 0.8573 | 0.6083 | 0.5663 |

**The no-teacher control wins on every dataset.** On ChartQA, B2/B3/B5 land below
zero-shot. Under a rank-4 adapter the KD term is costing accuracy, not adding it.

## Two things that constrain what these numbers mean

- **These are LoRA runs** (rank 4, 2.27M of 7.94B trainable). Per ADR-0005 no LoRA
  number may enter the Phase 1 table: a null cannot be attributed between "the
  signal does not transfer" and "the adapter lacked the capacity to hold it".
  Headline runs need `--param-efficiency full` on A100/H100-class hardware.
- **Scores are not leaderboard-comparable.** CEED's own splits, prompt and
  decoding (`harness_version: ceed-direct-1`). They are meaningful only against
  each other.
