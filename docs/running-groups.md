# Running and evaluating B0, B1, and B2

This is the operational guide for the three baseline Groups. Everything runs
through `uv` from the repository root.

| Group | What it is | Trains? | Needs the Teacher? |
| --- | --- | --- | --- |
| **B0** | The Student as it ships, zero-shot | no | no |
| **B1** | Supervised fine-tuning on gold answers (`kd_weight: 0`) | yes | no |
| **B2** | The primary baseline: cross-entropy **+ top-k logit distillation** | yes | **yes** |

Every experimental Group later in the plan is B2 plus one or more auxiliary
signals, so B2 is the number they are all measured against, and B1 is the
no-teacher control that says how much of B2's gain is distillation rather than
fine-tuning.

## The three scripts

```
scripts/build_corpus.py           # once: assemble the shared corpus
scripts/extract_teacher_logits.py # once per corpus: cache what B2 distils from
scripts/run_group.py              # per Group: train (if it trains) and score
```

---

## Step 1 — Build the corpus (once)

Every Group must read the *identical* examples or the comparison means nothing,
so the corpus is assembled once, split deterministically by example id, and
written with a manifest recording exactly what went where.

```bash
uv run python scripts/build_corpus.py --output data/corpus --limit 200
```

- `--limit` is **per dataset**. Start small: the sources are gigabytes.
- `--datasets docvqa gqa chartqa` selects sources (all three by default).
- Splits are fixed at 80/10/10 train/validation/test, seeded by `--seed`.

Produces:

```
data/corpus/
  examples.jsonl          # one Example per line
  corpus_manifest.json    # seed, fractions, and the ids in each split
  images/                 # images addressed by content hash
```

If a source fails (gated repo, renamed dataset) the script warns and continues
with the rest, so one broken source does not cost you the others.

## Step 2 — Cache the Teacher's logits (only for B2)

B2's backbone needs the Teacher's **top-k next-token logits at each gold answer
token**. That is an ordinary teacher-forced forward pass — it needs none of the
MoE hooking (router internals, expert ablation) the Causal Expert Attribution
work requires, which is why the primary baseline is runnable now.

```bash
uv run python scripts/extract_teacher_logits.py \
    --corpus data/corpus --store data/store \
    --group b2 --split train --top-k 64
```

- The 26B Teacher is sharded across GPUs by `--device-map auto`. On the 4×V100
  box it needs all four cards.
- `--skip-correctness` skips the free-generation pass; much faster for a smoke
  run, at the cost of the per-example correctness flag.
- **It resumes.** Examples already in the store are skipped, so a preempted run
  is restarted with the same command.
- Shard across processes with `--n-workers N --worker-index I --worker-id wI`;
  workers need no coordination.

The store lands at `data/store/<extraction_fingerprint>/`. The fingerprint is a
hash of the extraction-relevant config, so artefacts from a different Teacher or
ablation can never be silently mixed into a Group that did not extract them.

## Step 3 — Run a Group

```bash
# B0 — zero-shot, no training
uv run python scripts/run_group.py --group b0 --limit 100

# B1 — supervised fine-tuning, no Teacher
uv run python scripts/run_group.py --group b1 --steps 2000 --limit 100

# B2 — the primary baseline (needs step 2 to have run)
uv run python scripts/run_group.py --group b2 --steps 2000 --limit 100
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--steps` | Override the config's step budget |
| `--limit` | Cap **evaluation** examples per dataset |
| `--train-limit` | Cap **training** examples encoded |
| `--seed` | Override the seed (part of the run identity) |
| `--param-efficiency lora\|full` | Override the mode — see the warning below |
| `--skip-eval` | Train only |
| `--eval-split` | Defaults to `validation`; use `test` only for final numbers |

What the script builds is decided by the Group's config, not by flags: a Group
with no `training:` block never constructs a trainer, and a Group whose
`kd_weight` is 0 never opens a store. A Group that *does* distil and finds no
store fails immediately with the exact extraction command to run.

### Output

```
runs/
  <config_hash>/
    run_record.json    # the durable statement of what this run was
    metrics.jsonl      # the event stream; disk is the source of truth
  checkpoints/<group>/checkpoint/   # accelerate state + the LoRA adapter
```

`run_record.json` carries the group, seed, **parameter-efficiency mode actually
trained**, layer mapping, both config hashes, the metrics, and the checkpoint
path.

## Resuming

Training checkpoints and resumes; a preempted Group is restarted with the same
command. Re-running a Group that already hit its step budget retrains nothing
and reuses the frozen checkpoint, which is what keeps a baseline from being
quietly retrained for a later comparison.

```bash
# ran 4 steps, then asked for 6: runs the 2 remaining
uv run python scripts/run_group.py --group b2 --steps 6 ...   # steps_run: 2, resumed: true
```

## How the numbers are produced

All three Groups are scored by the **same** evaluator (`ceed_eval.DirectEvaluator`):
the same held-out split, greedy decoding enforced in code (A9), and the metric
each dataset is reported under — ANLS for DocVQA, exact match for GQA, relaxed
accuracy for ChartQA. That identity is what makes B0/B1/B2 comparable to each
other.

Two things are deliberate and worth knowing:

- **Prompts are shared.** One builder renders the prompt for teacher extraction,
  training, and evaluation, including the short-answer instruction. Without that
  instruction an instruction-tuned model answers `"The total written in the image
  is **28**."` against gold `"28"` and scores zero on every metric — all Groups
  sink equally and any real difference is hidden.
- **`harness_version` is `ceed-direct-1`** on these records, so a direct score is
  never confused with an `lmms-eval` score. `LmmsEvalEvaluator` remains available
  for B0 when you want a number comparable to published baselines.

## ⚠️ LoRA vs full fine-tuning (ADR-0005)

`configs/base.yaml` sets `param_efficiency: lora`, the development default — it
is the only mode that fits on the local box. **No LoRA number may appear in the
Phase 1 comparison table.** The reason is interpretive, not procedural: the plan
expects some null results, and under LoRA a null is unattributable — it could
mean the signal does not transfer, or merely that a rank-4 adapter lacked the
capacity to hold it.

The run record states the mode that *actually trained* (sourced from the training
outcome, not the config), plus the adapter rank, so the two paths can never be
confused after the fact. Headline runs need `--param-efficiency full` on
A100/H100-class hardware.

## Verified smoke path

The following was run end to end on the 4×V100 box against a small synthetic
corpus, and is the fastest way to confirm your environment before committing to a
real corpus:

```bash
uv run python scripts/run_group.py --group b0 --corpus <corpus> --limit 3
uv run python scripts/run_group.py --group b1 --corpus <corpus> --steps 3 --train-limit 4
uv run python scripts/extract_teacher_logits.py --corpus <corpus> --store <store> \
    --group b2 --limit 4 --top-k 32 --skip-correctness
uv run python scripts/run_group.py --group b2 --corpus <corpus> --store <store> \
    --steps 4 --train-limit 4 --limit 3
```

Observed: B0 scored 1.0 on the synthetic questions; B1 trained with `kd = 0.0`
exactly (no teacher); B2 trained with a genuine non-zero `kd` term read from the
26B Teacher's cached logits; and a follow-up B2 run with a larger budget resumed
rather than restarting.

## Troubleshooting

**"Group B2 distils from the Teacher but no artifact store exists at …"** — run
step 2. The message includes the exact command.

**"no row for ('docvqa:q1', 3) in store"** — the Student encoded more answer
tokens than the store holds for that example. The store was built from a
different corpus or a different prompt; re-extract against the corpus you are
training on. This is a deliberate hard failure: misaligned supervision is
otherwise silent.

**"no adaptable nn.Linear modules named [...]"** — the LoRA target names do not
match this Student. gemma-4 keeps plain `nn.Linear` projections in the language
model and `Gemma4ClippableLinear` (which PEFT cannot adapt) in the vision tower;
targets are resolved to the language path automatically.

**CUDA OOM during extraction** — the 26B Teacher needs several cards; keep
`--device-map auto` and do not pin it to one GPU.

**Everything scores ~0** — check the prompt reached the model with its
short-answer instruction (see above).
