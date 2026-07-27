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
scripts/infer.py                  # after a run: load the trained Student and query it
scripts/merge_adapter.py          # optional: fold the adapter into a standalone model
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

---

## Step 4 — Load a trained Group and run inference

A finished B1 or B2 run leaves a checkpoint you can load and query. What a LoRA
run writes is an **adapter** — about 9 MB — not a full model; loading it means
loading the base Student and applying the adapter on top.

```
runs/checkpoints/b1/checkpoint/
  adapter/                  # what inference loads (adapter_model.safetensors + config)
  model.safetensors         # accelerate's state, for *resuming* training
  optimizer.bin
  progress.json             # completed_steps and the last loss
```

### From the command line

`scripts/infer.py` handles the loading for you. Point it at a run directory and
it reads the checkpoint path off the run record:

```bash
# replay held-out examples: prediction beside gold, with the score
uv run python scripts/infer.py --run runs/<config_hash> --corpus data/corpus --limit 10

# ask your own questions about an image
uv run python scripts/infer.py --run runs/<config_hash> \
    --image page.png \
    --question "What is the total?" --question "Who signed it?"

# the same questions against the *untrained* Student, to see what training changed
uv run python scripts/infer.py --run runs/<config_hash> --image page.png \
    --question "What is the total?" --base
```

`--checkpoint runs/checkpoints/b1/checkpoint` works too if you would rather name
the checkpoint directly than go through a run record.

### From Python

`load_trained_student` is the one call — it loads the base Student, applies the
Group's adapter, and returns a model ready to generate:

```python
from pathlib import Path

from ceed_core import DecodingConfig, RunRecord
from ceed_student import load_trained_student

record = RunRecord.read(Path("runs/<config_hash>"))
model, processor = load_trained_student(
    "google/gemma-4-e4b-it",          # the base the Group trained from
    Path(record.checkpoint_dir),      # the adapter to apply
    DecodingConfig(max_new_tokens=64),
)
```

Then decode an answer. Use `generate_answer` rather than hand-rolling a prompt —
it renders the **same** prompt the Group was trained and scored under, so what
you see is what the reported number measured:

```python
from ceed_data import Example, ImageStore
from ceed_eval.direct import generate_answer

image_store = ImageStore(Path("data/corpus/images"))
example = Example(
    example_id="docvqa:mine",
    dataset="docvqa",
    image_fingerprint=image_store.put(Path("page.png").read_bytes()),
    question="What is the total?",
    answers=("",),                    # unknown at inference time
)
print(generate_answer(model, processor, example, image_store, DecodingConfig(max_new_tokens=64)))
```

To score predictions as the harness does, add
`ceed_eval.score_answer(dataset, prediction, golds)`.

For the **untrained** Student (B0, or a before/after comparison) swap the loader
and skip the adapter entirely:

```python
from ceed_student import load_student

model, processor = load_student("google/gemma-4-e4b-it", DecodingConfig())
model.eval()
```

### Why not `PeftModel.from_pretrained` directly

The trainer saves the adapter from inside the `CeedStudent` wrapper, so the
recorded module paths carry that wrapper's prefix and applying the adapter to a
bare `AutoModelForImageTextToText` raises *"No modules were targeted for
adaptation."* `load_trained_student` (and `apply_adapter`, if you already have a
model loaded) re-wraps before loading and hands back the inner model, which still
has the `generate` you need.

### What a real B1 run looks like

From the completed 2000-step B1 run in this repository (LoRA rank 4 — 2.27M
trainable of 7.94B — scoring **0.792** ANLS on held-out DocVQA):

```
[OK  ] docvqa:49177
       Q:    What time is 'question and answers' session?
       pred: '12:25 to 12:58 p.m.'
       gold: ['12:25 to 12:58 p.m.']   score 1.000
[MISS] docvqa:57413
       Q:    Name the 4 significant personal care brands of ITC?
       pred: 'Sunfeast, Bingo!, Yippee!, Aim\n\n**Note:** ... it does not explicitly state ...'
       gold: ['Essenza Di Wills, Fiama Di Wills, Vivel and Superia']   score 0.000
```

The miss shows the failure mode worth watching: on a question it cannot answer
from the page, the Student abandons the short-answer instruction and starts
reasoning aloud until `max_new_tokens` truncates it. That scores zero under ANLS
even when the reasoning is sensible. It is a genuine model behaviour, not a
harness bug — but if you see it often, raising `max_new_tokens` will *not* help,
and the answers are worth reading rather than trusting the aggregate.

---

## Step 5 — Merge the adapter into a standalone model (optional)

A LoRA checkpoint is only meaningful next to its base model. To hand the trained
Student to someone else, upload it to the Hub, or serve it with a backend that
knows nothing about CEED, fold the adapter into the weights:

```bash
uv run python scripts/merge_adapter.py --run runs/<config_hash> \
    --output merged/ceed-b1 --verify --corpus data/corpus --limit 8
```

The output is an ordinary Hugging Face checkpoint — same architecture, same
config, same processor as `google/gemma-4-e4b-it`, different weight values:

```
merged/ceed-b1/
  config.json  model.safetensors  generation_config.json
  tokenizer.json  tokenizer_config.json  processor_config.json  chat_template.jinja
  ceed_provenance.json      # which run these weights came from
```

Load it like any other model — no PEFT, no CEED imports:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("merged/ceed-b1", dtype="float16")
processor = AutoProcessor.from_pretrained("merged/ceed-b1")
```

Or upload it:

```bash
uv run huggingface-cli upload <your-org>/<your-repo> merged/ceed-b1
```

### Two things the script does deliberately

**The merge runs in fp32 on CPU**, and the weights are cast to `--save-dtype`
(default fp16) only once, on save. Folding `BA` into `W` at fp16 across 132
projections accumulates rounding error; doing the arithmetic at full precision
costs a few minutes of CPU and ~32 GB of RAM, and removes the question.

**`--verify` reloads the result the way a downstream user would** — plain
`from_pretrained`, no adapter, no wrapper — and re-scores it. That is the check
that the merge preserved behaviour rather than merely producing a file.

### Verified on the B1 run

Merging this repository's 2000-step B1 run and scoring the merged checkpoint
against the adapter path on the same 8 held-out examples produced **identical
generations, token for token** — including a 60-token rambling miss — and the
same 0.7500 mean. The merged model is a true drop-in.

### Caveats

- **Size.** The adapter is 9 MB; the merged checkpoint is ~15 GB per Group.
- **It is still a LoRA result.** Merging does not turn a LoRA run into a full
  fine-tune, and ADR-0005 turns on that distinction — so `ceed_provenance.json`
  carries the source run's `config_hash`, `param_efficiency`, and metrics beside
  the weights. Do not let a merged checkpoint become an unattributed number.
- **Serving backends are a separate question.** The merged model is
  architecturally indistinguishable from the base, so *any backend that can serve
  `google/gemma-4-e4b-it` can serve it identically*. Whether a given backend
  supports `Gemma4ForConditionalGeneration` — a multimodal architecture with a
  text-side MoE block and Per-Layer Embeddings — is not something merging can
  affect. Test the **stock base model** on your target backend first; if that
  works, the merged checkpoint will too.

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

**"No modules were targeted for adaptation"** when loading a trained checkpoint —
you applied the adapter to a bare model. Use `load_trained_student` or
`apply_adapter`; see *Why not `PeftModel.from_pretrained` directly* above.

**CUDA OOM during extraction** — the 26B Teacher needs several cards; keep
`--device-map auto` and do not pin it to one GPU.

**Everything scores ~0** — check the prompt reached the model with its
short-answer instruction (see above).
