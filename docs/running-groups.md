# Running and evaluating the baseline Groups, B0 to B5

This is the operational guide for the six baseline Groups. Everything runs
through `uv` from the repository root.

| Group | What it is | Trains? | Teacher artefacts it reads |
| --- | --- | --- | --- |
| **B0** | The Student as it ships, zero-shot | no | none |
| **B1** | Supervised fine-tuning on gold answers (`kd_weight: 0`) | yes | none |
| **B2** | The primary baseline: cross-entropy **+ top-k logit distillation** | yes | `top_k_logit_*` |
| **B3** | B2 **+ hidden-state projection distillation** | yes | `+ hidden_states` |
| **B4** | B2 **+ VA-OPD's visual-advantage reweighting** | yes | `+ visual_advantage` |
| **B5** | B2 **+ router combine-weight probing** | yes | `+ combine_weights` |

Every experimental Group later in the plan is B2 plus one or more auxiliary
signals, so B2 is the number they are all measured against; B1 is the no-teacher
control that says how much of B2's gain is distillation rather than fine-tuning;
and B3 to B5 are the three published-alternative comparators.

**B3, B4 and B5 differ from B2 in their auxiliary signal and in nothing else** —
same corpus, same backbone weighting, same step budget, same decoding, same layer
mapping. That is asserted in `tests/test_b3_b4_b5.py` rather than maintained by
hand, because an accidental asymmetry would produce a plausible wrong number
rather than a failure.

## The scripts

```
scripts/build_corpus.py              # once: assemble the shared corpus
scripts/extract_teacher_artifacts.py # once per corpus: cache what B2-B5 read
scripts/run_group.py                 # per Group: train (if it trains) and score
scripts/run_baselines.sh             # all six Groups in order, resumable
scripts/infer.py                     # after a run: load the trained Student and query it
scripts/merge_adapter.py             # optional: fold the adapter into a standalone model
```

The whole set, once the corpus and store exist:

```bash
scripts/run_baselines.sh data/corpus data/store-full runs logs [steps]
```

One Group per GPU — the Student is 8B in fp16 and fits on a single V100 alongside
a document-length sequence — so the six Groups run in four lanes rather than in
sequence. It is resumable in the strong sense: a Group that has reached its step
budget is skipped rather than retrained, and a Group killed halfway continues
from its last checkpoint. Re-running the script after a preemption is the
recovery.

---

## Step 1 — Build the corpus (once)

Every Group must read the *identical* examples or the comparison means nothing,
so the corpus is assembled once, split deterministically by example id, and
written with a manifest recording exactly what went where.

```bash
uv run python scripts/build_corpus.py --output data/corpus --limit 200
```

- `--limit` is **per dataset**, and **zero or negative means the whole source
  split**. Start small: the sources are gigabytes.
- `--dataset-limit gqa=5000` overrides `--limit` for one dataset. The three
  sources differ by three orders of magnitude, so one cap rarely suits all of
  them.
- `--datasets docvqa gqa chartqa` selects sources (all three by default).
- Splits are fixed at 80/10/10 train/validation/test, seeded by `--seed`.
- **A requested dataset that yields nothing fails the build.** Previously it was
  warned about and dropped, which meant asking for three datasets and silently
  getting one. `--allow-partial` restores the old behaviour when you mean it.

### Which source split each dataset comes from

These are the splits the published sources actually serve, which is not what
their papers name:

| Dataset | Source | Config | Split | Size |
| --- | --- | --- | --- | --- |
| DocVQA | `lmms-lab/DocVQA` | `DocVQA` | `validation` | ~5.3k questions |
| GQA | `lmms-lab/GQA` | `{split}_balanced_instructions` **joined to** `{split}_balanced_images` | `train` | ~943k questions |
| ChartQA | `lmms-lab/ChartQA` | `default` | `test` | ~2.5k questions |

DocVQA withholds its test answers, so validation is the usable split. ChartQA
publishes only `test`. CEED's own 80/10/10 re-split applies on top of whichever
split it drew from, so CEED's `test` is held out from CEED regardless.

**GQA is a join, not a stream.** Its questions and its images are in separate
configs — the instructions config is text only, carrying an `imageId`. The loader
takes a prefix of the images, then keeps the questions whose image it holds. The
other direction reads all ~72k images to satisfy a few thousand questions. One
consequence worth knowing: GQA examples carry **no `answer_region`**, because the
answer-linked box lives in the scene-graph release rather than in the question
rows. Nothing in B0–B5 reads it; Phase 2's region interventions will have to
supply it.

To take everything a source has:

```bash
uv run python scripts/build_corpus.py --output data/corpus-full --datasets docvqa --limit 0
```

Do not do that for GQA. `--limit 0` there means ~943k questions and every image
behind them.

### Check the sources before you trust a build

The loaders talk to the Hub, so they are out of the fast tier. Run them
deliberately:

```bash
uv run pytest -m slow tests/test_loaders_live.py
```

That is what says the sources still serve the fields the loaders read. Two of the
three loaders were broken against the live Hub for a long time precisely because
nothing exercised them.

Produces:

```
data/corpus/
  examples.jsonl          # one Example per line
  corpus_manifest.json    # seed, fractions, and the ids in each split
  images/                 # images addressed by content hash
```

### The corpus is part of a run's identity

`corpus_manifest.json` hashes to a **corpus fingerprint** — a content hash of
exactly which example ids landed in which split — and every entry point fills it
into the Group's `corpus.fingerprint` before the run hash is computed.

This is not bookkeeping. `CorpusConfig.name` is the constant `ceed-vqa` in every
Group's overlay, so without the fingerprint a Group trained on DocVQA alone and
the same Group trained on all three datasets hash **identically**, and the
consequences are silent rather than loud:

- `runs/<hash>/run_record.json` is overwritten, so the first result is replaced
  by the second under a hash claiming to describe both;
- the checkpoint directory is keyed by the resume key, which is that same hash
  with the step budget normalised away — so the second run finds a
  `progress.json` written by a matching configuration, adopts it, and, being
  already at its step budget, **trains zero steps** before being scored on the
  new corpus and reported as a new result.

The resume guard cannot catch that on its own: it compares configurations, and
the configurations really are identical. The corpus is what differs. Changing the
corpus now changes the hash, so the two runs cannot collide.

A corpus directory with no manifest — the synthetic ones the fast tier builds —
contributes no fingerprint, and those Groups hash as they always did.

### Building the three-dataset corpus

The DocVQA-only corpus is the one the published baselines used. To train on
everything CEED supports:

```bash
uv run python scripts/build_corpus.py \
    --output data/corpus-all \
    --datasets docvqa gqa chartqa \
    --limit 0 \
    --dataset-limit gqa=5000 \
    --seed 0
```

`--limit 0` takes DocVQA and ChartQA whole; the override keeps GQA to a
comparable share instead of the ~943k questions it would otherwise contribute.
Roughly:

| Dataset | Examples | Metric it is scored under |
| --- | --- | --- |
| DocVQA | ~5,350 | ANLS |
| GQA | 5,000 | exact match |
| ChartQA | ~2,500 | relaxed accuracy |
| | **~12,850** → train ~10,280 / validation ~1,285 / test ~1,285 | |

Two things follow from mixing sources that did not matter with one:

- **The split is not stratified.** 80/10/10 is applied to the pooled corpus, so
  each split's dataset mix is what chance gives it. With thousands per dataset
  that is close enough to proportional; check `dataset_counts` in the manifest
  and the per-dataset `n` in the run record rather than assuming.
- **The evaluator reports one metric per dataset**, and there is no aggregate.
  A run record on this corpus carries `docvqa`, `gqa` and `chartqa` separately —
  which is correct, because ANLS and relaxed accuracy do not average into
  anything meaningful.

## Step 2 — Cache the Teacher's artefacts (for B2 and above)

The artifact store is the single seam between teacher measurement and student
training: the Teacher is **never loaded during training**, so everything a Group's
objective needs is measured once here.

```bash
# B2 only: the top-k logits its backbone distils from
uv run python scripts/extract_teacher_artifacts.py \
    --corpus data/corpus --store data/store \
    --group b2 --split train --top-k 64

# everything B2 through B5 need, in one pass
uv run python scripts/extract_teacher_artifacts.py \
    --corpus data/corpus --store data/store-full \
    --group b2 --split train --top-k 64 --all
```

What `--all` adds, and what each costs:

| Flag | Artefact | For | Cost |
| --- | --- | --- | --- |
| `--with-hidden-states` | residual state at 6 candidate layers | B3 | ~34 KB per answer token |
| `--with-combine-weights` | effective combine weight, all 30 layers | B5 | ~8 KB per answer token |
| `--with-visual-advantage` | per-token visual advantage | B4 | **a second forward per example** |

None of this is the Causal Expert Attribution extraction, which needs the hooked
forward that can ablate an expert and re-run the tail. Combine weights are a
byproduct of the ordinary forward — which is exactly why B5 ships with the
baselines and the CEA Groups do not.

> **A store's schema is fixed when it is created.** Adding an artefact kind to an
> existing store is refused, because rows already written do not have the column.
> If you extracted for B2 and now want B3 to B5, extract into a **new** `--store`
> directory with `--all`. The B2 store stays valid and B2 is not retrained.

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

# B3, B4, B5 — each needs the store its signal reads, so extract with --all
uv run python scripts/run_group.py --group b3 --store data/store-full --steps 2000 --limit 100
uv run python scripts/run_group.py --group b4 --store data/store-full --steps 2000 --limit 100
uv run python scripts/run_group.py --group b5 --store data/store-full --steps 2000 --limit 100
```

A Group whose signal reads an artefact the store never cached refuses to start,
naming the missing kind — the check happens before the Student is loaded, not at
hour six.

### How much training a Group actually does

`steps` is **optimiser steps**, and each one consumes `batch_size` examples by
gradient accumulation: the Student sees one example's activations at a time (it
is 8B, and a page is thousands of visual tokens) while the gradient is the mean
over the batch. So the training set is walked

    steps x batch_size / |train split|   times,

which at the checked-in `2000 x 8` over the 4,282-example DocVQA-only train split
is **~3.7 epochs**. The run record carries `train.epochs` so this is read off the
record rather than recomputed — and the metrics log carries `examples_seen`
beside it.

**The budget does not follow the corpus.** Enlarging the corpus without raising
`steps` buys fewer passes over it, not more training, so a Group on a corpus
2.4x the size at the same budget sees each example 1.6 times rather than 3.7.
Decide which you are holding fixed — examples seen, or passes over the corpus —
and say which in the write-up, because a Group compared against a baseline that
held the other one fixed is not a comparison. `scripts/run_baselines.sh` takes
the budget as its fifth argument for exactly this.

The corpus is **shuffled once per epoch**, deterministically from the Group's
seed. Two Groups at the same seed therefore walk it in the same order, which is
what keeps their difference the auxiliary signal rather than the shuffle; and the
order is a pure function of the global position, so a preempted run resumes the
same walk instead of restarting the epoch.

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

Note that `--limit` and `--train-limit` are different flags: `--limit` caps only
what is *scored*. **To train on the whole corpus, omit `--train-limit`** — it
defaults to no cap. The corpus's own size is fixed earlier, by step 1.

Two things about the step budget, as the loop currently stands:

- **One step consumes `batch_size` examples**, by gradient accumulation — the
  Student sees one example's activations at a time and the optimiser steps on the
  mean of eight. So `--steps` counts optimiser steps, not examples: walking a
  4,282-example corpus once at `batch_size: 8` is `--steps 536`.
- **Examples are encoded on demand, not up front.** `build_batches` reads every
  cached teacher row in one query and returns a lazy corpus that encodes an
  example when the loop indexes it. Encoding eagerly cost ~8 MB per example in
  pixel values and took the host past its memory with four lanes running.

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
  checkpoints/<group>-<key>/checkpoint/          # accelerate state + the LoRA adapter
  checkpoints/<group>-<key>/checkpoint/probes/   # B3/B5 only: the discarded probes
```

`run_record.json` carries the group, seed, **parameter-efficiency mode actually
trained**, layer mapping, both config hashes, the metrics, and the checkpoint
path. For a Group with an auxiliary signal it also carries a
`train.aux.<signal_name>` metric, so a row in the Phase 1 table states its own
independent variable.

**Probes are deletable.** B3's projections and B5's probes are parameters of the
*signal*, never of the Student, so nothing has to be stripped out afterwards: the
Student a probing run leaves behind is architecturally identical to the base
model, and the zero-added-inference-cost claim is literally true. They are written
to `probes/` beside the checkpoint for analysis, where no server loading the
adapter can pick them up.

They are not the same size, and that matters when reading the numbers. On the real
Student, B3's projections are **21.6M** parameters and B5's probes **1.0M**,
against a rank-4 adapter's 2.3M in the Student itself. B3's projection can absorb
much of its own matching task, so a B3 null under LoRA is even harder to attribute
than ADR-0005 already warns.

---

## Step 4 — Load a trained Group and run inference

A finished B1 or B2 run leaves a checkpoint you can load and query. What a LoRA
run writes is an **adapter** — about 9 MB — not a full model; loading it means
loading the base Student and applying the adapter on top.

```
runs/checkpoints/b1-<key>/checkpoint/
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

`--checkpoint runs/checkpoints/b1-<key>/checkpoint` works too if you would rather name
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
uv run hf upload <your-org>/<your-repo> merged/ceed-b1 .
```

The trailing `.` is the path *in the repo*, and it matters: it defaults to the
local path, so omitting it puts the checkpoint in a `merged/ceed-b1/`
subdirectory of the repo rather than at the root, where `from_pretrained` will
not find it. Add `--private` if the repo does not exist yet and should not be
public — the flag is ignored once the repo exists.

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

Training checkpoints and resumes, so a multi-day Group survives preemption. A
Group that has already reached its step budget is **not retrained** — its frozen
checkpoint is reused, which is what stops a baseline being retrained for every
later comparison.

```bash
# ran 4 steps, then asked for 6: runs the 2 remaining
uv run python scripts/run_group.py --group b1 --steps 4
uv run python scripts/run_group.py --group b1 --steps 6
```

The checkpoint directory is named `<group>-<key>`, where the key is the run hash
with the *step budget* normalised away. Everything else about a Group — its
objective, its signals, its batch size, its layer mapping, its seed — changes what
training does, so a Group whose configuration changed gets a different directory
and starts fresh. Raising the budget is the one change that is genuinely a
continuation, because the example order is a pure function of the global position.

`progress.json` records the configuration that wrote it, and a checkpoint written
by a different one is **refused** rather than adopted:

```
the checkpoint here was written by configuration 335406898cb8, but Group B1 is
31802aeade42. Resuming it would report the older configuration's training under
this one's name.
```

That message is worth understanding, because the failure it prevents is silent:
a completed checkpoint from a superseded configuration would be picked up, train
zero steps because it is already "complete", and be scored and reported as the
new Group.

## How the numbers are produced

Every Group is scored by the **same** evaluator (`ceed_eval.DirectEvaluator`):
the same held-out split, greedy decoding enforced in code (A9), and the metric
each dataset is reported under — ANLS for DocVQA, exact match for GQA, relaxed
accuracy for ChartQA. That identity is what makes B0 to B5 comparable to each
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
uv run python scripts/extract_teacher_artifacts.py --corpus <corpus> --store <store> \
    --group b2 --limit 4 --top-k 32 --skip-correctness
uv run python scripts/run_group.py --group b2 --corpus <corpus> --store <store> \
    --steps 4 --train-limit 4 --limit 3
```

Observed: B0 scored 1.0 on the synthetic questions; B1 trained with `kd = 0.0`
exactly (no teacher); B2 trained with a genuine non-zero `kd` term read from the
26B Teacher's cached logits; and a follow-up B2 run with a larger budget resumed
rather than restarting.

## Troubleshooting

**"Group B2 reads the Teacher's cached artefacts but no artifact store exists at
…"** — run step 2. The message includes the exact command, with `--all` if the
Group needs it.

**"store … is missing artefact kind(s) ['combine_weights']"** — the store was
extracted without the kind this Group's signal reads. Re-extract into a new
`--store` directory with `--all`; a store's schema cannot be widened in place.

**"store at … already exists with different metadata"** — you asked for more
artefact kinds than the store at that path was created with. Same fix: a new
`--store` directory. The old store stays valid, so Groups already trained against
it are not invalidated and are not retrained.

**"teacher layer 17 was not cached; this store holds [4, 9, 14, 19, 24, 29]"** —
`base.yaml`'s `layer_mapping` names a teacher layer the extraction did not cache.
Either map to a cached layer or re-extract with `--hidden-state-layers`.

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
