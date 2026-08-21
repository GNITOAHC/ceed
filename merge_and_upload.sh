#!/usr/bin/env bash
# Step 3 of 3 — fold each Group's adapter into the base weights, write a model
# card, and (only when asked) upload to the Hub.
#
#   ./merge_and_upload.sh                    # merge + verify + card, no upload
#   UPLOAD=1 ./merge_and_upload.sh           # ... and upload, private
#   UPLOAD=1 PUBLIC=1 ./merge_and_upload.sh  # ... and upload, public
#   ./merge_and_upload.sh b2                 # one Group
#
# Nothing is published unless UPLOAD=1. Read the generated README.md before you
# set it: the card states that these are LoRA results and that the scores are
# not leaderboard-comparable, and both of those claims are load-bearing.
#
# B0 is skipped. It trains nothing, so its "checkpoint" is the unmodified
# google/gemma-4-e4b-it -- there is nothing to merge and nothing worth uploading.

set -euo pipefail

CORPUS="${CORPUS:-data/corpus-all}"
OUTPUT="${OUTPUT:-runs}"
STEPS="${STEPS:-4800}"
MERGED="${MERGED:-merged}"
HF_OWNER="${HF_OWNER:-gnitoahc}"
UPLOAD="${UPLOAD:-0}"
PUBLIC="${PUBLIC:-0}"
# Each merged checkpoint is ~15 GB. Keeping one at a time holds the peak at 15 GB
# instead of 75 GB; set KEEP=1 if you have the disk and would rather not re-merge.
KEEP="${KEEP:-0}"

TARGET_GROUPS=("$@")
[ ${#TARGET_GROUPS[@]} -eq 0 ] && TARGET_GROUPS=(b1 b2 b3 b4 b5)

mkdir -p "$MERGED"

# B1 is the no-teacher control, and every distilled Group's card states how it
# did against it. A card that quotes a distillation gain without the control
# beside it is the one number a reader would most want and least be given.
control=""

# Resolve a Group to the run directory it actually wrote, by recomputing the
# config hash from this corpus and this budget. Matching on group_code alone
# would pick up runs from superseded configurations, of which there are several.
run_dir_for() {
    uv run python - "$1" "$CORPUS" "$OUTPUT" "$STEPS" <<'PY'
import importlib.util
import sys
from pathlib import Path

group, corpus, output, steps = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
spec = importlib.util.spec_from_file_location("rg", "scripts/run_group.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)
from ceed_core import run_hash  # noqa: E402

argv = ["--group", group, "--corpus", corpus]
if group != "b0":
    argv += ["--steps", steps]
path = output / run_hash(rg.resolve_config(rg.parse_args(argv)))
if not (path / "run_record.json").exists():
    sys.exit(f"no run record at {path} -- has {group} been run on this corpus at {steps} steps?")
print(path)
PY
}

control="$(run_dir_for b1)"

for group in "${TARGET_GROUPS[@]}"; do
    echo "=== $group"
    run="$(run_dir_for "$group")"
    out="$MERGED/ceed-$group"
    repo="$HF_OWNER/ceed-$group"
    echo "    run  $run"
    echo "    out  $out"

    # -- merge, and verify by reloading the way a downstream user would -------
    uv run python scripts/merge_adapter.py \
        --run "$run" \
        --output "$out" \
        --overwrite \
        --verify --corpus "$CORPUS" --limit 8

    # -- model card ----------------------------------------------------------
    uv run python - "$group" "$run" "$out" "$CORPUS" "$repo" "$control" <<'PY'
import json
import sys
from pathlib import Path

group, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
corpus, repo, control = Path(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
record = json.loads((run / "run_record.json").read_text())
metrics = record["metrics"]
manifest = json.loads((corpus / "corpus_manifest.json").read_text())
control_metrics = json.loads((control / "run_record.json").read_text())["metrics"]

TITLE = {
    "B1": "supervised fine-tuning, no teacher",
    "B2": "distilled from a sparse MoE teacher",
    "B3": "distilled, with hidden-state projection",
    "B4": "distilled, with visual-advantage reweighting",
    "B5": "distilled, with router combine-weight probing",
}
OBJECTIVE = {
    "B1": "cross-entropy on gold answers only, with **no teacher** -- the control that "
          "says how much of a distilled Group's gain is distillation rather than "
          "fine-tuning",
    "B2": "cross-entropy **plus top-k logit distillation** from the teacher",
    "B3": "B2's objective plus **hidden-state projection distillation** at three mapped layers",
    "B4": "B2's objective plus **VA-OPD's visual-advantage reweighting** (arXiv:2605.21924)",
    "B5": "B2's objective plus **router combine-weight probing**",
}
METRIC = {"docvqa": "ANLS", "gqa": "exact match", "chartqa": "relaxed accuracy"}
code = record["group_code"]

scores = "\n".join(
    f"| {d} | {METRIC[d]} | {metrics[d]:.4f} | {int(metrics[d + '.n'])} |"
    for d in ("docvqa", "gqa", "chartqa")
    if d in metrics
)
counts = ", ".join(f"{k} {v:,}" for k, v in sorted(manifest["dataset_counts"].items()))

# State the control's result rather than characterising it. On this corpus B1
# beat every distilled Group on every dataset, and a card that omitted that
# while advertising distillation would be selling a result the numbers do not
# support.
if code == "B1":
    control_note = (
        "- **This Group has no teacher.** B1 is the study's control: it isolates "
        "how much of a distilled Group's gain is distillation rather than plain "
        "fine-tuning. It is not itself a distillation result."
    )
else:
    beaten = [
        f"{d} {metrics[d]:.4f} vs {control_metrics[d]:.4f}"
        for d in ("docvqa", "gqa", "chartqa")
        if d in metrics and d in control_metrics
    ]
    worse = sum(
        1
        for d in ("docvqa", "gqa", "chartqa")
        if d in metrics and d in control_metrics and metrics[d] < control_metrics[d]
    )
    verdict = (
        "scored **above this checkpoint on every dataset**"
        if worse == len(beaten)
        else "is competitive with this checkpoint"
    )
    control_note = (
        f"- **The distillation gain is not established.** The no-teacher control "
        f"(CEED B1), trained identically but with `kd_weight: 0`, {verdict} "
        f"({'; '.join(beaten)}). Whatever this checkpoint's objective contributes, "
        f"it is not visible as an advantage over supervised fine-tuning here."
    )

card = f"""---
license: gemma
base_model: google/gemma-4-e4b-it
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - visual-question-answering
  - knowledge-distillation
  - lora
  - merged
  - research
datasets:
  - lmms-lab/DocVQA
  - lmms-lab/GQA
  - lmms-lab/ChartQA
language:
  - en
---

# CEED {code} — gemma-4-e4b-it, {TITLE[code]}

A LoRA fine-tune of [`google/gemma-4-e4b-it`](https://huggingface.co/google/gemma-4-e4b-it)
trained with {OBJECTIVE[code]}.
{"The teacher is the sparse mixture-of-experts [`google/gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26b-a4b-it)." if code != "B1" else ""}

The adapter has been folded into the base weights, so this is a standalone
checkpoint: load it exactly like the base model, with no PEFT and no CEED code.

This is **Group {code}** of the CEED study (Causal Expert–Evidence Distillation),
a research artifact published for reproducibility. It is not a product.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("{repo}", dtype="float16")
processor = AutoProcessor.from_pretrained("{repo}")
```

The model was trained and scored with a short-answer instruction in the prompt.
Without it an instruction-tuned model answers `"The total written in the image is
**28**."` against gold `"28"` and scores zero on every metric here.

## Training

| | |
| --- | --- |
| Corpus | {counts} ({sum(manifest['dataset_counts'].values()):,} examples, 80/10/10 split by example id) |
| Passes over the training split | {metrics.get('train.epochs', 0):.2f} |
| Adapter | LoRA rank {int(metrics.get('train.lora_rank', 0))} |
| Final cross-entropy | {metrics.get('train.cross_entropy', float('nan')):.4f} |
| Final KD term | {metrics.get('train.kd', float('nan')):.4f} |
| Seed | {record['seed']} |
| Run identity | `{record['config_hash']}` |

## Evaluation

| Dataset | Metric | Score | n |
| --- | --- | --- | --- |
{scores}

Scored by CEED's own harness (`harness_version: {record.get('harness_version')}`)
with greedy decoding, on CEED's own 10% validation split.

**These numbers are not comparable to published DocVQA / GQA / ChartQA leaderboard
results.** Different splits, different prompt, different decoding. They are
meaningful only against the other CEED Groups, which were scored identically.

## Limitations

- **This is a LoRA result.** Merging folds the adapter into the weights; it does
  not turn a rank-{int(metrics.get('train.lora_rank', 0))} adapter into a full
  fine-tune. CEED's own ADR-0005 bars LoRA numbers from the study's headline
  table, because a null result under a small adapter cannot be attributed between
  "the signal does not transfer" and "the adapter lacked the capacity to hold it".
  Read any comparison involving this checkpoint with that in mind.
{control_note}
- Trained on document, natural-image and chart VQA in English only. Behaviour
  outside that is untested.
- Inherits the base model's limitations and the Gemma licence.

`ceed_provenance.json` beside the weights carries the source run's identity,
parameter-efficiency mode, and metrics.
"""
(out / "README.md").write_text(card)
print(f"[card] wrote {out / 'README.md'}")
PY

    # -- upload, only when asked --------------------------------------------
    if [ "$UPLOAD" = "1" ]; then
        visibility=(--private)
        [ "$PUBLIC" = "1" ] && visibility=()
        echo "    uploading to $repo"
        uv run hf upload "$repo" "$out" . "${visibility[@]}"
        echo "    https://huggingface.co/$repo"
    else
        echo "    not uploading (set UPLOAD=1). Review $out/README.md first, then:"
        echo "      uv run hf upload $repo $out . --private"
    fi

    if [ "$KEEP" != "1" ] && [ "$UPLOAD" = "1" ]; then
        rm -rf "$out"
        echo "    removed $out (15 GB reclaimed; set KEEP=1 to keep it)"
    fi
    echo
done

echo "=== done"
