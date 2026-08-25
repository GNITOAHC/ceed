#!/usr/bin/env bash
# Step 2 of 3 — read the results off disk and check they describe one comparison.
#
#   ./collect_results.sh
#
# Prints one row per Group, and refuses to look tidy if the rows are not
# comparable: a Group at a different step budget, a different seed, or a
# different corpus is not a row in the same table, and a table is exactly where
# that stops being visible.

set -euo pipefail

CORPUS="${CORPUS:-data/corpus-all}"
OUTPUT="${OUTPUT:-runs}"
STEPS="${STEPS:-4800}"

uv run python - "$CORPUS" "$OUTPUT" "$STEPS" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

corpus, output, steps = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

# Resolve each Group the way run_group.py does, against this corpus and this
# budget. That gives the exact config_hash the run wrote to, so the row is
# matched rather than guessed at -- several runs of the same Group exist on disk
# under superseded configurations, and picking "the newest B2" would silently
# report one of those.
spec = importlib.util.spec_from_file_location("rg", "scripts/run_group.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)
from ceed_core import run_hash  # noqa: E402

GROUPS = ["b0", "b1", "b2", "b3", "b4", "b5"]
DATASETS = ["docvqa", "gqa", "chartqa"]
METRIC = {"docvqa": "ANLS", "gqa": "exact match", "chartqa": "relaxed acc"}

rows, missing = [], []
for group in GROUPS:
    argv = ["--group", group, "--corpus", str(corpus)]
    if group != "b0":
        argv += ["--steps", steps]
    config = rg.resolve_config(rg.parse_args(argv))
    path = output / run_hash(config) / "run_record.json"
    if not path.exists():
        missing.append((group, run_hash(config)[:12]))
        continue
    rows.append((group, json.loads(path.read_text())))

if rows:
    from ceed_data.manifest import corpus_fingerprint

    print(f"corpus {corpus}  fingerprint {corpus_fingerprint(corpus)}  steps {steps}")
    print("metrics: " + ", ".join(f"{d} = {METRIC[d]}" for d in DATASETS) + "\n")
    head = f"{'Group':6} {'hash':14} {'epochs':>7} {'mode':6}"
    for d in DATASETS:
        head += f" {d:>22}"
    print(head)
    print("-" * len(head))

for group, record in rows:
    m = record["metrics"]
    line = (
        f"{record['group_code']:6} {record['config_hash'][:12]:14} "
        f"{m.get('train.epochs', 0):7.2f} {record['param_efficiency']:6}"
    )
    for d in DATASETS:
        cell = f"{m[d]:.4f} (n={int(m[d + '.n'])})" if d in m else "-"
        line += f" {cell:>22}"
    print(line)

print()
for group, record in rows:
    aux = {k: v for k, v in record["metrics"].items() if k.startswith("train.aux.")}
    if aux:
        print(f"    {record['group_code']} auxiliary: {aux}")

sys.stdout.flush()
if missing:
    print("\nnot yet run (or run under a different corpus/budget):", file=sys.stderr)
    for group, expected in missing:
        print(f"    {group}  expected runs/{expected}...", file=sys.stderr)

# -- the checks that decide whether this is one table -----------------------

problems = []
seeds = {r["seed"] for _, r in rows}
if len(seeds) > 1:
    problems.append(f"mixed seeds {seeds}: the Groups did not walk the corpus in one order")

# Only Groups that trained have a parameter-efficiency mode worth comparing. B0
# trains nothing, so its `full` is the config default describing an untouched
# model rather than a fine-tuning run, and counting it here would flag every
# complete table as incomparable.
trained = [r for _, r in rows if r["checkpoint_dir"]]
modes = {r["param_efficiency"] for r in trained}
if len(modes) > 1:
    problems.append(f"mixed modes {modes}: a LoRA row and a full-fine-tune row are not comparable")

epochs = {round(r["metrics"]["train.epochs"], 3) for _, r in rows if "train.epochs" in r["metrics"]}
if len(epochs) > 1:
    problems.append(f"mixed step budgets (epochs {sorted(epochs)}): the Groups trained different amounts")

fingerprints = {r["extraction_fingerprint"] for _, r in rows}
if len(fingerprints) > 1:
    problems.append(f"mixed extraction fingerprints {fingerprints}: different teacher artefacts")

sys.stdout.flush()
if problems:
    print("\nTHESE ROWS ARE NOT ONE COMPARISON:", file=sys.stderr)
    for p in problems:
        print(f"    - {p}", file=sys.stderr)
    sys.exit(1)

sys.stdout.flush()

if rows and "lora" in modes:
    print(
        "\nNote: these are LoRA runs. ADR-0005 -- no LoRA number may appear in the\n"
        "Phase 1 comparison table, because a null result under a rank-4 adapter\n"
        "cannot be attributed between 'the signal does not transfer' and 'the\n"
        "adapter lacked the capacity to hold it'."
    )
PY
