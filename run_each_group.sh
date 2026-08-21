#!/usr/bin/env bash
# Step 1 of 3 — train and score every baseline Group, one at a time.
#
#   ./run_each_group.sh              # all six, in order
#   ./run_each_group.sh b3 b4 b5     # only these
#
# Sequential by request. The same work in four lanes is ~12.8 h instead of ~31 h:
#   setsid nohup bash scripts/run_baselines.sh data/corpus-all data/store-v2 runs logs 4800 \
#       > logs/baselines-all.log 2>&1 < /dev/null &
#
# It is resumable. A Group that reached its step budget is reused rather than
# retrained, and one killed halfway continues from its last checkpoint, so
# re-running this script is the correct recovery after a preemption.

set -euo pipefail

CORPUS="${CORPUS:-data/corpus-all}"
STORE="${STORE:-data/store-v2}"
OUTPUT="${OUTPUT:-runs}"
LOGDIR="${LOGDIR:-logs}"
GPU="${GPU:-0}"

# Optimiser steps, each consuming batch_size (8) examples by gradient
# accumulation. Over this corpus's 14,278 training examples that is
#   4800 * 8 / 14278 = 2.69 passes.
# Raising it is a genuine continuation -- the resume key normalises the budget
# away -- so 6600 (3.70 passes) can be run later over the same checkpoints
# without starting from scratch. What matters for the comparison is only that
# every Group shares whatever value is set here.
STEPS="${STEPS:-4800}"

TARGET_GROUPS=("$@")
[ ${#TARGET_GROUPS[@]} -eq 0 ] && TARGET_GROUPS=(b0 b1 b2 b3 b4 b5)

mkdir -p "$LOGDIR"

# -- preflight: fail now, not six hours in ----------------------------------

[ -f "$CORPUS/corpus_manifest.json" ] || {
    echo "no corpus at $CORPUS -- run scripts/build_corpus.py first" >&2
    exit 1
}
[ -d "$STORE" ] || {
    echo "no artifact store at $STORE -- run scripts/extract_teacher_artifacts.py first" >&2
    exit 1
}

echo "=== corpus  $CORPUS"
uv run python -c "
import json
m = json.load(open('$CORPUS/corpus_manifest.json'))
print('    datasets', m['dataset_counts'])
print('    splits  ', {k: len(v) for k, v in m['splits'].items()})
"
echo "=== store   $STORE"
echo "=== steps   $STEPS  (x8 examples per step)"
echo "=== groups  ${TARGET_GROUPS[*]}"
echo

# -- run them ---------------------------------------------------------------
#
# Each Group is allowed to fail without taking the rest of the night with it,
# but a failure is remembered and reported at the end. `pipefail` is what makes
# that work at all: without it the pipeline's status is tee's, and a Group that
# died of CUDA OOM in minute three exits 0 and reads as a completed run.

set -o pipefail
declare -A STATUS=()
overall=0

for group in "${TARGET_GROUPS[@]}"; do
    log="$LOGDIR/all-$group.log"
    budget=()
    [ "$group" != "b0" ] && budget=(--steps "$STEPS")   # B0 trains nothing

    echo "=== $group starting $(date -Is) -> $log"
    if CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
        uv run python scripts/run_group.py \
            --group "$group" \
            --corpus "$CORPUS" \
            --store "$STORE" \
            --output "$OUTPUT" \
            "${budget[@]}" 2>&1 | tee "$log"
    then
        STATUS[$group]=ok
        echo "=== $group finished $(date -Is)"
    else
        STATUS[$group]=FAILED
        overall=1
        echo "=== $group FAILED $(date -Is) -- see $log" >&2
    fi
    echo
done

echo "=== summary $(date -Is)"
for group in "${TARGET_GROUPS[@]}"; do
    printf '    %-4s %s\n' "$group" "${STATUS[$group]}"
done

[ "$overall" -eq 0 ] && echo "=== all groups done; next: ./collect_results.sh"
exit "$overall"
