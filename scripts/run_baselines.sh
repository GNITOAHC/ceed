#!/usr/bin/env bash
# Run the whole baseline set, B0 through B5, in order.
#
# Every Group here reads one corpus and one artifact store, so the only thing
# that differs between them is the auxiliary signal — which is the entire point
# of the comparison, and the reason this is a script rather than six commands
# typed by hand at different times.
#
# It is resumable. Each Group's checkpoint is frozen and reused, so a Group that
# has already reached its step budget is skipped rather than retrained, and a run
# killed halfway through continues from its last checkpoint. Re-running this
# script after a preemption is the correct recovery.
#
# The store must already hold every artefact kind B3 to B5 read:
#
#   uv run python scripts/extract_teacher_artifacts.py \
#       --corpus data/corpus --store data/store-full --group b2 --split train \
#       --top-k 64 --skip-correctness --all
#
# Usage:
#   scripts/run_baselines.sh [corpus] [store] [output] [logdir]

set -euo pipefail

CORPUS="${1:-data/corpus}"
STORE="${2:-data/store-full}"
OUTPUT="${3:-runs}"
LOGDIR="${4:-logs}"

mkdir -p "$LOGDIR"

for group in b0 b1 b2 b3 b4 b5; do
    log="$LOGDIR/$group.log"
    echo "=== $group -> $log  ($(date -Is))"
    uv run python scripts/run_group.py \
        --group "$group" \
        --corpus "$CORPUS" \
        --store "$STORE" \
        --output "$OUTPUT" \
        >"$log" 2>&1
    # The record's own summary, so the console carries the numbers and not just
    # the fact that something ran.
    sed -n '/=== run record/,$p' "$log"
done

echo "=== all baselines done ($(date -Is))"
