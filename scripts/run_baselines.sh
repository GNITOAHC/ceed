#!/usr/bin/env bash
# Run the whole baseline set, B0 through B5, one Group per GPU.
#
# Every Group here reads one corpus and one artifact store, so the only thing
# that differs between them is the auxiliary signal — which is the entire point
# of the comparison, and the reason this is a script rather than six commands
# typed by hand at different times.
#
# Each Group is pinned to a single GPU with CUDA_VISIBLE_DEVICES, because the
# Student is 8B in fp16 and fits on one V100 with room for a document-length
# sequence. Groups are distributed over the available cards and run in parallel;
# the lanes below are sized so the longest one is two Groups, not three.
#
# It is resumable. Each Group's checkpoint is frozen and reused, so a Group that
# has already reached its step budget is skipped rather than retrained, and a run
# killed halfway continues from its last checkpoint. Re-running this script after
# a preemption is the correct recovery.
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

# One lane per GPU, in the order they should run. This is a scheduling choice
# and nothing more — the Groups are independent, so any partition trains the
# same six models. B0 trains nothing and B1 has no teacher to read, so both are
# cheap and share the first lane.
LANES=("b0 b1 b5" "b2" "b3" "b4")

run_lane() {
    local gpu="$1"
    shift
    for group in "$@"; do
        local log="$LOGDIR/$group.log"
        echo "[lane $gpu] starting $group -> $log ($(date -Is))"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
            uv run python scripts/run_group.py \
            --group "$group" \
            --corpus "$CORPUS" \
            --store "$STORE" \
            --output "$OUTPUT" \
            >"$log" 2>&1
        echo "[lane $gpu] finished $group ($(date -Is))"
    done
}

pids=()
for gpu in "${!LANES[@]}"; do
    # shellcheck disable=SC2086
    run_lane "$gpu" ${LANES[$gpu]} &
    pids+=("$!")
    # Staggered, so four processes do not all pull 16 GB of weights through the
    # host at the same moment.
    sleep 90
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo "=== all lanes done ($(date -Is)), status $status"
exit "$status"
