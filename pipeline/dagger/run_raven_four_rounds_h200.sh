#!/usr/bin/env bash
# Execute Raven's four independent warm-started Image3F DAgger jobs in order.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
CHECKPOINT=${1:?usage: $0 SELECTED_THREE_VIEW_BC_CHECKPOINT [NUM_ENVS]}
NUM_ENVS=${2:-16}
[[ -f $CHECKPOINT ]] || { echo "BC checkpoint missing: $CHECKPOINT" >&2; exit 2; }
[[ $NUM_ENVS =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be positive" >&2; exit 2; }
mkdir -p "$ROOT/runtime/dagger"
MARKER=""
cleanup() {
    [[ -z $MARKER ]] || rm -f -- "$MARKER"
}
trap cleanup EXIT

for round in 1 2 3 4; do
    MARKER=$(mktemp "$ROOT/runtime/dagger/dagger-round-${round}.XXXXXX")
    "$ROOT/pipeline/dagger/run_raven_dagger_round_h200.sh" \
        "$round" "$CHECKPOINT" "$NUM_ENVS"
    mapfile -t candidates < <(
        find "$ROOT/checkpoints/students/dagger_${round}" -type f -name last.pt \
            -newer "$MARKER" -printf '%T@ %p\n' | sort -nr
    )
    (( ${#candidates[@]} > 0 )) || {
        echo "round $round completed without last.pt" >&2
        exit 2
    }
    rm -f -- "$MARKER"
    MARKER=""
    CHECKPOINT=${candidates[0]#* }
    [[ -f $CHECKPOINT ]] || { echo "round $round checkpoint vanished" >&2; exit 2; }
    echo "round_${round}_checkpoint=$CHECKPOINT"
done
