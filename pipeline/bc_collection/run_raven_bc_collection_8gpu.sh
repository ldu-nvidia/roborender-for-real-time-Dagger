#!/usr/bin/env bash
# Scale Raven's collector over H200s. Total environment count is the only
# scientific parameter changed from Raven's configuration.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
TOTAL_ENVS=${1:-256}
GPU_LIST=${ROBORENDER_BC_GPUS:-0,1,2,3,4,5,6,7}
RUN_TAG=${ROBORENDER_BC_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${ROBORENDER_BC_ISAAC_OUTPUT_ROOT:-$ROOT/data/BC/isaac}
RUN_ROOT=$ROOT/data/BC/.collection_runs/$RUN_TAG

[[ $TOTAL_ENVS =~ ^[1-9][0-9]*$ ]] || {
    echo "usage: $0 [TOTAL_ENVIRONMENTS]" >&2
    exit 2
}
IFS=, read -r -a gpus <<< "$GPU_LIST"
world_size=${#gpus[@]}
(( world_size > 0 )) || { echo "GPU list is empty" >&2; exit 2; }
(( TOTAL_ENVS % world_size == 0 )) || {
    echo "TOTAL_ENVIRONMENTS must divide evenly over $world_size GPUs" >&2
    exit 2
}
envs_per_rank=$((TOTAL_ENVS / world_size))

python3 "$ROOT/pipeline/bc_collection/validate_raven_bc_collection_contract.py" >/dev/null
[[ ! -e $RUN_ROOT ]] || {
    echo "refusing to reuse collection run metadata: $RUN_ROOT" >&2
    exit 2
}
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT/logs"

pids=()
for rank in "${!gpus[@]}"; do
    gpu=${gpus[$rank]}
    [[ $gpu =~ ^[0-9]+$ ]] || { echo "invalid GPU index: $gpu" >&2; exit 2; }
    "$ROOT/pipeline/bc_collection/run_raven_bc_rank.sh" \
        "$gpu" "$rank" "$world_size" "$envs_per_rank" "$OUTPUT_ROOT" "$RUN_TAG" \
        >"$RUN_ROOT/logs/rank-$(printf '%02d' "$rank").log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "one or more Raven BC collector ranks failed; inspect $RUN_ROOT/logs" >&2
    exit 1
fi
python3 "$ROOT/pipeline/bc_collection/seal_raven_bc_collection.py" \
    --dataset-dir "$OUTPUT_ROOT" \
    --run-tag "$RUN_TAG" \
    --world-size "$world_size" \
    --envs-per-rank "$envs_per_rank" \
    --contract "$ROOT/pipeline/bc_collection/raven_bc_collection_contract.v1.json"
echo "Raven BC collection ranks completed: $OUTPUT_ROOT"
