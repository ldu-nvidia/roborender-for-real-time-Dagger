#!/usr/bin/env bash
# Scale Raven's collector over H200s. Total environment count is the only
# scientific parameter changed from Raven's configuration.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
TOTAL_ENVS=${1:-256}
TARGET_CHUNKS=${2:-4096}
RUN_TAG=${ROBORENDER_BC_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${ROBORENDER_BC_ISAAC_OUTPUT_ROOT:-$ROOT/data/BC/isaac}
RUN_ROOT=$ROOT/data/BC/.collection_runs/$RUN_TAG
STAGING_ROOT=$RUN_ROOT/staging
GPU_LIST=${ROBORENDER_BC_GPUS:-0,4}
# Raven's source configuration uses 32 environments per Isaac process. Four
# sequential shards per GPU preserve that load while covering 256 unique
# environment/seed slots over the two currently selected physical GPUs.
SHARDS_PER_GPU=${ROBORENDER_BC_SHARDS_PER_GPU:-4}

[[ $TOTAL_ENVS =~ ^[1-9][0-9]*$ ]] || {
    echo "usage: $0 [TOTAL_ENVIRONMENTS] [TARGET_RAVEN_CHUNKS]" >&2
    exit 2
}
[[ $TARGET_CHUNKS =~ ^[1-9][0-9]*$ ]] || {
    echo "TARGET_RAVEN_CHUNKS must be positive" >&2
    exit 2
}
IFS=, read -r -a gpus <<< "$GPU_LIST"
gpu_count=${#gpus[@]}
(( gpu_count > 0 )) || { echo "GPU list is empty" >&2; exit 2; }
[[ $SHARDS_PER_GPU =~ ^[1-9][0-9]*$ ]] || {
    echo "ROBORENDER_BC_SHARDS_PER_GPU must be positive" >&2
    exit 2
}
world_size=$((gpu_count * SHARDS_PER_GPU))
(( TOTAL_ENVS % world_size == 0 )) || {
    echo "TOTAL_ENVIRONMENTS must divide evenly over $world_size collection shards" >&2
    exit 2
}
envs_per_rank=$((TOTAL_ENVS / world_size))
(( TARGET_CHUNKS % world_size == 0 )) || {
    echo "TARGET_RAVEN_CHUNKS must divide evenly over $world_size collection shards" >&2
    exit 2
}
chunks_per_rank=$((TARGET_CHUNKS / world_size))

python3 "$ROOT/pipeline/bc_collection/validate_raven_bc_collection_contract.py" >/dev/null
check_sha256() {
    local path=$1 expected=$2 actual
    [[ -f $path ]] || { echo "required checkpoint missing: $path" >&2; exit 2; }
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ $actual == "$expected" ]] || {
        echo "checkpoint identity mismatch: $path" >&2
        exit 2
    }
}
check_sha256 \
    "$ROOT/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt" \
    943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38
check_sha256 \
    "$ROOT/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt" \
    2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e
check_sha256 \
    "$ROOT/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt" \
    ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054
[[ ! -e $RUN_ROOT ]] || {
    echo "refusing to reuse collection run metadata: $RUN_ROOT" >&2
    exit 2
}
[[ ! -e $OUTPUT_ROOT || -z $(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit) ]] || {
    echo "canonical BC directory must be empty: $OUTPUT_ROOT" >&2
    exit 2
}
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT/logs" "$STAGING_ROOT"

pids=()
stop_group() {
    local pgid=$1
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 -- "-$pgid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
}
stop_all() {
    local pid
    for pid in "${pids[@]:-}"; do
        [[ -n $pid ]] && stop_group "$pid"
    done
}
trap stop_all INT TERM
for ((wave=0; wave<SHARDS_PER_GPU; wave++)); do
    pids=()
    finished=()
    for ((slot=0; slot<gpu_count; slot++)); do
        gpu=${gpus[$slot]}
        rank=$((wave * gpu_count + slot))
        [[ $gpu =~ ^[0-9]+$ ]] || { echo "invalid GPU index: $gpu" >&2; exit 2; }
        rank_stage=$STAGING_ROOT/rank-$(printf '%02d' "$rank")
        mkdir -p "$rank_stage"
        setsid "$ROOT/pipeline/bc_collection/run_raven_bc_rank.sh" \
            "$gpu" "$rank" "$world_size" "$envs_per_rank" "$rank_stage" "$RUN_TAG" \
            >"$RUN_ROOT/logs/rank-$(printf '%02d' "$rank").log" 2>&1 &
        pids+=("$!")
        finished+=(0)
    done

    while :; do
        remaining=0
        status_line="wave${wave}"
        for ((slot=0; slot<gpu_count; slot++)); do
            (( finished[slot] == 0 )) || continue
            rank=$((wave * gpu_count + slot))
            rank_stage=$STAGING_ROOT/rank-$(printf '%02d' "$rank")
            rank_log=$RUN_ROOT/logs/rank-$(printf '%02d' "$rank").log
            if rg -q 'ERROR_DEVICE_LOST|GPU crash is detected|A GPU crash occurred|CUDA error: an illegal memory access' "$rank_log"; then
                echo "rank $rank reported a CUDA/Vulkan GPU failure; inspect $rank_log" >&2
                stop_all
                exit 1
            fi
            complete=$(python3 - "$rank_stage" "$RUN_TAG" "$rank" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "pipeline/bc_collection"))
from publish_raven_bc_quota import rank_files
print(len(rank_files(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))))
PY
)
            status_line+=" rank${rank}=${complete}/${chunks_per_rank}"
            if (( complete >= chunks_per_rank )); then
                stop_group "${pids[$slot]}"
                finished[$slot]=1
            elif kill -0 "${pids[$slot]}" 2>/dev/null; then
                remaining=1
            else
                echo "rank $rank exited before its quota; inspect $RUN_ROOT/logs" >&2
                stop_all
                exit 1
            fi
        done
        printf '%s %s\n' "$(date -u +%FT%TZ)" "$status_line"
        (( remaining != 0 )) || break
        sleep 5
    done

    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
done
pids=()
trap - INT TERM
python3 "$ROOT/pipeline/bc_collection/publish_raven_bc_quota.py" \
    --staging-root "$STAGING_ROOT" \
    --dataset-dir "$OUTPUT_ROOT" \
    --run-tag "$RUN_TAG" \
    --world-size "$world_size" \
    --chunks-per-rank "$chunks_per_rank"
python3 "$ROOT/pipeline/bc_collection/seal_raven_bc_collection.py" \
    --dataset-dir "$OUTPUT_ROOT" \
    --run-tag "$RUN_TAG" \
    --world-size "$world_size" \
    --envs-per-rank "$envs_per_rank" \
    --contract "$ROOT/pipeline/bc_collection/raven_bc_collection_contract.v1.json"
echo "Raven BC collection sealed with exactly $TARGET_CHUNKS chunks: $OUTPUT_ROOT"
