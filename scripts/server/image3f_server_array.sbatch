#!/bin/bash
# Production Image3F fleet element. Submit through scripts/scaling/submit_production.sh
# so every array task receives a unique endpoint and queue directory.
#SBATCH --job-name=i3f_scale
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:15:00
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-$STAGE_ROOT}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" \
        "$ROOT/scripts/server/image3f_server_array.sbatch"
fi
RUN_DIR=${RUN_DIR:-${FLEET_DIR:?FLEET_DIR is required; use scripts/scaling/submit_production.sh}}
TASK_ID=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}
INDEX=$(printf '%03d' "$TASK_ID")

export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
export PYTHONDONTWRITEBYTECODE=1
export IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
export IMAGE3F_ENV=${IMAGE3F_ENV:-$STAGE_ROOT/runtime/envs/image3f-cu128}
export IMAGE3F_PYTHON=${IMAGE3F_PYTHON:-$IMAGE3F_ENV/bin/python}
export ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models}
export ENDPOINT_FILE="$RUN_DIR/endpoints/renderer_${INDEX}.txt"
export IMAGE3F_QUEUE="$RUN_DIR/queues/renderer_${INDEX}"
export IMAGE3F_SERVER_ID="${IMAGE3F_RUN_ID:-${SLURM_ARRAY_JOB_ID}}:${INDEX}"
export IMAGE3F_STOP_FILE="$RUN_DIR/STOP"
export IMAGE3F_STOP_ACK_FILE="$RUN_DIR/stop_acks/renderer_${INDEX}.json"
export IMAGE3F_STOP_POLL_S=1
# Let the kernel choose a collision-free port. The server publishes the bound port atomically.
export IMAGE3F_PORT=0
export IMAGE3F_MAX_STREAMS=${IMAGE3F_MAX_STREAMS:-16}
export IMAGE3F_ALLOW_PY_QUEUE=0
# This legacy Slurm-array topology serves remote trainer nodes.  Unlike the
# round-based local actor fleet it therefore cannot use the loopback-only bind.
export IMAGE3F_BIND_HOST=${IMAGE3F_BIND_HOST:-0.0.0.0}

mkdir -p "$RUN_DIR/endpoints" "$RUN_DIR/stop_acks" \
         "$IMAGE3F_QUEUE/inbox" "$IMAGE3F_QUEUE/done"
rm -f "$ENDPOINT_FILE" "$IMAGE3F_QUEUE/READY" "$IMAGE3F_STOP_ACK_FILE"

# Avoid 26+ cold model loads hitting shared storage in the same second.
STARTUP_STAGGER_S=${STARTUP_STAGGER_S:-5}
STAGGER_INDEX=$TASK_ID
if [ "${STARTUP_STAGGER_SCOPE:-global}" = "local" ]; then
    STAGGER_INDEX=${SLURM_LOCALID:-$TASK_ID}
fi
if [ "$STARTUP_STAGGER_S" -gt 0 ]; then
    sleep $((STAGGER_INDEX * STARTUP_STAGGER_S))
fi

cleanup() {
    rm -f "$ENDPOINT_FILE" "$IMAGE3F_QUEUE/READY"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$IMAGE3F_PYTHON" "$ROOT/scripts/server/image3f_server.py"
[ -f "$IMAGE3F_STOP_FILE" ] && [ ! -L "$IMAGE3F_STOP_FILE" ] && \
[ -s "$IMAGE3F_STOP_FILE" ] || {
    echo "FATAL: Image3F server exited zero without the exact STOP sentinel" >&2
    exit 74
}
[ -f "$IMAGE3F_STOP_ACK_FILE" ] && [ ! -L "$IMAGE3F_STOP_ACK_FILE" ] && \
[ -s "$IMAGE3F_STOP_ACK_FILE" ] || {
    echo "FATAL: Image3F server exited zero without its exact nonempty STOP ACK" >&2
    exit 74
}
