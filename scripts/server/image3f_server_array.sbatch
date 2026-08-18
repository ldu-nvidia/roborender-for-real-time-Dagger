#!/bin/bash
# Production Image3F fleet element. Submit through scripts/scaling/submit_production.sh
# so every array task receives a unique endpoint and queue directory.
#SBATCH --job-name=i3f_scale
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/i3f_scale_%A_%a.log
set -Eeuo pipefail

ROOT=${ROOT:-/svl/u/ravenh/fast_roborender}
RUN_DIR=${RUN_DIR:-${FLEET_DIR:?FLEET_DIR is required; use scripts/scaling/submit_production.sh}}
TASK_ID=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}
INDEX=$(printf '%03d' "$TASK_ID")

export HOME=${HOME_OVERRIDE:-/sailhome/ravenh}
export PYTHONDONTWRITEBYTECODE=1
export IMAGE3F_ROOT=${IMAGE3F_ROOT:-/vision/u/ravenh/roborender_image_model_accelerated}
export ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-/vision/u/ravenh/roborender_image3f_portable_buildonly_v1/assets/models}
export ENDPOINT_FILE="$RUN_DIR/endpoints/renderer_${INDEX}.txt"
export IMAGE3F_QUEUE="$RUN_DIR/queues/renderer_${INDEX}"
export IMAGE3F_SERVER_ID="${IMAGE3F_RUN_ID:-${SLURM_ARRAY_JOB_ID}}:${INDEX}"
# Let the kernel choose a collision-free port. The server publishes the bound port atomically.
export IMAGE3F_PORT=0
export IMAGE3F_MAX_STREAMS=${IMAGE3F_MAX_STREAMS:-16}
export IMAGE3F_ALLOW_PY_QUEUE=0

mkdir -p "$RUN_DIR/endpoints" "$IMAGE3F_QUEUE/inbox" "$IMAGE3F_QUEUE/done"
rm -f "$ENDPOINT_FILE" "$IMAGE3F_QUEUE/READY"

# Avoid 26+ cold model loads hitting shared storage in the same second.
STARTUP_STAGGER_S=${STARTUP_STAGGER_S:-5}
if [ "$STARTUP_STAGGER_S" -gt 0 ]; then
    sleep $((TASK_ID * STARTUP_STAGGER_S))
fi

cleanup() {
    rm -f "$ENDPOINT_FILE" "$IMAGE3F_QUEUE/READY"
}
trap cleanup EXIT INT TERM

"$IMAGE3F_ROOT/.venv/bin/python" "$ROOT/scripts/server/image3f_server.py"
