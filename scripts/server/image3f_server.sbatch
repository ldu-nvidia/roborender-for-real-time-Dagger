#!/bin/bash
#SBATCH --job-name=i3f_warm_server
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/i3f_warm_server_%j.log
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
FR_ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$FR_ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" "$ROOT/scripts/server/image3f_server.sbatch"
fi
export HOME=$HOME_OVERRIDE
export PYTHONDONTWRITEBYTECODE=1
export ENDPOINT_FILE=${ENDPOINT_FILE:-$FR_ROOT/image3f_server_endpoint.txt}
export IMAGE3F_QUEUE=${IMAGE3F_QUEUE:-$FR_ROOT/image3f_queue}
# This Slurm service is consumed by trainer/evaluation jobs on other nodes.
# Remote exposure is explicit; the local round-based fleet remains loopback-only.
export IMAGE3F_BIND_HOST=${IMAGE3F_BIND_HOST:-0.0.0.0}
if [ -z "${IMAGE3F_ADVERTISE_HOST:-}" ]; then
    IMAGE3F_ADVERTISE_HOST=$(hostname -f 2>/dev/null || hostname)
fi
[ -n "$IMAGE3F_ADVERTISE_HOST" ]
export IMAGE3F_ADVERTISE_HOST
rm -f $IMAGE3F_QUEUE/READY
exec "$IMAGE3F_PYTHON" "$FR_ROOT/scripts/server/image3f_server.py"
