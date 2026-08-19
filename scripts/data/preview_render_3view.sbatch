#!/bin/bash
# Render a small preview set with per-view RGB, for approval before a full re-render.
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=preview_3view
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/preview_3view_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -euo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
cd "$ROOT"
EP=$ROOT/raven_server_endpoint.txt
echo "render server: $(cat $EP)"
exec "$ROBORENDER_ACCEL_PYTHON" scripts/data/offline_render_dataset.py \
    --dataset_dir "$ROOT/videogen_preview_3view" \
    --endpoint_file $EP \
    --prompt_override "$(cat "$ROOT/render_prompt.txt")" \
    --views hand_left,hand_right,head_color \
    --primary_view head_color \
    --preview_n 0
