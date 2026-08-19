#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --account=general_cs_infra
#SBATCH --job-name=vg_render_shard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/render_shard_%j.log
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
cd "$ROOT"
exec "$WANDB_PYTHON" "$ROOT/scripts/data/offline_render_dataset.py" \
    --dataset_dir "$ROOT/videogen_dataset" \
    --endpoint_file "$ROOT/teacher_endpoint_${SHARD}.txt" \
    --prompt_override "$(cat "$ROOT/render_prompt.txt")" \
    --views hand_left,hand_right,head_color --primary_view head_color \
    --max_chain 1 --shard ${SHARD} --nshards 3 --follow
