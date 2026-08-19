#!/bin/bash
# Re-render the dataset with per-view generated RGB (rgb_gen__<view>), 50-step teacher.
# Finite: exits when its shard is done (no --follow). SHARD / NSHARDS from the environment.
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=rerender3v
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/rerender3v_%x_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -euo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
EP=$ROOT/teacher_endpoint_${SHARD}.txt
[ -f "$EP" ] || { echo "FATAL: no endpoint file $EP"; exit 1; }
echo "shard ${SHARD}/${NSHARDS} -> $(cat $EP)"
exec "$ROBORENDER_ACCEL_PYTHON" "$ROOT/scripts/data/offline_render_dataset.py" \
    --dataset_dir $ROOT/videogen_dataset \
    --endpoint_file "$EP" \
    --prompt_override "$(cat $ROOT/render_prompt.txt)" \
    --views hand_left,hand_right,head_color --primary_view head_color \
    --max_chain 1 --shard ${SHARD} --nshards ${NSHARDS}
