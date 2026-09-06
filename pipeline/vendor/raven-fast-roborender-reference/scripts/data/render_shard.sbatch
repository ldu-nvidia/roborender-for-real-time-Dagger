#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=vg_render_shard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/render_shard.log
set -e
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
exec /svl/u/ravenh/fast_roborender/roborender-acceleration/.venv/bin/python /svl/u/ravenh/fast_roborender/scripts/data/offline_render_dataset.py \
    --dataset_dir /svl/u/ravenh/fast_roborender/videogen_dataset \
    --endpoint_file /svl/u/ravenh/fast_roborender/teacher_endpoint_${SHARD}.txt \
    --prompt_override "$(cat /svl/u/ravenh/fast_roborender/render_prompt.txt)" \
    --views hand_left,hand_right,head_color --primary_view head_color \
    --max_chain 1 --shard ${SHARD} --nshards 3 --follow
