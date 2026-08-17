#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=videogen_render
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/offline_render_%j.log
# CPU-only: streams chunks to the standing raven_server job
set -e
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
exec roborender-acceleration/.venv/bin/python offline_render_dataset.py \
    --dataset_dir /svl/u/ravenh/fast_roborender/videogen_dataset \
    --prompt_override "$(cat /svl/u/ravenh/fast_roborender/render_prompt.txt)" \
    --views "${RENDER_VIEWS:-hand_left,hand_right,head_color}" \
    ${RENDER_ARGS:---follow}
