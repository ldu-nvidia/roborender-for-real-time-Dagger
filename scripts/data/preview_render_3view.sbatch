#!/bin/bash
# Render a small preview set with per-view RGB, for approval before a full re-render.
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=preview_3view
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/preview_3view_%j.log
set -euo pipefail
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
EP=/svl/u/ravenh/fast_roborender/raven_server_endpoint.txt
echo "render server: $(cat $EP)"
exec roborender-acceleration/.venv/bin/python scripts/data/offline_render_dataset.py \
    --dataset_dir /svl/u/ravenh/fast_roborender/videogen_preview_3view \
    --endpoint_file $EP \
    --prompt_override "$(cat /svl/u/ravenh/fast_roborender/render_prompt.txt)" \
    --views hand_left,hand_right,head_color \
    --primary_view head_color \
    --preview_n 0
