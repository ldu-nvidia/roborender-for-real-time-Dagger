#!/bin/bash
# Re-render the dataset with per-view generated RGB (rgb_gen__<view>), 50-step teacher.
# Finite: exits when its shard is done (no --follow). SHARD / NSHARDS from the environment.
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=rerender3v
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=viscam10,viscam12
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/rerender3v_%x_%j.log
set -euo pipefail
export HOME=/sailhome/ravenh
ROOT=/svl/u/ravenh/fast_roborender
EP=$ROOT/teacher_endpoint_${SHARD}.txt
[ -f "$EP" ] || { echo "FATAL: no endpoint file $EP"; exit 1; }
echo "shard ${SHARD}/${NSHARDS} -> $(cat $EP)"
exec $ROOT/roborender-acceleration/.venv/bin/python $ROOT/scripts/data/offline_render_dataset.py \
    --dataset_dir $ROOT/videogen_dataset \
    --endpoint_file "$EP" \
    --prompt_override "$(cat $ROOT/render_prompt.txt)" \
    --views hand_left,hand_right,head_color --primary_view head_color \
    --max_chain 1 --shard ${SHARD} --nshards ${NSHARDS}
