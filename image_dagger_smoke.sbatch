#!/bin/bash
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=img_dagger_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/image_dagger_smoke_%j.log
set -e
export HOME=/sailhome/ravenh
export XDG_CACHE_HOME=/svl/u/ravenh/.cache_xdg XDG_DATA_HOME=/svl/u/ravenh/.share_xdg
export TMPDIR=/svl/u/ravenh/tmp OMNI_KIT_ACCEPT_EULA=YES
export IMAGE_DAGGER_DUMP_DIR=/svl/u/ravenh/fast_roborender/image_dagger_previews
source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
cd /svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger \
    num_envs=${NUM_ENVS:-4} \
    headless=True \
    +max_iterations=2 \
    experiment_name=image_dagger_smoke \
    project_name=image_dagger
echo SMOKE_DONE
