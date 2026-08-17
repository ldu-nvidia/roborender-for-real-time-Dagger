#!/bin/bash
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=diag_frozen58
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/diag_frozen58_%j.log
set -e
export HOME=/sailhome/ravenh
export XDG_CACHE_HOME=/svl/u/ravenh/.cache_xdg XDG_DATA_HOME=/svl/u/ravenh/.share_xdg
export TMPDIR=/svl/u/ravenh/tmp OMNI_KIT_ACCEPT_EULA=YES
source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
cd /svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real
CKPT=/svl/u/ravenh/fast_roborender/diag_frozen58/last.pt
echo "ARM=${ARM} evaluating $CKPT relabel=${IMAGE_DAGGER_EVAL_RELABEL:-0}"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=$CKPT \
    +num_envs=4 +headless=True \
    +simulator.config.render_results=true
echo EVAL_DONE
