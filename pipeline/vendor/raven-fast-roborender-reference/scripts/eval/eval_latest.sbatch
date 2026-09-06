#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=eval_student_b
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/eval_student_b_%j.log
set -e
export HOME=/sailhome/ravenh
export XDG_CACHE_HOME=/svl/u/ravenh/.cache_xdg XDG_DATA_HOME=/svl/u/ravenh/.share_xdg
export TMPDIR=/svl/u/ravenh/tmp OMNI_KIT_ACCEPT_EULA=YES
source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
cd /svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real
RUN=$(ls -td logs_rl/videogen_offline_bc/*/ | head -1)
CKPT=$(ls "$RUN"model_step_*.pt | sort -t_ -k3 -n | tail -1)
echo "evaluating $CKPT"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=$CKPT \
    +num_envs=4 +headless=True \
    +simulator.config.render_results=true
echo EVAL_DONE
