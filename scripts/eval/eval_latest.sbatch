#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=eval_student_b
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/eval_student_b_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE OMNI_KIT_ACCEPT_EULA=YES
source "$VIRAL_VENV_ACTIVATE"
cd "$GR00T_ROOT"
RUN=$(ls -td logs_rl/videogen_offline_bc/*/ | head -1)
CKPT=$(ls "$RUN"model_step_*.pt | sort -t_ -k3 -n | tail -1)
echo "evaluating $CKPT"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=$CKPT \
    +num_envs=4 +headless=True \
    +simulator.config.render_results=true
echo EVAL_DONE
