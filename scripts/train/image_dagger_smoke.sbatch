#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=img_dagger_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/image_dagger_smoke_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE OMNI_KIT_ACCEPT_EULA=YES
export IMAGE_DAGGER_DUMP_DIR=${IMAGE_DAGGER_DUMP_DIR:-$ROBORENDER_STAGE_ROOT/runtime/image_dagger_previews}
source "$VIRAL_VENV_ACTIVATE"
cd "$GR00T_ROOT"
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger \
    num_envs=${NUM_ENVS:-4} \
    headless=True \
    +max_iterations=2 \
    experiment_name=image_dagger_smoke \
    project_name=image_dagger
echo SMOKE_DONE
