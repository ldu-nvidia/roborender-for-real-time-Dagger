#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --account=general_cs_infra
#SBATCH --job-name=mkprev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/make_previews_%j.log
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
cd "$ROOT"
exec "$WANDB_PYTHON" "$ROOT/tools/make_previews.py"
