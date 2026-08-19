#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=img_dagger_r4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/image_dagger_legacy_%j.log
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" "$ROOT/scripts/train/image_dagger_r4.sbatch"
fi
export HOME=$HOME_OVERRIDE OMNI_KIT_ACCEPT_EULA=YES
source "$VIRAL_VENV_ACTIVATE"
# wait for server endpoint freshness (servers 2/3 chain-start after anchors)
NOW=$(date +%s)
for i in $(seq 1 120); do
  M2=$(stat -c %Y "$ROOT/image3f_server_endpoint_2.txt" 2>/dev/null || echo 0)
  M3=$(stat -c %Y "$ROOT/image3f_server_endpoint_3.txt" 2>/dev/null || echo 0)
  [ "$M2" -gt "$((NOW - 7200))" ] && [ "$M3" -gt "$((NOW - 7200))" ] && break
  sleep 30
done
: "${CKPT:?CKPT must point to a staged checkpoint for this legacy warm-start run}"
cd "$GR00T_ROOT"
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger \
    num_envs=16 \
    headless=True \
    checkpoint="$CKPT" \
    experiment_name=image_dagger_run4 \
    project_name=image_dagger_r4
echo RUN_DONE
