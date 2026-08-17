#!/bin/bash
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=img_dagger_r4
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/image_dagger_r4_%j.log
set -e
export HOME=/sailhome/ravenh
export XDG_CACHE_HOME=/svl/u/ravenh/.cache_xdg XDG_DATA_HOME=/svl/u/ravenh/.share_xdg
export TMPDIR=/svl/u/ravenh/tmp OMNI_KIT_ACCEPT_EULA=YES
source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
# wait for server endpoint freshness (servers 2/3 chain-start after anchors)
NOW=$(date +%s)
for i in $(seq 1 120); do
  M2=$(stat -c %Y /svl/u/ravenh/fast_roborender/image3f_server_endpoint_2.txt 2>/dev/null || echo 0)
  M3=$(stat -c %Y /svl/u/ravenh/fast_roborender/image3f_server_endpoint_3.txt 2>/dev/null || echo 0)
  [ "$M2" -gt "$((NOW - 7200))" ] && [ "$M3" -gt "$((NOW - 7200))" ] && break
  sleep 30
done
cd /svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger \
    num_envs=16 \
    headless=True \
    checkpoint=/svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real/logs_rl/image_dagger_r3/image_dagger_run3-20260805_002828/last.pt \
    experiment_name=image_dagger_run4 \
    project_name=image_dagger_r4
echo RUN_DONE
