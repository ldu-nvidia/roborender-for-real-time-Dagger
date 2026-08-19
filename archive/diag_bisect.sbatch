#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/diag_bisect_%x_%j.log
set -e
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
export XDG_CACHE_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/xdg XDG_DATA_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/share
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp OMNI_KIT_ACCEPT_EULA=YES
source /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/conda/etc/profile.d/conda.sh
conda activate viral
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/GR00T-VisualSim2Real
CKPT=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/diag_bisect_${ARM}/last.pt
echo "BISECT arm=${ARM} ckpt=$CKPT (frozen 58% weights, raytraced obs, no relabel)"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=$CKPT \
    +num_envs=4 +headless=True \
    +simulator.config.render_results=true
echo EVAL_DONE
