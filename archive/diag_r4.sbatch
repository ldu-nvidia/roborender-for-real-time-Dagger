#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=eval_r4
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/diag_r4_origcfg_%j.log
set -e
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
export XDG_CACHE_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/xdg XDG_DATA_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/share
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp OMNI_KIT_ACCEPT_EULA=YES
source /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/conda/etc/profile.d/conda.sh
conda activate viral
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/GR00T-VisualSim2Real
echo 'round-4 mid-run ckpt under ORIGINAL offline-BC config, raytraced obs'
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/diag_r4_origcfg/last.pt \
    +num_envs=4 +headless=True \
    +simulator.config.render_results=true
echo EVAL_DONE
