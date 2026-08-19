#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=eval_r4_final300
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/eval_r4_final300_%j.log
set -e
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
export XDG_CACHE_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/xdg XDG_DATA_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/share
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp OMNI_KIT_ACCEPT_EULA=YES
source /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/conda/etc/profile.d/conda.sh
conda activate viral
# stage: final round-4 weights + ORIGINAL offline-BC eval config
R=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/GR00T-VisualSim2Real/logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202
D=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/diag_r4_final
mkdir -p $D
# checkpoint already staged and md5-verified; re-copy only if size differs
SRC_SZ=$(stat -c %s $R/last.pt); DST_SZ=$(stat -c %s $D/last.pt 2>/dev/null || echo 0)
[ $SRC_SZ = $DST_SZ ] || cp $R/last.pt $D/last.pt
cp /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/GR00T-VisualSim2Real/logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447/config.yaml $D/config.yaml
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/GR00T-VisualSim2Real
echo "final round-4 ckpt, 300 episodes, ORIGINAL config, raytraced obs"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=$D/last.pt \
    +num_envs=4 +headless=True \
    algo.config.eval.num_eval_episodes=300 \
    +simulator.config.render_results=true
echo EVAL_DONE
