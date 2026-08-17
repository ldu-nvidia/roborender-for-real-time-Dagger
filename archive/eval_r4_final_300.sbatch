#!/bin/bash
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=eval_r4_final300
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/eval_r4_final300_%j.log
set -e
export HOME=/sailhome/ravenh
export XDG_CACHE_HOME=/svl/u/ravenh/.cache_xdg XDG_DATA_HOME=/svl/u/ravenh/.share_xdg
export TMPDIR=/svl/u/ravenh/tmp OMNI_KIT_ACCEPT_EULA=YES
source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
# stage: final round-4 weights + ORIGINAL offline-BC eval config
R=/svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real/logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202
D=/svl/u/ravenh/fast_roborender/diag_r4_final
mkdir -p $D
# checkpoint already staged and md5-verified; re-copy only if size differs
SRC_SZ=$(stat -c %s $R/last.pt); DST_SZ=$(stat -c %s $D/last.pt 2>/dev/null || echo 0)
[ $SRC_SZ = $DST_SZ ] || cp $R/last.pt $D/last.pt
cp /svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real/logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447/config.yaml $D/config.yaml
cd /svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real
echo "final round-4 ckpt, 300 episodes, ORIGINAL config, raytraced obs"
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=$D/last.pt \
    +num_envs=4 +headless=True \
    algo.config.eval.num_eval_episodes=300 \
    +simulator.config.render_results=true
echo EVAL_DONE
