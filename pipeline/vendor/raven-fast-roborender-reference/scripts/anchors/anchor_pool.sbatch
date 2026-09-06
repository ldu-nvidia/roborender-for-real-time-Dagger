#!/bin/bash
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=anchor_pool
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/anchor_pool_%j.log
set -x
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
PORT=$((18000 + SLURM_JOB_ID % 1000))
/svl/u/ravenh/fast_roborender/run_raven_server.sh --host 127.0.0.1 --port $PORT --teacher_lora /svl/u/ravenh/renderscale/Sim2RealGen_Video_Model/lora_ckpts/Wan2.1-Fun-V1.1-1.3B-Mixed-DROID-AGIBOT-3View-DualControl-PrevFrame-droidnew_continue_lora/step-5294.safetensors &
SERVER_PID=$!
/svl/u/ravenh/miniconda3/envs/gr00t/bin/python /svl/u/ravenh/fast_roborender/scripts/anchors/render_anchor_pool.py \
  --start ${START:-0} --end ${END:-8} --out ${OUT:-/vision/u/ravenh/anchor_pool_2048} --port $PORT
RC=$?
kill $SERVER_PID 2>/dev/null
[ $RC -ne 0 ] && { echo POOL_FAIL; exit 1; }
echo POOL_SHARD_DONE
