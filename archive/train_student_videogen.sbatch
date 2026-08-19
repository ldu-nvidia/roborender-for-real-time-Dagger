#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=viral_videogen
#SBATCH --gres=gpu:rtxpro6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/train_student_videogen_%j.log
# GPU 0: Isaac Sim + student training (viral conda env)
# GPU 1: Raven video-gen render server (roborender venv)
set -e
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender

SERVER_LOG=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/raven_server_${SLURM_JOB_ID}.log
RAVEN_PORT=$((18000 + SLURM_JOB_ID % 1000))
echo "raven port $RAVEN_PORT"
CUDA_VISIBLE_DEVICES=1 /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/run_raven_server.sh --port $RAVEN_PORT > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "raven server pid $SERVER_PID, log $SERVER_LOG"
until grep -q RAVEN_SERVER_READY "$SERVER_LOG" 2>/dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "raven server died during startup:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  sleep 5
done
echo "raven server ready"

# Keep Isaac Sim / pip caches off the small /sailhome quota.
export XDG_CACHE_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/xdg
export XDG_DATA_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/share
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
export OMNI_KIT_ACCEPT_EULA=YES

source /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/conda/etc/profile.d/conda.sh
conda activate viral

cd GR00T-VisualSim2Real
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_student_videogen \
    num_envs=${NUM_ENVS:-4} \
    headless=True \
    experiment_name=wsdpt_student_videogen \
    project_name=wsdpt_student_videogen \
    algo.config.videogen.port=$RAVEN_PORT

kill $SERVER_PID 2>/dev/null || true
echo TRAIN_DONE
