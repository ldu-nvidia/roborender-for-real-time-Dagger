#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=viral_videogen
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/train_student_videogen_remote_%j.log
# Training only; connects to a raven_server.sbatch job via raven_server_endpoint.txt
set -e
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender
until [ -s raven_server_endpoint.txt ]; do echo "waiting for server endpoint"; sleep 10; done
read RAVEN_HOST RAVEN_PORT < raven_server_endpoint.txt
echo "using raven server at $RAVEN_HOST:$RAVEN_PORT"

export XDG_CACHE_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/xdg
export XDG_DATA_HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/share
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
export OMNI_KIT_ACCEPT_EULA=YES

source /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/conda/etc/profile.d/conda.sh
conda activate viral

cd GR00T-VisualSim2Real
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_student_videogen \
    num_envs=${NUM_ENVS:-4} \
    headless=True \
    experiment_name=wsdpt_student_videogen \
    project_name=wsdpt_student_videogen \
    algo.config.videogen.host=$RAVEN_HOST \
    algo.config.videogen.port=$RAVEN_PORT
echo TRAIN_DONE
