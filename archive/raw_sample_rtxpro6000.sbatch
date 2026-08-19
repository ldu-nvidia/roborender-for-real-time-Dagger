#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=raven_raw
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/raw_sample_rtxpro6000.log
set -e
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
source .venv/bin/activate
./bin/run_raw_sample.sh
echo RAW_SAMPLE_DONE
