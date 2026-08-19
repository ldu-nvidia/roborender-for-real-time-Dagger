#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -e
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets
export RAVEN_OUTPUTS=$PWD/outputs/$RAVEN_TAG
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
source .venv/bin/activate
ln -sfn "generated/$RAVEN_ENGINE_SET" assets/engines/current
nvidia-smi -L
./bin/smoke_test.sh
echo SMOKE_DONE
