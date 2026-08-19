#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -e
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
source .venv/bin/activate
nvidia-smi -L
./bin/build_engines.sh
echo BUILD_ENGINES_DONE
