#!/bin/bash
#SBATCH --partition=sc-loprio
#SBATCH --account=viscam
#SBATCH --job-name=raven_trt_build
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/build_engines_rtxpro6000.log
set -e
cd /svl/u/ravenh/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/svl/u/ravenh/tmp
source .venv/bin/activate
nvidia-smi
./bin/build_engines.sh
echo BUILD_ENGINES_DONE
