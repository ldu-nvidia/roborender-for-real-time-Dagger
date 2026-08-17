#!/bin/bash
#SBATCH --partition=sc-loprio
#SBATCH --account=viscam
#SBATCH --job-name=raven_smoke
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/smoke_rtxpro6000.log
set -e
cd /svl/u/ravenh/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/svl/u/ravenh/tmp
source .venv/bin/activate
./bin/smoke_test.sh
echo SMOKE_DONE
