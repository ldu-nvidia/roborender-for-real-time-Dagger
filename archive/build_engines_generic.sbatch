#!/bin/bash
#SBATCH --partition=sc-loprio
#SBATCH --account=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -e
cd /svl/u/ravenh/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets RAVEN_OUTPUTS=$PWD/outputs
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/svl/u/ravenh/tmp
source .venv/bin/activate
nvidia-smi -L
./bin/build_engines.sh
echo BUILD_ENGINES_DONE
