#!/bin/bash
#SBATCH --partition=sc-loprio
#SBATCH --account=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -e
cd /svl/u/ravenh/fast_roborender/roborender-acceleration
export RAVEN_ROOT=$PWD RAVEN_ASSETS=$PWD/assets
export RAVEN_OUTPUTS=$PWD/outputs/$RAVEN_TAG
export PYTHONPATH=$PWD/vendor/sim2realgen
export TMPDIR=/svl/u/ravenh/tmp
source .venv/bin/activate
ln -sfn "generated/$RAVEN_ENGINE_SET" assets/engines/current
nvidia-smi -L
./bin/run_raw_sample.sh
echo RAW_SAMPLE_DONE
