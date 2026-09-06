#!/bin/bash
#SBATCH --partition=sc-freecpu
#SBATCH --job-name=raven_unzip
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/unzip_verify.log
set -e
cd /svl/u/ravenh/fast_roborender
unzip -o -q raven_ladd_5step_trt_build.zip
cd raven_ladd_5step_trt_build_inputs_v2
sha256sum --check BUILD_INPUT_MANIFEST.sha256
echo ALL_VERIFIED
