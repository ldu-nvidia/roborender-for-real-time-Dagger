#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=raven_unzip
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/unzip_verify.log
set -e
cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender
unzip -o -q raven_ladd_5step_trt_build.zip
cd raven_ladd_5step_trt_build_inputs_v2
sha256sum --check BUILD_INPUT_MANIFEST.sha256
echo ALL_VERIFIED
