#!/bin/bash
# Re-open the saved SM100 renderer image and validate imports from the sealed
# SquashFS, independently of the writable build step.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=rr_gb200_renderer_validate
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
CONTAINER=${CONTAINER:-$STAGE_ROOT/containers/roborender-image3f-arm64-sm100-cu130-trt10.13.3.9.sqsh}
REPORT=${REPORT:-$STAGE_ROOT/validation/renderer_sealed_imports.json}
[ -s "$CONTAINER" ] || { echo "FATAL: renderer container missing: $CONTAINER" >&2; exit 2; }

srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
    --container-workdir=/opt/roborender \
    env HOME="$STAGE_ROOT/runtime/home" \
        /opt/roborender-venv/bin/python /opt/renderer_import_gate.py | tee "$REPORT"
grep -q HSG_GB200_RENDERER_IMPORT_GATE_PASS "$REPORT"
echo "HSG_GB200_RENDERER_SEALED_PASS"
