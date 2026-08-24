#!/bin/bash
# Re-open and validate the immutable H100 renderer SquashFS independently.
#SBATCH --account=general_cs_infra
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=rr_h100_renderer_validate
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}
ROOT=${ROOT:?submit with ROOT set to the immutable staged source snapshot}
CONTAINER=${CONTAINER:-$STAGE_ROOT/containers/roborender-image3f-x86_64-sm90-cu128-trt10.13.3.9.sqsh}
REPORT=${REPORT:-$STAGE_ROOT/validation/h100_renderer_sealed_imports.json}
[ -s "$CONTAINER" ] || { echo "FATAL: renderer container missing: $CONTAINER" >&2; exit 2; }

srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
    --container-workdir=/opt/roborender \
    env HOME="$STAGE_ROOT/runtime/home" \
        /usr/local/bin/roborender-sm90 python /opt/renderer_import_gate.py | tee "$REPORT"
grep -q ROBORENDER_RENDERER_IMPORT_GATE_PASS "$REPORT"
echo "OCI_NRT_H100_RENDERER_SEALED_PASS"
