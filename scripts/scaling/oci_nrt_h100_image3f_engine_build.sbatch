#!/bin/bash
# Build a target-local SM90 TensorRT engine and execute latent plus RGB smoke.
#SBATCH --account=general_cs_infra
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=224G
#SBATCH --time=03:30:00
#SBATCH --job-name=rr_h100_image3f_engine
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}
ROOT=${ROOT:?submit with ROOT set to the immutable staged source snapshot}
CONTAINER=${CONTAINER:-$STAGE_ROOT/containers/roborender-image3f-x86_64-sm90-cu128-trt10.13.3.9.sqsh}
MODEL_ROOT=${MODEL_ROOT:-$STAGE_ROOT/assets/models}
GENERATED_ROOT=${GENERATED_ROOT:-$STAGE_ROOT/runtime/image3f-generated}
OUTPUT_ROOT=${OUTPUT_ROOT:-$STAGE_ROOT/validation/h100_image3f_runtime}
export STAGE_ROOT ROOT CONTAINER MODEL_ROOT GENERATED_ROOT OUTPUT_ROOT

[ -s "$CONTAINER" ] || { echo "FATAL: sealed renderer container missing: $CONTAINER" >&2; exit 2; }
[ -d "$MODEL_ROOT" ] || { echo "FATAL: foundation model root missing: $MODEL_ROOT" >&2; exit 2; }
mkdir -p "$GENERATED_ROOT/onnx" "$GENERATED_ROOT/engines" \
    "$GENERATED_ROOT/timing" "$OUTPUT_ROOT" "$STAGE_ROOT/runtime/home"

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --container-image="$CONTAINER" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$MODEL_ROOT:/models:ro,$GENERATED_ROOT:/opt/roborender/assets/generated,$OUTPUT_ROOT:/outputs" \
        --container-workdir=/opt/roborender \
        env ROBORENDER_CONTAINERIZED=1 STAGE_ROOT="$STAGE_ROOT" ROOT="$ROOT" \
            MODEL_ROOT=/models GENERATED_ROOT=/opt/roborender/assets/generated \
            OUTPUT_ROOT=/outputs HOME="$STAGE_ROOT/runtime/home" \
            ROBORENDER_MODEL_BASE=/models ROBORENDER_OUTPUTS=/outputs \
            TRT_WORKSPACE_GB=48 PROBE_CUDA_GRAPH=0 \
            bash "$ROOT/scripts/scaling/oci_nrt_h100_image3f_engine_build.sbatch"
fi

export PATH=/opt/roborender-h100-venv/bin:$PATH
export ROBORENDER_ROOT=/opt/roborender
export ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-/models}
export ROBORENDER_OUTPUTS=${ROBORENDER_OUTPUTS:-/outputs}
export DIFFSYNTH_MODEL_BASE_PATH=$ROBORENDER_MODEL_BASE
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export TRT_WORKSPACE_GB=${TRT_WORKSPACE_GB:-48}
export PROBE_CUDA_GRAPH=0
PY=/opt/roborender-h100-venv/bin/python

cd /opt/roborender
"$PY" bin/doctor.py --stage package --fast
"$PY" bin/doctor.py --stage pre-export --model-root "$ROBORENDER_MODEL_BASE" --fast
PYTHON="$PY" bin/export_onnx.sh
PYTHON="$PY" bin/build_engine.sh
"$PY" bin/doctor.py --stage runtime --model-root "$ROBORENDER_MODEL_BASE"
PYTHON="$PY" bin/smoke_test.sh
echo "OCI_NRT_H100_IMAGE3F_ENGINE_AND_SMOKE_PASS"
