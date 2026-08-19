#!/bin/bash
# One-time target-local Image3F environment, ONNX export, sm90 TRT build, and smoke gate.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --job-name=rr_i3f_sm90_build
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control}
BASE_PYTHON=${BASE_PYTHON:-/isaac-sim/python.sh}
PYTHON=${PYTHON:-$IMAGE3F_ROOT/.venv/bin/python}
export ROBORENDER_MODEL_BASE PYTHONDONTWRITEBYTECODE=1
export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STAGE_ROOT/runtime/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STAGE_ROOT/runtime/cache/pip}
export TMPDIR=${TMPDIR:-$STAGE_ROOT/runtime/tmp}
export TRT_WORKSPACE_GB=${TRT_WORKSPACE_GB:-64}
export PROBE_CUDA_GRAPH=${PROBE_CUDA_GRAPH:-0}
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

[ -d "$IMAGE3F_ROOT" ] || { echo "FATAL: Image3F bundle missing: $IMAGE3F_ROOT" >&2; exit 2; }
[ -d "$ROBORENDER_MODEL_BASE" ] || { echo "FATAL: foundation weights missing: $ROBORENDER_MODEL_BASE" >&2; exit 2; }
[ -x "$BASE_PYTHON" ] || { echo "FATAL: Isaac Python missing: $BASE_PYTHON" >&2; exit 2; }

cd "$IMAGE3F_ROOT"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$BASE_PYTHON" ./bin/create_baremetal_env.sh
fi
[ -x "$PYTHON" ] || { echo "FATAL: Image3F environment creation failed" >&2; exit 2; }

if [ -s assets/generated/engines/current/engine_build_manifest.json ]; then
    "$PYTHON" bin/doctor.py --stage runtime --model-root "$ROBORENDER_MODEL_BASE"
    PYTHON="$PYTHON" ./bin/smoke_test.sh
    echo "IMAGE3F_H100_ALREADY_READY"
    exit 0
fi

"$PYTHON" bin/doctor.py --stage package
"$PYTHON" bin/doctor.py --stage pre-export --model-root "$ROBORENDER_MODEL_BASE"
PYTHON="$PYTHON" ./bin/export_onnx.sh
PYTHON="$PYTHON" ./bin/build_engine.sh
"$PYTHON" bin/doctor.py --stage runtime --model-root "$ROBORENDER_MODEL_BASE"
PYTHON="$PYTHON" ./bin/calibrate_teacache.sh
PYTHON="$PYTHON" ./bin/smoke_test.sh
echo "IMAGE3F_H100_SETUP_DONE"
