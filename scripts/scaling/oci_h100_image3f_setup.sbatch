#!/bin/bash
# One-time target-local Image3F environment, ONNX export, sm90 TRT build, and smoke gate.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --job-name=rr_i3f_sm90_build
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-$STAGE_ROOT}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" \
        "$ROOT/scripts/scaling/oci_h100_image3f_setup.sbatch"
fi
IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
IMAGE3F_ENV=${IMAGE3F_ENV:-$STAGE_ROOT/runtime/envs/image3f-cu128}
ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models}
BASE_PYTHON=${BASE_PYTHON:-/isaac-sim/python.sh}
PYTHON=${PYTHON:-$IMAGE3F_ENV/bin/python}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
TORCH_VERSION=${TORCH_VERSION:-2.7.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.7.0}
TENSORRT_VERSION=${TENSORRT_VERSION:-10.13.3.9}
ENV_READY_MARKER=$IMAGE3F_ENV/.oci_environment_ready
export ROBORENDER_MODEL_BASE PYTHONDONTWRITEBYTECODE=1
export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STAGE_ROOT/runtime/cache/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STAGE_ROOT/runtime/cache/pip}
export TMPDIR=${TMPDIR:-$STAGE_ROOT/runtime/tmp}
export TRT_WORKSPACE_GB=${TRT_WORKSPACE_GB:-64}
export PROBE_CUDA_GRAPH=${PROBE_CUDA_GRAPH:-0}
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$IMAGE3F_ENV")"

[ -d "$IMAGE3F_ROOT" ] || { echo "FATAL: Image3F bundle missing: $IMAGE3F_ROOT" >&2; exit 2; }
[ -d "$ROBORENDER_MODEL_BASE" ] || { echo "FATAL: foundation weights missing: $ROBORENDER_MODEL_BASE" >&2; exit 2; }
[ -x "$BASE_PYTHON" ] || { echo "FATAL: Isaac Python missing: $BASE_PYTHON" >&2; exit 2; }

cd "$IMAGE3F_ROOT"
if [ ! -s "$ENV_READY_MARKER" ]; then
    if [ ! -x "$PYTHON" ]; then
        "$BASE_PYTHON" -m venv "$IMAGE3F_ENV"
    fi
    "$PYTHON" -m pip install --upgrade pip setuptools wheel
    "$PYTHON" -m pip install --index-url "$TORCH_INDEX_URL" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION"
    "$PYTHON" -m pip install -r container/requirements.runtime.txt
    "$PYTHON" -m pip install --no-deps \
        "tensorrt-cu12-libs==$TENSORRT_VERSION" \
        "tensorrt-cu12-bindings==$TENSORRT_VERSION" \
        "tensorrt-cu12==$TENSORRT_VERSION"
    "$PYTHON" -m pip install --no-deps -e vendor/raven-image3f-joint
    "$PYTHON" - <<'PY'
import tensorrt
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the OCI Image3F environment")
major, minor = torch.cuda.get_device_capability()
if (major, minor) != (9, 0):
    raise SystemExit(f"expected an H100 (sm90), found sm{major}{minor}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} tensorrt={tensorrt.__version__}")
PY
    {
        echo "torch=$TORCH_VERSION"
        echo "torch_index=$TORCH_INDEX_URL"
        echo "tensorrt=$TENSORRT_VERSION"
        echo "gpu_arch=sm90"
    } > "$ENV_READY_MARKER"
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
