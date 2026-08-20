#!/bin/bash
# Build and save the immutable ARM64/CUDA13 Image3F renderer environment.
# Model foundations and target-local ONNX/TRT artifacts remain external.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=230G
#SBATCH --time=02:00:00
#SBATCH --job-name=rr_gb200_renderer_build
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
export ROBORENDER_STAGE_ROOT=$STAGE_ROOT ROOT
source "$ROOT/scripts/setup/hsg_gb200_env.sh"

PACKAGE_ROOT=${PACKAGE_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
CONTAINER_ROOT=${CONTAINER_ROOT:-$STAGE_ROOT/containers}
CONTAINER_OUT=${CONTAINER_OUT:-$CONTAINER_ROOT/roborender-image3f-arm64-sm100-cu130-trt10.13.3.9.sqsh}
PARTIAL="$CONTAINER_OUT.partial.${SLURM_JOB_ID:-$$}"
REPORT=${REPORT:-$STAGE_ROOT/validation/renderer_build_imports.json}
export PACKAGE_ROOT REPORT

mkdir -p "$CONTAINER_ROOT" "$STAGE_ROOT/validation" "$HOME_OVERRIDE" \
    "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
[ -d "$PACKAGE_ROOT" ] || { echo "FATAL: portable Image3F package missing: $PACKAGE_ROOT" >&2; exit 2; }
[ ! -e "$CONTAINER_OUT" ] || { echo "FATAL: refusing to overwrite $CONTAINER_OUT" >&2; exit 2; }
[ ! -e "$PARTIAL" ] || { echo "FATAL: stale partial output exists: $PARTIAL" >&2; exit 2; }

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    set +e
    srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --container-image="$HSG_RENDERER_CONTAINER_IMAGE" \
        --container-writable --container-remap-root \
        --container-save="$PARTIAL" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir="$ROOT" \
        env ROBORENDER_CONTAINERIZED=1 ROBORENDER_STAGE_ROOT="$STAGE_ROOT" \
            ROOT="$ROOT" PACKAGE_ROOT="$PACKAGE_ROOT" REPORT="$REPORT" \
            HOME="$HOME_OVERRIDE" HOME_OVERRIDE="$HOME_OVERRIDE" \
            XDG_CACHE_HOME="$XDG_CACHE_HOME" PIP_CACHE_DIR="$PIP_CACHE_DIR" \
            TMPDIR="$TMPDIR" bash "$ROOT/scripts/scaling/hsg_gb200_renderer_build.sbatch"
    BUILD_RC=$?
    set -e
    [ "$BUILD_RC" -eq 0 ] || { echo "FATAL: renderer image build failed: $BUILD_RC" >&2; exit "$BUILD_RC"; }
    [ -s "$PARTIAL" ] || { echo "FATAL: Pyxis did not save $PARTIAL" >&2; exit 2; }
    mv "$PARTIAL" "$CONTAINER_OUT"
    sha256sum "$CONTAINER_OUT" | tee "$CONTAINER_OUT.sha256"
    echo "HSG_GB200_RENDERER_CONTAINER_READY=$CONTAINER_OUT"
    exit 0
fi

export HOME=${HOME:-$HOME_OVERRIDE}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

python -m venv /opt/roborender-venv
PY=/opt/roborender-venv/bin/python
PIP=("$PY" -m pip)
"${PIP[@]}" install --upgrade pip==26.1.2 setuptools wheel
"${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130

# onnxruntime-gpu does not publish an ARM64 wheel.  The package does not use
# ORT for inference or engine build, but CPU ORT is retained for tooling/import
# parity; TensorRT remains the GPU execution backend.
mapfile -t RUNTIME_REQUIREMENTS < <(
    grep -Ev '^[[:space:]]*(#|$)|^onnxruntime-gpu==' \
        "$PACKAGE_ROOT/container/requirements.runtime.txt"
)
"${PIP[@]}" install "${RUNTIME_REQUIREMENTS[@]}" onnxruntime==1.24.4
"${PIP[@]}" install --no-deps \
    tensorrt-cu13-libs==10.13.3.9 \
    tensorrt-cu13-bindings==10.13.3.9 \
    tensorrt-cu13==10.13.3.9

cp -a "$PACKAGE_ROOT" /opt/roborender
"${PIP[@]}" install --no-deps -e /opt/roborender/vendor/raven-image3f-joint
install -m 0555 "$ROOT/containers/gb200/renderer_exec.sh" /usr/local/bin/roborender-sm100
install -m 0444 "$ROOT/containers/gb200/renderer_import_gate.py" /opt/renderer_import_gate.py

cd /opt/roborender
"$PY" bin/doctor.py --stage package --fast
"$PY" /opt/renderer_import_gate.py | tee "$REPORT"
set +e
PIP_CHECK_OUTPUT=$("${PIP[@]}" check 2>&1)
PIP_CHECK_RC=$?
set -e
printf '%s\n' "$PIP_CHECK_OUTPUT"
if [ "$PIP_CHECK_RC" -ne 0 ]; then
    # ARM64 PyTorch supplies the CUDA runtime as `nvidia-cuda-runtime`, while
    # TensorRT's metapackage asks pip for the equivalent suffixed distribution.
    # The official cu130 torch wheel also carries an ARM64 cuSPARSELt wheel that
    # pip 26 misclassifies despite successful load and SM100 kernel execution.
    # Both are proven dynamically by renderer_import_gate.py; reject every
    # other dependency error.
    UNEXPECTED=$(printf '%s\n' "$PIP_CHECK_OUTPUT" | grep -Fvx \
        -e 'tensorrt-cu13-libs 10.13.3.9 requires nvidia-cuda-runtime-cu13, which is not installed.' \
        -e 'nvidia-cusparselt-cu13 0.8.0 is not supported on this platform.' || true)
    [ -z "$UNEXPECTED" ] || { echo "FATAL: unexpected pip check errors:" >&2; printf '%s\n' "$UNEXPECTED" >&2; exit 2; }
    echo "Known ARM64 package-metadata discrepancies accepted after dynamic CUDA/TRT gate"
fi
echo "HSG_GB200_RENDERER_BUILD_CONTENT_PASS"
