#!/bin/bash
# Build and save a sealed x86_64/CUDA 12.8 Image3F renderer for H100 SM90.
#SBATCH --account=general_cs_infra
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=224G
#SBATCH --time=02:30:00
#SBATCH --job-name=rr_h100_renderer_build
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}
ROOT=${ROOT:?submit with ROOT set to the immutable staged source snapshot}
export ROBORENDER_STAGE_ROOT=$STAGE_ROOT ROOT
source "$ROOT/scripts/setup/oci_nrt_h100_env.sh"

PACKAGE_ROOT=${PACKAGE_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
CONTAINER_ROOT=${CONTAINER_ROOT:-$STAGE_ROOT/containers}
CONTAINER_OUT=${CONTAINER_OUT:-$CONTAINER_ROOT/roborender-image3f-x86_64-sm90-cu128-trt10.13.3.9.sqsh}
PARTIAL="$CONTAINER_OUT.partial.${SLURM_JOB_ID:-$$}"
REPORT=${REPORT:-$STAGE_ROOT/validation/h100_renderer_build_imports.json}
export PACKAGE_ROOT REPORT

mkdir -p "$CONTAINER_ROOT" "$STAGE_ROOT/validation" "$HOME_OVERRIDE" \
    "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
[ -d "$PACKAGE_ROOT" ] || { echo "FATAL: portable Image3F package missing: $PACKAGE_ROOT" >&2; exit 2; }
[ ! -e "$CONTAINER_OUT" ] || { echo "FATAL: refusing to overwrite $CONTAINER_OUT" >&2; exit 2; }
[ ! -e "$PARTIAL" ] || { echo "FATAL: stale partial output exists: $PARTIAL" >&2; exit 2; }

if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    set +e
    srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --container-image="$OCI_H100_BASE_IMAGE" \
        --container-writable --container-remap-root \
        --container-save="$PARTIAL" \
        --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
        --container-workdir="$ROOT" \
        env ROBORENDER_CONTAINERIZED=1 ROBORENDER_STAGE_ROOT="$STAGE_ROOT" \
            ROOT="$ROOT" PACKAGE_ROOT="$PACKAGE_ROOT" REPORT="$REPORT" \
            HOME="$HOME_OVERRIDE" HOME_OVERRIDE="$HOME_OVERRIDE" \
            XDG_CACHE_HOME="$XDG_CACHE_HOME" PIP_CACHE_DIR="$PIP_CACHE_DIR" \
            TMPDIR="$TMPDIR" bash "$ROOT/scripts/scaling/oci_nrt_h100_renderer_build.sbatch"
    BUILD_RC=$?
    set -e
    [ "$BUILD_RC" -eq 0 ] || { echo "FATAL: renderer image build failed: $BUILD_RC" >&2; exit "$BUILD_RC"; }
    [ -s "$PARTIAL" ] || { echo "FATAL: Pyxis did not save $PARTIAL" >&2; exit 2; }
    mv "$PARTIAL" "$CONTAINER_OUT"
    sha256sum "$CONTAINER_OUT" | tee "$CONTAINER_OUT.sha256"
    echo "OCI_NRT_H100_RENDERER_CONTAINER_READY=$CONTAINER_OUT"
    exit 0
fi

export HOME=${HOME:-$HOME_OVERRIDE}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

/isaac-sim/python.sh -m venv /opt/roborender-h100-venv
PY=/opt/roborender-h100-venv/bin/python
PIP=("$PY" -m pip)
"${PIP[@]}" install --upgrade pip==26.1.2 setuptools wheel
"${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
"${PIP[@]}" install -r "$PACKAGE_ROOT/container/requirements.runtime.txt"
"${PIP[@]}" install --no-deps \
    tensorrt-cu12-libs==10.13.3.9 \
    tensorrt-cu12-bindings==10.13.3.9 \
    tensorrt-cu12==10.13.3.9

cp -a "$PACKAGE_ROOT" /opt/roborender
"${PIP[@]}" install --no-deps -e /opt/roborender/vendor/raven-image3f-joint
install -m 0555 "$ROOT/containers/h100/renderer_exec.sh" /usr/local/bin/roborender-sm90
install -m 0444 "$ROOT/containers/gb200/renderer_import_gate.py" /opt/renderer_import_gate.py

cd /opt/roborender
"$PY" bin/doctor.py --stage package --fast
ROBORENDER_EXPECTED_MACHINE=x86_64 \
ROBORENDER_EXPECTED_CC=9,0 \
ROBORENDER_EXPECTED_TORCH_CUDA=12.8 \
ROBORENDER_EXPECTED_TENSORRT=10.13.3.9 \
    "$PY" /opt/renderer_import_gate.py | tee "$REPORT"
grep -q ROBORENDER_RENDERER_IMPORT_GATE_PASS "$REPORT"
"${PIP[@]}" check
echo "OCI_NRT_H100_RENDERER_BUILD_CONTENT_PASS"
