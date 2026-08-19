#!/bin/bash
#SBATCH --job-name=i3f_ft_setup
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/i3f_ft_setup_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
FR_ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$FR_ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR=$ROBORENDER_STAGE_ROOT/runtime/cache/pip
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
cd "$IMAGE3F_ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

[ -x "$IMAGE3F_PYTHON" ] || {
  echo "FATAL: target-local environment missing; run scripts/scaling/oci_h100_image3f_setup.sbatch" >&2
  exit 2
}
export PYTHON=$IMAGE3F_PYTHON
echo ENV_OK

"$PYTHON" bin/doctor.py --stage package || { echo PKG_DOCTOR_FAIL; exit 1; }
echo PKG_DOCTOR_OK
"$PYTHON" bin/doctor.py --stage pre-export --model-root "$ROBORENDER_MODEL_BASE" || { echo PREEXPORT_FAIL; exit 1; }
echo PREEXPORT_OK
PYTHON="$PYTHON" ./bin/export_onnx.sh || { echo EXPORT_FAIL; exit 1; }
echo EXPORT_OK
PYTHON="$PYTHON" ./bin/build_engine.sh || { echo BUILD_FAIL; exit 1; }
echo BUILD_OK
"$PYTHON" bin/doctor.py --stage runtime --model-root "$ROBORENDER_MODEL_BASE" || { echo RT_DOCTOR_FAIL; exit 1; }
echo RT_DOCTOR_OK
PYTHON="$PYTHON" ./bin/calibrate_teacache.sh || echo CALIBRATE_WARN
PYTHON="$PYTHON" ./bin/smoke_test.sh || { echo SMOKE_FAIL; exit 1; }
echo SMOKE_OK
echo SETUP_ALL_DONE
