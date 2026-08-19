#!/bin/bash
#SBATCH --job-name=i3f_ft_setup_hi
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/i3f_ft_setup_hi_%j.log
set -x
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/pip
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
ROOT=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets/roborender_image_model_accelerated
export ROBORENDER_MODEL_BASE=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets/models
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  sed -i 's|--index-url "${TORCH_INDEX_URL}" \\|--index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \\|' bin/create_baremetal_env.sh
  PYTHON=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets/roborender_image_model_accelerated/.venv/bin/python ./bin/create_baremetal_env.sh; RC=$?
  cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets && unzip -o -q roborender_image_fine_tune.zip roborender_image_model_accelerated/bin/create_baremetal_env.sh && cd "$ROOT"
  [ $RC -ne 0 ] && { echo ENV_FAIL; exit 1; }
fi
export PYTHON="$ROOT/.venv/bin/python"
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
