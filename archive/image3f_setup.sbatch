#!/bin/bash
#SBATCH --job-name=image3f_setup
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/image3f_setup_%j.log

set -x
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
export PIP_CACHE_DIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/cache/pip
export TMPDIR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

ROOT=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets/roborender_image_model_accelerated

# 0. extract bundle if not already complete (login-node unzip gets reaped)
if [ ! -f "$ROOT/.extract_ok" ]; then
  cd /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets
  rm -rf roborender_image3f_portable_buildonly_v1
  unzip -o -q roborender_image_model_accelerated.zip || { echo UNZIP_FAIL; exit 1; }
  cd "$ROOT"
  test -s assets/checkpoint/step-228270.safetensors || { echo MISSING_CKPT; exit 1; }
  test -s assets/onnx/image3f_dit_block_stack_reference_bf16.onnx || { echo MISSING_ONNX; exit 1; }
  test -s assets/onnx/image3f_dit_block_stack_reference_bf16.onnx.data || { echo MISSING_ONNX_DATA; exit 1; }
  test ! -e assets/engines/current || { echo UNEXPECTED_ENGINE; exit 1; }
  touch "$ROOT/.extract_ok"
  ln -sfn "$ROOT" /lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/image3f
  echo EXTRACT_OK
fi
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# 1. env (skip if already built)
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  PYTHON=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets/roborender_image_model_accelerated/.venv/bin/python ./bin/create_baremetal_env.sh || { echo ENV_FAIL; exit 1; }
fi
export PYTHON="$ROOT/.venv/bin/python"
echo ENV_OK

# 2. verify assets
"$PYTHON" bin/doctor.py --verify-all-hashes --allow-missing-engine || { echo DOCTOR1_FAIL; exit 1; }
echo DOCTOR1_OK

# 3. build target-native engine
if [ ! -e "$ROOT/assets/engines/current" ]; then
  PYTHON="$PYTHON" ./bin/build_engine.sh || { echo BUILD_FAIL; exit 1; }
fi
echo BUILD_OK
"$PYTHON" bin/doctor.py --verify-all-hashes || { echo DOCTOR2_FAIL; exit 1; }
echo DOCTOR2_OK

# 4. smoke
PYTHON="$PYTHON" ./bin/smoke_test.sh || { echo SMOKE_FAIL; exit 1; }
echo SMOKE_OK

# 5. eval on saved GR00T chunks (both depth variants)
FR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender
CHUNKS="$FR/videogen_dataset/chunk_16222830_0001309.npz $FR/videogen_dataset/chunk_16222830_0000051.npz $FR/videogen_dataset/chunk_16222830_0000501.npz"
export IMAGE3F_ROOT="$ROOT"
"$PYTHON" -m pip install --quiet pillow numpy 2>/dev/null
"$PYTHON" "$FR/image3f_eval.py" --chunks $CHUNKS --out "$FR/image3f_eval_out" --depth_variant droid || echo EVAL_DROID_FAIL
"$PYTHON" "$FR/image3f_eval.py" --chunks $CHUNKS --out "$FR/image3f_eval_out" --depth_variant bctrl || echo EVAL_BCTRL_FAIL

# 6. comparison videos + metrics (cv2 from gr00t env, CPU work)
GPY=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/viral/bin/python
"$GPY" "$FR/build_image3f_comparison.py" \
  --gen_npz "$FR"/image3f_eval_out/chunk_*_gen_*.npz \
  --out "$FR/image3f_eval_out/cmp" || echo CMP_FAIL

echo ALL_DONE
