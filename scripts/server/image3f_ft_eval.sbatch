#!/bin/bash
#SBATCH --job-name=i3f_ft_eval
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/i3f_ft_eval_%j.log
set -x
export HOME=/sailhome/ravenh
export PYTHONDONTWRITEBYTECODE=1
ROOT=/vision/u/ravenh/roborender_image_model_accelerated
PYTHON="$ROOT/.venv/bin/python"
FR=/svl/u/ravenh/fast_roborender
CHUNKS="${CHUNKS_OVERRIDE:- /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0000123.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0000404.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0000777.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0001111.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0001666.npz /svl/u/ravenh/fast_roborender/videogen_dataset/chunk_16222830_0001999.npz}"
export IMAGE3F_ROOT="$ROOT"
export ROBORENDER_MODEL_BASE=/vision/u/ravenh/roborender_image3f_portable_buildonly_v1/assets/models
"$PYTHON" "$FR/image3f_eval2.py" --chunks $CHUNKS --out "$FR/image3f_eval_out" \
  --prev_modes ${PREV_MODES:-fixed} --tag_prefix ${TAG_PREFIX:-ftx_} \
  --model_root "$ROBORENDER_MODEL_BASE" ${EXTRA_ARGS:-} || { echo EVAL_FAIL; exit 1; }
GPY=/svl/u/ravenh/miniconda3/envs/gr00t/bin/python
"$GPY" "$FR/build_image3f_comparison.py" --gen_npz "$FR"/image3f_eval_out/chunk_*_gen_ft_*.npz --out "$FR/image3f_eval_out/cmp" || echo CMP_FAIL
echo ALL_DONE
