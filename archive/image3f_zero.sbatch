#!/bin/bash
#SBATCH --job-name=image3f_zero
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --nodelist=viscam14,viscam15
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/image3f_zero_%j.log
set -x
export HOME=/sailhome/ravenh
ROOT=/vision/u/ravenh/roborender_image3f_portable_buildonly_v1
PYTHON="$ROOT/.venv/bin/python"
FR=/svl/u/ravenh/fast_roborender
CHUNKS="$FR/videogen_dataset/chunk_16222830_0001309.npz $FR/videogen_dataset/chunk_16222830_0000051.npz $FR/videogen_dataset/chunk_16222830_0000501.npz"
export IMAGE3F_ROOT="$ROOT"
"$PYTHON" "$FR/image3f_eval.py" --chunks $CHUNKS --out "$FR/image3f_eval_out" --depth_variant droid --prev_mode zero --tag droidzero || echo EVAL_FAIL
GPY=/svl/u/ravenh/miniconda3/envs/gr00t/bin/python
"$GPY" "$FR/build_image3f_comparison.py" --gen_npz "$FR"/image3f_eval_out/chunk_*_gen_droidzero.npz --out "$FR/image3f_eval_out/cmp" || echo CMP_FAIL
echo ALL_DONE
