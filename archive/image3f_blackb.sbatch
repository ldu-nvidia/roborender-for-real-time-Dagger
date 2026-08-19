#!/bin/bash
#SBATCH --job-name=image3f_blackb
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/image3f_blackb_%j.log
set -x
export HOME=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/home
ROOT=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/assets/roborender_image_model_accelerated
PYTHON="$ROOT/.venv/bin/python"
FR=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender
CHUNKS="$FR/videogen_dataset/chunk_16222830_0001309.npz $FR/videogen_dataset/chunk_16222830_0000051.npz $FR/videogen_dataset/chunk_16222830_0000501.npz"
export IMAGE3F_ROOT="$ROOT"
"$PYTHON" "$FR/image3f_eval.py" --chunks $CHUNKS --out "$FR/image3f_eval_out" --depth_variant bctrl --prev_mode black --tag bctrlblack || echo EVAL_FAIL
GPY=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/runtime/envs/viral/bin/python
"$GPY" "$FR/build_image3f_comparison.py" --gen_npz "$FR"/image3f_eval_out/chunk_*_gen_bctrlblack.npz --out "$FR/image3f_eval_out/cmp" || echo CMP_FAIL
echo ALL_DONE
