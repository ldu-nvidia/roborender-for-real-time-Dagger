#!/bin/bash
#SBATCH --job-name=i3f_ft_eval
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/i3f_ft_eval_%j.log
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
FR=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$FR/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" "$ROOT/scripts/server/image3f_ft_eval.sbatch"
fi
export HOME=$HOME_OVERRIDE
export PYTHONDONTWRITEBYTECODE=1
PYTHON=$IMAGE3F_PYTHON
CHUNKS="${CHUNKS_OVERRIDE:- $FR/videogen_dataset/chunk_16222830_0000123.npz $FR/videogen_dataset/chunk_16222830_0000404.npz $FR/videogen_dataset/chunk_16222830_0000777.npz $FR/videogen_dataset/chunk_16222830_0001111.npz $FR/videogen_dataset/chunk_16222830_0001666.npz $FR/videogen_dataset/chunk_16222830_0001999.npz}"
"$PYTHON" "$FR/tools/image3f_eval2.py" --chunks $CHUNKS --out "$FR/image3f_eval_out" \
  --prev_modes ${PREV_MODES:-fixed} --tag_prefix ${TAG_PREFIX:-ftx_} \
  --model_root "$ROBORENDER_MODEL_BASE" ${EXTRA_ARGS:-} || { echo EVAL_FAIL; exit 1; }
GPY=$VIRAL_PYTHON
"$GPY" "$FR/tools/build_image3f_comparison.py" --gen_npz "$FR"/image3f_eval_out/chunk_*_gen_ft_*.npz --out "$FR/image3f_eval_out/cmp" || echo CMP_FAIL
echo ALL_DONE
