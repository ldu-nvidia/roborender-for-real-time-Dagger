#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=teacher_anchor
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/teacher_anchor_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
cd "$ROOT"
FR=$ROOT
PORT=$((18000 + SLURM_JOB_ID % 1000))
: "${TEACHER_LORA:?TEACHER_LORA must point to the staged teacher LoRA}"
"$ROOT/run_raven_server.sh" --host 127.0.0.1 --port "$PORT" --teacher_lora "$TEACHER_LORA" &
SERVER_PID=$!
CHUNKS="${CHUNKS_OVERRIDE:- $ROOT/videogen_dataset/chunk_16222830_0000123.npz $ROOT/videogen_dataset/chunk_16222830_0000404.npz $ROOT/videogen_dataset/chunk_16222830_0000777.npz $ROOT/videogen_dataset/chunk_16222830_0001111.npz $ROOT/videogen_dataset/chunk_16222830_0001666.npz $ROOT/videogen_dataset/chunk_16222830_0001999.npz}"
"$VIRAL_PYTHON" "$ROOT/scripts/anchors/render_teacher_fullres.py" \
  --chunks $CHUNKS --out "$FR/teacher_fullres" --port $PORT
RC=$?
kill $SERVER_PID 2>/dev/null
[ $RC -ne 0 ] && { echo RENDER_FAIL; exit 1; }
echo ANCHOR_JOB_DONE
