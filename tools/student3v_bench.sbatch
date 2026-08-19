#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=student3v_bench
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/student3v_bench_%j.log
#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"
#SBATCH --container-mounts=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale:/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
export HOME=$HOME_OVERRIDE
cd "$ROOT"
PORT=$((18000 + SLURM_JOB_ID % 1000))
"$ROOT/run_raven_server.sh" --host 127.0.0.1 --port "$PORT" \
  --student_pth "$ROBORENDER_ACCEL_ROOT/assets/student/raven_native3view_ladd_5step_iter5250.pth" &
SERVER_PID=$!
FR=$ROOT
CHUNKS="$FR/videogen_dataset/chunk_16222830_0001309.npz $FR/videogen_dataset/chunk_16222830_0000051.npz $FR/videogen_dataset/chunk_16222830_0000501.npz"
"$VIRAL_PYTHON" "$ROOT/scripts/anchors/render_teacher_fullres.py" \
  --chunks $CHUNKS --out "$FR/student3v_fullres" --port $PORT
RC=$?
kill $SERVER_PID 2>/dev/null
[ $RC -ne 0 ] && { echo BENCH_FAIL; exit 1; }
echo BENCH_DONE
