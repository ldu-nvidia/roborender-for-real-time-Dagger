#!/bin/bash
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=anchor_pool
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/anchor_pool_%j.log
set -Eeuo pipefail
ROBORENDER_STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$ROBORENDER_STAGE_ROOT/src/fast_roborender}
source "$ROOT/scripts/setup/oci_env.sh"
if [ "${ROBORENDER_CONTAINERIZED:-0}" != 1 ]; then
    exec bash "$ROOT/scripts/setup/reexec_in_oci_container.sh" "$ROOT/scripts/anchors/anchor_pool.sbatch"
fi
export HOME=$HOME_OVERRIDE
cd "$ROOT"
PORT=$((18000 + SLURM_JOB_ID % 1000))
: "${TEACHER_LORA:?TEACHER_LORA must point to the staged teacher LoRA}"
"$ROOT/run_raven_server.sh" --host 127.0.0.1 --port "$PORT" --teacher_lora "$TEACHER_LORA" &
SERVER_PID=$!
"$VIRAL_PYTHON" "$ROOT/scripts/anchors/render_anchor_pool.py" \
  --start "${START:-0}" --end "${END:-8}" --out "${OUT:-$ANCHORS}" --port "$PORT"
RC=$?
kill $SERVER_PID 2>/dev/null
[ $RC -ne 0 ] && { echo POOL_FAIL; exit 1; }
echo POOL_SHARD_DONE
