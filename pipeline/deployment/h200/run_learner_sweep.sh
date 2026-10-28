#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
POD=${1:?pod identity required}
[[ $POD == leo-roborender-beta-smoke-1009-1734 && ${ROBORENDER_SMOKE_ID:-} == "$POD" ]] || exit 2
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) == 4 ]] || exit 2
OUT="$ROOT/runtime/migrations/alpha-to-beta-20261008/h200/$POD/learner-sweep"
mkdir -p "$OUT"
source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
export PYTHONPATH="$ROOT:$PYTHONPATH" WANDB_MODE=disabled
cd "$ROOT"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
BATCHES=(8 16 28 56)
for recompute in on off; do
  PIDS=()
  for gpu in 0 1 2 3; do
    batch=${BATCHES[$gpu]}
    CUDA_VISIBLE_DEVICES=$gpu "$GR00T_PYTHON" -m pipeline.deployment.h200.benchmark_learner \
      --output "$OUT/b${batch}-recompute-${recompute}" --batch-chunks "$batch" \
      --checkpoint-activations "$recompute" >"$OUT/b${batch}-recompute-${recompute}.log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || true; done
done
PIDS=()
printf 'Learner sweep finished; inspect each result.json (OOM cases are not winners).\n'
