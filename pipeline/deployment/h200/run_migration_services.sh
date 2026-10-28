#!/usr/bin/env bash
# Genuine model services for a bounded migration smoke or throughput sweep. No learner
# job is automatically submitted and no production state directory is reused.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
SMOKE_ID=${1:?usage: $0 leo-roborender-beta-smoke-ID [attempt-ID]}
ATTEMPT=${2:-attempt1}
[[ $SMOKE_ID =~ ^leo-roborender-beta-smoke-[a-z0-9-]+$ ]] || exit 2
[[ $ATTEMPT =~ ^attempt[0-9]+$ ]] || exit 2
[[ ${ROBORENDER_SMOKE_ID:-} == "$SMOKE_ID" ]] || {
  echo "workload identity differs from the reserved smoke pod" >&2; exit 2;
}
GPU_COUNT=${ROBORENDER_BENCHMARK_GPU_COUNT:-1}
[[ $GPU_COUNT == 1 || $GPU_COUNT == 4 ]] || exit 2
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) == "$GPU_COUNT" ]] || {
  echo "allocated GPU count differs from the explicit benchmark size" >&2; exit 2;
}
WORK="$ROOT/runtime/migrations/alpha-to-beta-20261008/h200/$SMOKE_ID/$ATTEMPT"
mkdir -p "$WORK/logs"
[[ ! -e $WORK/services/state.sqlite3 ]] || {
  echo "refusing to reuse existing simulator/learner smoke state" >&2; exit 2;
}
export DAGGER_SERVICE_STATE_ROOT="$WORK/services"
export DAGGER_SERVICE_TOKEN_FILE="$WORK/services.token"
export DAGGER_RENDER_POOL_ROOT="$WORK/render_pool"
export DAGGER_LEARNER_WORKER="$ROOT/pipeline/dagger/run_raven_learner_worker_h200.sh"
if [[ $GPU_COUNT == 4 ]]; then
  export DAGGER_RENDER_GPUS=0,1,2,3 IMAGE3F_PORTS=18500,18501,18502,18503
  export DAGGER_GATEWAY_PORTS=18620,18621,18622,18623
  export DAGGER_STUDENT_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
else
  export DAGGER_RENDER_GPUS=0 IMAGE3F_PORTS=18500 DAGGER_GATEWAY_PORTS=18620
  export DAGGER_STUDENT_DEVICES=cuda:0
fi
export DAGGER_EPHEMERAL_WINDOW_CACHE=false
export WANDB_MODE=${ROBORENDER_RUNTIME_WANDB_MODE:-disabled}
[[ $WANDB_MODE == disabled || $WANDB_MODE == online ]] || exit 2
PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
# Create the token once before concurrent role startup; never print it.
python3 - "$DAGGER_SERVICE_TOKEN_FILE" <<'PY'
import os, secrets, sys
fd = os.open(sys.argv[1], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write(secrets.token_hex(32) + "\n")
PY
for role in coordinator trajectory registry learner; do
  bash "$ROOT/pipeline/dagger/run_h200_service.sh" "$role" >"$WORK/logs/$role.log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  case "$role" in
    coordinator) port=18610;; trajectory) port=18611;;
    registry) port=18612;; learner) port=18613;;
  esac
  # Schema/WAL initialization is a cold-path operation. Wait until one role
  # is ready before another process opens the new SQLite database.
  deadline=$((SECONDS + 60))
  until curl -fsS --max-time 1 "http://127.0.0.1:$port/ready" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "$role exited before readiness" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "$role readiness timed out" >&2; exit 1; }
    sleep 1
  done
done
bash "$ROOT/pipeline/dagger/run_h200_render_pool.sh" >"$WORK/logs/render_pool.log" 2>&1 &
PIDS+=("$!")
printf 'Migration model services starting: %s (%s H200, no automatic optimization)\n' "$SMOKE_ID" "$GPU_COUNT"
wait -n "${PIDS[@]}"
