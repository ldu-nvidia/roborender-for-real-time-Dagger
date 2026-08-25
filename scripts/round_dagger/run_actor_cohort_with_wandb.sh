#!/usr/bin/env bash
# Optional W&B scalar/image/video sidecar around one synchronous actor cohort.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=${ROUND_DAGGER_ACTOR_RUN_DIR:?missing ROUND_DAGGER_ACTOR_RUN_DIR}
case "$RUN_DIR" in /*) ;; *) echo "actor run directory must be absolute" >&2; exit 2 ;; esac
TRAIN_DIR=${TRAIN_RUN_DIR:?missing TRAIN_RUN_DIR}
case "$TRAIN_DIR" in /*) ;; *) echo "TRAIN_RUN_DIR must be absolute" >&2; exit 2 ;; esac
[[ "$RUN_DIR" == "$TRAIN_DIR" ]] || {
    echo "ROUND_DAGGER_ACTOR_RUN_DIR must equal TRAIN_RUN_DIR" >&2; exit 2;
}

# Validate the independent actor identity before the explicitly selected owner
# is allowed to create any remote telemetry state.  The other rank bypasses the
# sidecar only after proving it is one half of the exact production cohort.
ACTOR_RANK=${ROUND_DAGGER_ACTOR_RANK:?missing ROUND_DAGGER_ACTOR_RANK}
ACTOR_WORLD_SIZE=${ROUND_DAGGER_ACTOR_WORLD_SIZE:?missing ROUND_DAGGER_ACTOR_WORLD_SIZE}
[[ $ACTOR_RANK == 0 || $ACTOR_RANK == 1 ]] || {
    echo "production actor rank must be exactly 0 or 1" >&2; exit 2;
}
[[ $ACTOR_WORLD_SIZE == 2 ]] || {
    echo "production actor world size must be exactly 2" >&2; exit 2;
}

ENABLED=${ROUND_DAGGER_WANDB_ENABLED:-0}
case "${ENABLED,,}" in
    0|false|no)
        exec /usr/bin/env bash "$ROOT/scripts/round_dagger/run_actor_cohort.sh"
        ;;
    1|true|yes) ;;
    *) echo "ROUND_DAGGER_WANDB_ENABLED must be boolean" >&2; exit 2 ;;
esac

# The production pair selects exactly one actually-launched rank on every
# unsealed attempt.  Keep the rank-local primitive backwards compatible for
# standalone validation, where rank zero was historically the sole owner, but
# never infer ownership when any pair-lifecycle variable is present: a retry
# must receive the pair launcher's explicit decision.
if [[ -v ROUND_DAGGER_WANDB_SIDECAR_OWNER ]]; then
    SIDECAR_OWNER=$ROUND_DAGGER_WANDB_SIDECAR_OWNER
elif [[ -v ROUND_DAGGER_WANDB_PAIR_LIFECYCLE ||
        -v ROUND_DAGGER_WANDB_ACTOR_DONE_MARKER ||
        -v ROUND_DAGGER_WANDB_PAIR_FINISH_MARKER ]]; then
    echo "missing explicit W&B sidecar owner for pair lifecycle" >&2
    exit 2
elif [[ $ACTOR_RANK == 0 ]]; then
    SIDECAR_OWNER=1
else
    SIDECAR_OWNER=0
fi
[[ "$SIDECAR_OWNER" == 0 || "$SIDECAR_OWNER" == 1 ]] || {
    echo "W&B sidecar owner flag must be exactly 0 or 1" >&2; exit 2;
}
if [[ $SIDECAR_OWNER == 0 ]]; then
    exec /usr/bin/env bash "$ROOT/scripts/round_dagger/run_actor_cohort.sh"
fi

WANDB_PYTHON=${ROUND_DAGGER_WANDB_PYTHON:?missing ROUND_DAGGER_WANDB_PYTHON}
WANDB_PROJECT=${ROUND_DAGGER_WANDB_PROJECT:?missing ROUND_DAGGER_WANDB_PROJECT}
WANDB_ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing ROUND_DAGGER_WANDB_ENTITY}
WANDB_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing ROUND_DAGGER_WANDB_EXPECTED_EMAIL}
READY_TIMEOUT=${ROUND_DAGGER_WANDB_READY_TIMEOUT_SECONDS:-120}
POLL_SECONDS=${ROUND_DAGGER_WANDB_POLL_SECONDS:-10}
ERROR_SCANS=${ROUND_DAGGER_WANDB_MAX_ERROR_SCANS:-4}
PAIR_LIFECYCLE=${ROUND_DAGGER_WANDB_PAIR_LIFECYCLE:-0}
case "$PAIR_LIFECYCLE" in 0|1) ;; *) echo "W&B pair lifecycle must be 0 or 1" >&2; exit 2 ;; esac

[ -x "$WANDB_PYTHON" ] || { echo "W&B sidecar Python is not executable" >&2; exit 2; }
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    echo "W&B ready timeout must be a positive integer" >&2; exit 2;
}
mkdir -p "$RUN_DIR/telemetry" "$RUN_DIR/logs"
[ -f "$RUN_DIR/telemetry/WANDB_PUBLIC_CONFIG.json" ] && \
    [ ! -L "$RUN_DIR/telemetry/WANDB_PUBLIC_CONFIG.json" ] || {
        echo "missing safe public W&B config" >&2; exit 2;
    }

READY="$RUN_DIR/telemetry/WANDB_WATCHER_READY.json"
FINISH_MARKER=RUN_DONE
ACTOR_DONE=
PAIR_HEARTBEAT=
PAIR_ORPHAN_TIMEOUT=
if [ "$PAIR_LIFECYCLE" = 1 ]; then
    FINISH_MARKER=${ROUND_DAGGER_WANDB_PAIR_FINISH_MARKER:?missing pair finish marker}
    ACTOR_DONE=${ROUND_DAGGER_WANDB_ACTOR_DONE_MARKER:?missing actor done marker}
    PAIR_HEARTBEAT=${ROUND_DAGGER_WANDB_PAIR_HEARTBEAT:?missing pair heartbeat}
    PAIR_ORPHAN_TIMEOUT=${ROUND_DAGGER_WANDB_PAIR_ORPHAN_TIMEOUT_SECONDS:-30}
    case "$FINISH_MARKER" in
        /*|*..*|*//*|"") echo "unsafe W&B pair finish marker" >&2; exit 2 ;;
    esac
    case "$ACTOR_DONE" in /*) ;; *) echo "W&B actor-done marker must be absolute" >&2; exit 2 ;; esac
    [[ "$ACTOR_DONE" == "$RUN_DIR"/* ]] || {
        echo "W&B actor-done marker must be inside the common run directory" >&2; exit 2;
    }
    [[ "$PAIR_HEARTBEAT" == "$RUN_DIR"/* && -f "$PAIR_HEARTBEAT" && \
       ! -L "$PAIR_HEARTBEAT" ]] || {
        echo "W&B pair heartbeat must be a safe file inside the run directory" >&2; exit 2;
    }
    [[ "$PAIR_ORPHAN_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
        echo "W&B pair orphan timeout must be a positive integer" >&2; exit 2;
    }
fi
DONE="$RUN_DIR/$FINISH_MARKER"
LOG="$RUN_DIR/logs/wandb_sidecar.log"
MARKERS=("$READY" "$DONE")
[ -z "$ACTOR_DONE" ] || MARKERS+=("$ACTOR_DONE")
for marker in "${MARKERS[@]}"; do
    if [ -e "$marker" ] || [ -L "$marker" ]; then
        [ -f "$marker" ] && [ ! -L "$marker" ] || {
            echo "unsafe W&B lifecycle marker" >&2; exit 2;
        }
        rm -f -- "$marker"
    fi
done
WATCHER=(
    "$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_run_watcher.py"
    --run-dir "$RUN_DIR"
    --project "$WANDB_PROJECT"
    --entity "$WANDB_ENTITY"
    --expected-email "$WANDB_EMAIL"
    --authorization-scope team
    --poll-interval-s "$POLL_SECONDS"
    --max-consecutive-error-scans "$ERROR_SCANS"
    --ready-marker "$READY"
    --finish-marker "$FINISH_MARKER"
)
[ -z "${ROUND_DAGGER_WANDB_NAME:-}" ] || WATCHER+=(--name "$ROUND_DAGGER_WANDB_NAME")
[ -z "${ROUND_DAGGER_WANDB_GROUP:-}" ] || WATCHER+=(--group "$ROUND_DAGGER_WANDB_GROUP")

"${WATCHER[@]}" >"$LOG" 2>&1 &
WATCHER_PID=$!

ready=0
for ((second=0; second<READY_TIMEOUT; second++)); do
    if [ -s "$READY" ] && [ ! -L "$READY" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
        wait "$WATCHER_PID" || true
        echo "W&B sidecar exited before readiness" >&2
        exit 2
    fi
    sleep 1
done
if [ "$ready" != 1 ]; then
    # Request a clean final drain rather than terminating the sidecar process.
    : >"$DONE"
    wait "$WATCHER_PID" || true
    echo "W&B sidecar readiness timed out" >&2
    exit 2
fi

set +e
/usr/bin/env bash "$ROOT/scripts/round_dagger/run_actor_cohort.sh"
ACTOR_STATUS=$?
set -e

# In pair mode the selected owner reports its local actor result, but the outer pair owns
# the finish marker.  It writes that marker only after rank one is reaped, so
# the sole sidecar drains the complete common telemetry stream.  Standalone
# mode retains the original actor-scoped lifecycle.
if [ "$PAIR_LIFECYCLE" = 1 ]; then
    printf '%s\n' "$ACTOR_STATUS" >"$ACTOR_DONE"
    while [ ! -f "$DONE" ]; do
        if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
            break
        fi
        NOW=$(date +%s)
        HEARTBEAT_MTIME=$(stat -c %Y "$PAIR_HEARTBEAT" 2>/dev/null || echo 0)
        if [ $((NOW - HEARTBEAT_MTIME)) -gt "$PAIR_ORPHAN_TIMEOUT" ]; then
            : >"$DONE"
            echo "W&B pair supervisor lease expired; self-finalizing sidecar" >&2
            break
        fi
        sleep 1
    done
else
    : >"$DONE"
fi
set +e
wait "$WATCHER_PID"
WATCHER_STATUS=$?
set -e

if [ "$ACTOR_STATUS" -ne 0 ]; then
    echo "actor cohort failed; immutable collection was not sealed" >&2
    exit "$ACTOR_STATUS"
fi
if [ "$WATCHER_STATUS" -ne 0 ]; then
    echo "W&B scalar/image/video sidecar failed" >&2
    exit "$WATCHER_STATUS"
fi
