#!/usr/bin/env bash
# Launch exactly one H200-side DAgger service role (one process per container).
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
ROLE=${1:?usage: $0 coordinator|trajectory|registry|learner|evaluation|render-student}
case "$ROLE" in
    coordinator) DEFAULT_PORT=18610 ;;
    trajectory)  DEFAULT_PORT=18611 ;;
    registry)    DEFAULT_PORT=18612 ;;
    learner)     DEFAULT_PORT=18613 ;;
    evaluation)  DEFAULT_PORT=18614 ;;
    render-student) DEFAULT_PORT=18620 ;;
    *) echo "invalid service role: $ROLE" >&2; exit 2 ;;
esac

if [[ $ROLE == render-student ]]; then
    source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
    PYTHON=${DAGGER_SERVICE_PYTHON:-$GR00T_PYTHON}
else
    PYTHON=${DAGGER_SERVICE_PYTHON:-/usr/bin/python3}
fi
HOST=${DAGGER_SERVICE_HOST:-127.0.0.1}
PORT=${DAGGER_SERVICE_PORT:-$DEFAULT_PORT}
STATE_ROOT=${DAGGER_SERVICE_STATE_ROOT:-$ROOT/runtime/dagger_services}
TOKEN_FILE=${DAGGER_SERVICE_TOKEN_FILE:-$ROOT/runtime/secrets/dagger_services.token}

[[ -x $PYTHON ]] || { echo "service Python is missing: $PYTHON" >&2; exit 2; }
[[ $PORT =~ ^[0-9]+$ && $PORT -ge 1024 && $PORT -le 65535 ]] || {
    echo "DAGGER_SERVICE_PORT must be 1024..65535" >&2
    exit 2
}
if [[ ! -f $TOKEN_FILE ]]; then
    mkdir -p "$(dirname "$TOKEN_FILE")"
    umask 077
    /usr/bin/python3 - "$TOKEN_FILE" <<'PY'
import secrets
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(secrets.token_hex(32) + "\n")
PY
fi
[[ $(stat -c %a "$TOKEN_FILE") == 600 ]] || chmod 600 "$TOKEN_FILE"

if [[ $ROLE == render-student ]]; then
    args=(
        -m pipeline.dagger.services.render_student_server
        --host "$HOST"
        --port "$PORT"
        --state-root "$STATE_ROOT"
        --token-file "$TOKEN_FILE"
        --trajectory-url "${DAGGER_TRAJECTORY_URL:-http://127.0.0.1:18611}"
        --image3f-port "${IMAGE3F_PORT:-18500}"
        --device "${DAGGER_STUDENT_DEVICE:-cuda:0}"
        --shared-local-trajectory
    )
    if [[ -n ${DAGGER_STUDENT_DEVICES:-} || -n ${IMAGE3F_PORTS:-} ]]; then
        [[ -n ${DAGGER_STUDENT_DEVICES:-} && -n ${IMAGE3F_PORTS:-} ]] || {
            echo "DAGGER_STUDENT_DEVICES and IMAGE3F_PORTS must be set together" >&2
            exit 2
        }
        args+=(--devices "$DAGGER_STUDENT_DEVICES" --image3f-ports "$IMAGE3F_PORTS")
    fi
    if [[ -n ${DAGGER_RENDER_TELEMETRY_JSONL:-} ]]; then
        args+=(--telemetry-jsonl "$DAGGER_RENDER_TELEMETRY_JSONL")
    fi
    if [[ -n ${DAGGER_PHASE_TELEMETRY_JSONL:-} ]]; then
        args+=(--phase-telemetry-jsonl "$DAGGER_PHASE_TELEMETRY_JSONL")
    fi
    if [[ ${DAGGER_EPHEMERAL_WINDOW_CACHE:-false} == true ]]; then
        args+=(--ephemeral-window-cache)
    fi
else
    args=(
        -m pipeline.dagger.services.server
        --role "$ROLE"
        --host "$HOST"
        --port "$PORT"
        --state-root "$STATE_ROOT"
        --token-file "$TOKEN_FILE"
    )
fi
if [[ $ROLE == learner ]]; then
    : "${DAGGER_LEARNER_WORKER:?set DAGGER_LEARNER_WORKER to the fixed production worker executable}"
    [[ -x $DAGGER_LEARNER_WORKER ]] || {
        echo "learner worker is not executable: $DAGGER_LEARNER_WORKER" >&2
        exit 2
    }
    args+=(--worker "$DAGGER_LEARNER_WORKER")
fi

cd "$ROOT"
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
exec "$PYTHON" "${args[@]}"
