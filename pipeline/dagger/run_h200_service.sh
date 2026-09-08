#!/usr/bin/env bash
# Launch exactly one H200-side DAgger service role (one process per container).
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
ROLE=${1:?usage: $0 coordinator|trajectory|registry|learner}
case "$ROLE" in
    coordinator) DEFAULT_PORT=18610 ;;
    trajectory)  DEFAULT_PORT=18611 ;;
    registry)    DEFAULT_PORT=18612 ;;
    learner)     DEFAULT_PORT=18613 ;;
    *) echo "invalid service role: $ROLE" >&2; exit 2 ;;
esac

PYTHON=${DAGGER_SERVICE_PYTHON:-/usr/bin/python3}
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

args=(
    -m pipeline.dagger.services.server
    --role "$ROLE"
    --host "$HOST"
    --port "$PORT"
    --state-root "$STATE_ROOT"
    --token-file "$TOKEN_FILE"
)
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
