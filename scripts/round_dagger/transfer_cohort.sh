#!/usr/bin/env bash
# Resumable RTX -> H100 cohort handoff. Source data is never modified/deleted.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${ROUND_DAGGER_PYTHON:-python3}
DRY_RUN=${DRY_RUN:-0}

usage() {
    echo "usage: $0 SOURCE_COHORT ABSOLUTE_DESTINATION [SSH_HOST]" >&2
    exit 2
}

[ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage
SOURCE=$1
DESTINATION=$2
REMOTE=${3:-}
case "$DESTINATION" in /*) ;; *) echo "destination must be absolute" >&2; exit 2 ;; esac

run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY_RUN'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [ -z "$REMOTE" ]; then
    run env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" -m round_dagger.handoff local \
        --source "$SOURCE" --destination "$DESTINATION"
    exit 0
fi

REMOTE_REPO=${ROUND_DAGGER_REMOTE_REPO:?set ROUND_DAGGER_REMOTE_REPO for SSH handoff}
REMOTE_PYTHON=${ROUND_DAGGER_REMOTE_PYTHON:-python3}
case "$REMOTE_REPO" in /*) ;; *) echo "ROUND_DAGGER_REMOTE_REPO must be absolute" >&2; exit 2 ;; esac

COHORT_SHA=$(env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - \
    "$SOURCE" <<'PY'
import sys
from round_dagger.artifacts import verify_cohort
print(verify_cohort(sys.argv[1])["cohort_sha256"])
PY
)
[[ "$COHORT_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid cohort digest" >&2; exit 2; }
DEST_PARENT=${DESTINATION%/*}
DEST_NAME=${DESTINATION##*/}
[ -n "$DEST_PARENT" ] && [ -n "$DEST_NAME" ] || usage
STAGE="$DEST_PARENT/.incoming-$DEST_NAME-${COHORT_SHA:0:16}"

if [ "$DRY_RUN" = 1 ]; then
    run ssh "$REMOTE" mkdir -p -- "$DEST_PARENT" "$STAGE/shards"
else
    ssh "$REMOTE" bash -s -- "$DEST_PARENT" "$STAGE" <<'REMOTE_PREPARE'
set -Eeuo pipefail
parent=$1
stage=$2
[ ! -L "$parent" ] && [ ! -L "$stage" ] || { echo "unsafe remote handoff path" >&2; exit 2; }
mkdir -p -- "$parent" "$stage/shards"
REMOTE_PREPARE
fi

run rsync --archive --no-links --protect-args --partial --append-verify \
    --delay-updates --delete-delay -- "$SOURCE/" "$REMOTE:$STAGE/"

if [ "$DRY_RUN" = 1 ]; then
    run ssh "$REMOTE" env PYTHONPATH="$REMOTE_REPO" "$REMOTE_PYTHON" \
        -m round_dagger.handoff publish-stage --stage "$STAGE" \
        --destination "$DESTINATION"
else
    ssh "$REMOTE" bash -s -- \
        "$REMOTE_REPO" "$REMOTE_PYTHON" "$STAGE" "$DESTINATION" <<'REMOTE_PUBLISH'
set -Eeuo pipefail
repo=$1
python=$2
stage=$3
destination=$4
cd "$repo"
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$python" \
    -m round_dagger.handoff verify-stage --stage "$stage" >/dev/null
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$python" \
    -m round_dagger.handoff publish-stage --stage "$stage" \
    --destination "$destination"
REMOTE_PUBLISH
fi
