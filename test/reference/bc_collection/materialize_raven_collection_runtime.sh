#!/usr/bin/env bash
# Materialize a clean, detached Raven execution tree for BC collection.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
SOURCE=${ROBORENDER_RAVEN_SOURCE_CHECKOUT:-$ROOT/pipeline/GR00T-VisualSim2Real}
TARGET=${ROBORENDER_RAVEN_COLLECTION_RUNTIME:-$ROOT/runtime/raven_gr00t_collection_94d7985}
EXPECTED=94d79851d9c471b1210be81c75116d9bea835a4d

[[ -d $SOURCE/.git ]] || {
    echo "missing Raven source checkout: $SOURCE" >&2
    exit 2
}
git -C "$SOURCE" cat-file -e "$EXPECTED^{commit}"

if [[ -e $TARGET ]]; then
    [[ -d $TARGET/.git || -f $TARGET/.git ]] || {
        echo "existing Raven runtime is not a Git worktree: $TARGET" >&2
        exit 2
    }
    actual=$(git -C "$TARGET" rev-parse HEAD)
    [[ $actual == "$EXPECTED" ]] || {
        echo "Raven runtime commit mismatch: $actual != $EXPECTED" >&2
        exit 2
    }
    [[ -z $(git -C "$TARGET" status --porcelain --untracked-files=all) ]] || {
        echo "Raven runtime has local drift: $TARGET" >&2
        exit 2
    }
    printf '%s\n' "$TARGET"
    exit 0
fi

mkdir -p "$(dirname "$TARGET")"
git -C "$SOURCE" worktree add --detach "$TARGET" "$EXPECTED" >&2
actual=$(git -C "$TARGET" rev-parse HEAD)
[[ $actual == "$EXPECTED" ]] || {
    echo "materialized Raven runtime commit mismatch" >&2
    exit 1
}
[[ -z $(git -C "$TARGET" status --porcelain --untracked-files=all) ]] || {
    echo "materialized Raven runtime is unexpectedly dirty" >&2
    exit 1
}
printf '%s\n' "$TARGET"
