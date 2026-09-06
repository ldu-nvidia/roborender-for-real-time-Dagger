#!/usr/bin/env bash
# Reconstruct Raven's exact three-patch GR00T execution tree.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TARGET=$ROOT/pipeline/GR00T-VisualSim2Real
REFERENCE=$ROOT/pipeline/vendor/raven-fast-roborender-reference/transfer
UPSTREAM=https://github.com/NVlabs/GR00T-VisualSim2Real.git
BASE=92bf0863d4a9b6ee29849736152b7769bd45c49c
EXPECTED=94d79851d9c471b1210be81c75116d9bea835a4d
PATCHES=(
    "$REFERENCE/0001-Add-videogen-observation-pipeline-offline-BC-and-onl.patch"
    "$REFERENCE/0002-Three-view-policy-support.patch"
    "$REFERENCE/0003-Add-DAgger-beta-schedule-for-the-training-rollout.patch"
)

if [[ -d $TARGET/.git ]]; then
    actual=$(git -C "$TARGET" rev-parse HEAD)
    [[ $actual == "$EXPECTED" ]] || {
        echo "existing GR00T checkout is at $actual, expected $EXPECTED" >&2
        exit 2
    }
    [[ -z $(git -C "$TARGET" status --porcelain) ]] || {
        echo "existing exact GR00T checkout is modified: $TARGET" >&2
        exit 2
    }
    echo "exact Raven GR00T checkout already present: $TARGET"
    exit 0
fi
[[ ! -e $TARGET ]] || {
    echo "target exists but is not a Git checkout: $TARGET" >&2
    exit 2
}
for patch in "${PATCHES[@]}"; do
    [[ -f $patch ]] || { echo "missing Raven patch: $patch" >&2; exit 2; }
done

stage=$(mktemp -d "$ROOT/pipeline/.gr00t-bootstrap.XXXXXX")
cleanup() {
    if [[ -n ${stage:-} && -d $stage ]]; then
        case "$stage" in "$ROOT/pipeline/.gr00t-bootstrap."*) rm -rf -- "$stage" ;; esac
    fi
}
trap cleanup EXIT

git clone --filter=blob:none --no-checkout "$UPSTREAM" "$stage/source"
git -C "$stage/source" checkout --detach "$BASE"
# These are the committer values of Raven's supplied three-patch checkout.
# Pinning them makes the reconstruction content- and commit-identical instead
# of merely producing an equivalent tree with today's committer timestamp.
GIT_COMMITTER_NAME=ldu \
GIT_COMMITTER_EMAIL=ldu@nvidia.com \
GIT_COMMITTER_DATE='Tue Aug 18 17:59:50 2026 -0700' \
    git -C "$stage/source" am "${PATCHES[@]}"
actual=$(git -C "$stage/source" rev-parse HEAD)
[[ $actual == "$EXPECTED" ]] || {
    echo "patched GR00T identity mismatch: got $actual, expected $EXPECTED" >&2
    exit 1
}
[[ -z $(git -C "$stage/source" status --porcelain) ]] || {
    echo "patched GR00T checkout is unexpectedly dirty" >&2
    exit 1
}
mv -- "$stage/source" "$TARGET"
rmdir -- "$stage"
stage=
echo "installed exact Raven GR00T checkout at $TARGET ($EXPECTED)"
