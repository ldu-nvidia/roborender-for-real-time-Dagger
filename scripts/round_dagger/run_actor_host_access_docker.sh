#!/usr/bin/env bash
# Coordinator-side normalization of legacy root-container actor telemetry.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
SOURCE_ROOT=${ROUND_DAGGER_COORDINATOR_SOURCE_ROOT:?missing ROUND_DAGGER_COORDINATOR_SOURCE_ROOT}
RUN_DIR=${TRAIN_RUN_DIR:?missing TRAIN_RUN_DIR}

fatal() { echo "actor host-access Docker preflight failed: $*" >&2; exit 2; }

[[ ${ROUND_DAGGER_TRUSTED_COORDINATOR_HOST_ACCESS:-0} == 1 ]] || \
    fatal "untrusted coordinator dispatch"
CAMPAIGN_ROOT=${ROUND_DAGGER_CAMPAIGN_ROOT:?missing ROUND_DAGGER_CAMPAIGN_ROOT}
case "$SOURCE_ROOT:$RUN_DIR:$CAMPAIGN_ROOT" in
    /*:/*:/*) ;;
    *) fatal "source, run, and campaign paths must be absolute" ;;
esac
[[ $ROOT == "$SOURCE_ROOT" && -d $SOURCE_ROOT && ! -L $SOURCE_ROOT ]] || \
    fatal "handoff source differs from the coordinator source"
[[ $(cd "$SOURCE_ROOT" && pwd -P) == "$SOURCE_ROOT" ]] || \
    fatal "coordinator source is not canonical"
[[ -d $RUN_DIR && ! -L $RUN_DIR && $(cd "$RUN_DIR" && pwd -P) == "$RUN_DIR" ]] || \
    fatal "run directory is unsafe or non-canonical"
[[ $RUN_DIR != "$SOURCE_ROOT" && $RUN_DIR != "$SOURCE_ROOT"/* && \
   $SOURCE_ROOT != "$RUN_DIR"/* ]] || \
    fatal "writable run directory must not overlap the read-only source"
[[ ${ROUND_DAGGER_COORDINATOR_SOURCE_INVENTORY_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || \
    fatal "coordinator source inventory SHA-256 is malformed"

HOST_UID=$(id -u)
HOST_GID=$(id -g)
[[ $HOST_UID =~ ^[1-9][0-9]*$ && $HOST_GID =~ ^[1-9][0-9]*$ ]] || \
    fatal "host UID/GID must be positive integers"
[[ $(stat -c %u "$RUN_DIR") == "$HOST_UID" && \
   $(stat -c %g "$RUN_DIR") == "$HOST_GID" ]] || \
    fatal "current host identity does not own the actor run directory"

DOCKER=${ROUND_DAGGER_DOCKER_BIN:-$(command -v docker || true)}
IMAGE=${ROUND_DAGGER_RTX_IMAGE:?missing ROUND_DAGGER_RTX_IMAGE}
EXPECTED_IMAGE_ID=${ROUND_DAGGER_RTX_IMAGE_ID_SHA256:?missing ROUND_DAGGER_RTX_IMAGE_ID_SHA256}
case "$DOCKER" in /*) ;; *) fatal "Docker executable must be absolute" ;; esac
[[ -x $DOCKER && ! -L $DOCKER ]] || fatal "Docker executable is unsafe or unavailable"
[[ $EXPECTED_IMAGE_ID =~ ^[0-9a-f]{64}$ ]] || fatal "actor image ID SHA-256 is malformed"
ACTUAL_IMAGE_ID=$(
    "$DOCKER" image inspect "$IMAGE" --format '{{.Id}}'
) || fatal "cannot inspect pinned actor image"
[[ $ACTUAL_IMAGE_ID == "sha256:$EXPECTED_IMAGE_ID" ]] || \
    fatal "actor image ID differs from the pinned production image"
PINNED_IMAGE="sha256:$EXPECTED_IMAGE_ID"

CAMPAIGN=${ROUND_DAGGER_CAMPAIGN_ID:?missing ROUND_DAGGER_CAMPAIGN_ID}
MACRO_ROUND=${ROUND_DAGGER_MACRO_ROUND:?missing ROUND_DAGGER_MACRO_ROUND}
COHORT_INDEX=${ROUND_DAGGER_COHORT_INDEX:?missing ROUND_DAGGER_COHORT_INDEX}
[[ $CAMPAIGN =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "campaign ID is unsafe"
[[ $MACRO_ROUND =~ ^(0|[1-9][0-9]*)$ && \
   $COHORT_INDEX =~ ^(0|[1-9][0-9]*)$ ]] || \
    fatal "round/cohort identity must use canonical integers"
printf -v EXPECTED_RUN '%s/actors/%s/macro_round_%06d/cohort_%02d_run' \
    "$CAMPAIGN_ROOT" "$CAMPAIGN" "$((10#$MACRO_ROUND))" \
    "$((10#$COHORT_INDEX))"
[[ $RUN_DIR == "$EXPECTED_RUN" ]] || \
    fatal "run directory differs from the exact campaign cohort"
CONTAINER="rd-${CAMPAIGN}-m${MACRO_ROUND}-c${COHORT_INDEX}-host-access-$$"

# This root operation has no network, GPU, host IPC, or source write access.
# Its sealed Python module is limited to telemetry/.wandb_media below RUN_DIR;
# the immutable cohort and rollout shards are outside that traversal.
exec "$DOCKER" run --rm \
    --name "$CONTAINER" \
    --user 0:0 \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --cap-drop=ALL \
    --cap-add=CHOWN \
    --cap-add=FOWNER \
    --cap-add=DAC_OVERRIDE \
    --security-opt=no-new-privileges \
    --ipc=private \
    --network=none \
    --workdir "$ROOT" \
    --mount "type=bind,src=$SOURCE_ROOT,dst=$SOURCE_ROOT,readonly" \
    --mount "type=bind,src=$RUN_DIR,dst=$RUN_DIR" \
    "$PINNED_IMAGE" \
    python3 -m round_dagger.actor_host_access normalize \
        --run-dir "$RUN_DIR" --host-uid "$HOST_UID" --host-gid "$HOST_GID"
