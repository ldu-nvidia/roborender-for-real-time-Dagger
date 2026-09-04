#!/usr/bin/env bash
# Evaluate policy-v0 fixed256 action quality before the first collection hook.
set -Eeuo pipefail

fatal() { echo "policy-v0 baseline failed: $*" >&2; exit 2; }

ROOT=${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:?missing evaluation source root}
[[ ${ROUND_DAGGER_TRUSTED_BASELINE_DISPATCH:-0} == 1 ]] || \
    fatal "trusted production baseline dispatch is required"
[[ $ROOT == /* && -d $ROOT && ! -L $ROOT ]] || fatal "evaluation source is unsafe"
CAMPAIGN_ROOT=${ROUND_DAGGER_CAMPAIGN_ROOT:?missing campaign root}
HELDOUT=${ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT:?missing held-out snapshot}
HELDOUT_FILE_SHA=${ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256:?missing held-out file SHA}
SOURCE_SHA=${ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256:?missing source identity}
DOCKER=${ROUND_DAGGER_DOCKER_BIN:?missing Docker binary}
IMAGE=${ROUND_DAGGER_RTX_IMAGE:?missing RTX image}
IMAGE_ID=${ROUND_DAGGER_RTX_IMAGE_ID_SHA256:?missing RTX image identity}
NETRC=${ROUND_DAGGER_WANDB_NETRC:?missing W&B netrc}
ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing W&B entity}
PROJECT=${ROUND_DAGGER_WANDB_PROJECT:?missing W&B project}
EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing W&B email}

CONTROL_PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:?missing control Python}
for value in "$HELDOUT_FILE_SHA" "$SOURCE_SHA" "$IMAGE_ID"; do
    [[ $value =~ ^[0-9a-f]{64}$ ]] || fatal "artifact SHA-256 is malformed"
done
for path in "$CAMPAIGN_ROOT" "$ROOT"; do
    [[ $path == /* && -d $path && ! -L $path ]] || fatal "directory input is unsafe: $path"
done
for path in "$HELDOUT" "$NETRC"; do
    [[ $path == /* && -f $path && ! -L $path ]] || fatal "file input is unsafe: $path"
done
[[ $(sha256sum -- "$HELDOUT" | awk '{print $1}') == "$HELDOUT_FILE_SHA" ]] || \
    fatal "held-out snapshot SHA-256 mismatch"
[[ -x $DOCKER && ! -L $DOCKER ]] || fatal "Docker executable is unsafe"
[[ -x $CONTROL_PYTHON ]] || fatal "control Python is unavailable"
[[ $(stat -c %a -- "$NETRC") == 600 ]] || fatal "W&B netrc mode must be 0600"
[[ $ENTITY == nvidia && $PROJECT == lepton-dagger-production-1 ]] || \
    fatal "W&B baseline destination differs from production"

GR00T_ROOT="$ROOT/GR00T-VisualSim2Real"
[[ -d $GR00T_ROOT && ! -L $GR00T_ROOT ]] || fatal "sealed GR00T source is missing"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT" "$CONTROL_PYTHON" \
    "$ROOT/scripts/round_dagger/verify_sealed_source_snapshot.py" \
    --root "$ROOT" --expected-inventory-sha256 "$SOURCE_SHA" >/dev/null || \
    fatal "sealed evaluation source verification failed"
ACTUAL_IMAGE_ID=$(
    "$DOCKER" image inspect --format '{{.Id}}' "$IMAGE"
) || fatal "cannot inspect RTX image"
[[ $ACTUAL_IMAGE_ID == "sha256:$IMAGE_ID" ]] || fatal "RTX image identity changed"
CONTAINER_HOME=/home/ldu
CONTAINER_NETRC=$CONTAINER_HOME/.netrc
CONTAINER="rd-${ROUND_DAGGER_CAMPAIGN_ID:?missing campaign ID}-policy-v0-baseline"
[[ $CONTAINER =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "container name is unsafe"

MOUNTS=()
IFS=, read -r -a ROOTS <<<"${ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS:-/mnt/data1,/mnt/data2}"
for mount_root in "${ROOTS[@]}"; do
    [[ $mount_root == /* && -d $mount_root && ! -L $mount_root ]] || \
        fatal "Docker mount root is unsafe: $mount_root"
    MOUNTS+=(--mount "type=bind,src=$mount_root,dst=$mount_root")
done
MOUNTS+=(
    --mount "type=bind,src=$ROOT,dst=$ROOT,readonly"
    --mount "type=bind,src=$HELDOUT,dst=$HELDOUT,readonly"
    --mount "type=bind,src=$NETRC,dst=$CONTAINER_NETRC,readonly"
)

exec "$DOCKER" run --rm --name "$CONTAINER" \
    --gpus '"device=0"' --ipc=private --shm-size=16g --network=host \
    --workdir "$ROOT" "${MOUNTS[@]}" \
    --env "HOME=$CONTAINER_HOME" --env "NETRC=$CONTAINER_NETRC" \
    --env WANDB_MODE=online --env PYTHONDONTWRITEBYTECODE=1 \
    "$IMAGE" /opt/round-dagger-rtx/bin/python \
    "$ROOT/scripts/round_dagger/evaluate_active_policy_heldout.py" \
    --campaign-root "$CAMPAIGN_ROOT" \
    --heldout-snapshot "$HELDOUT" \
    --heldout-snapshot-file-sha256 "$HELDOUT_FILE_SHA" \
    --code-sha256 "$SOURCE_SHA" --device cuda --batch-size 32 \
    --gr00t-root "$GR00T_ROOT" \
    --wandb-project "$PROJECT" --wandb-entity "$ENTITY" \
    --wandb-expected-email "$EMAIL" --wandb-authorization-scope team
