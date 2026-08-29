#!/usr/bin/env bash
# Recover one sealed actor-pair W&B stream in the pinned RTX actor image.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
SOURCE_ROOT=${ROUND_DAGGER_ACTOR_SOURCE_ROOT:?missing ROUND_DAGGER_ACTOR_SOURCE_ROOT}
RUN_DIR=${TRAIN_RUN_DIR:?missing TRAIN_RUN_DIR}
NETRC_PATH=${ROUND_DAGGER_WANDB_NETRC:?missing ROUND_DAGGER_WANDB_NETRC}

fatal() { echo "W&B recovery Docker preflight failed: $*" >&2; exit 2; }

[[ ${ROUND_DAGGER_TRUSTED_COLLECT_DISPATCH:-0} == 1 ]] || \
    fatal "untrusted collect dispatch"
case "$SOURCE_ROOT:$RUN_DIR:$NETRC_PATH" in
    /*:/*:/*) ;;
    *) fatal "source, run, and netrc paths must be absolute" ;;
esac
[[ $ROOT == "$SOURCE_ROOT" && -d $SOURCE_ROOT && ! -L $SOURCE_ROOT ]] || \
    fatal "recovery source differs from the sealed actor source"
[[ $(cd "$SOURCE_ROOT" && pwd -P) == "$SOURCE_ROOT" ]] || \
    fatal "sealed actor source is not canonical"
[[ -d $RUN_DIR && ! -L $RUN_DIR && $(cd "$RUN_DIR" && pwd -P) == "$RUN_DIR" ]] || \
    fatal "run directory is unsafe or non-canonical"
[[ -f $NETRC_PATH && ! -L $NETRC_PATH && $(stat -c %a "$NETRC_PATH") == 600 ]] || \
    fatal "W&B netrc must be a mode-0600 regular file"
[[ $RUN_DIR != "$SOURCE_ROOT" && $RUN_DIR != "$SOURCE_ROOT"/* && \
   $SOURCE_ROOT != "$RUN_DIR"/* ]] || \
    fatal "writable run directory must not overlap the read-only source"
[[ ${ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || \
    fatal "sealed actor source inventory SHA-256 is malformed"

WANDB_PYTHON=${ROUND_DAGGER_WANDB_PYTHON:?missing ROUND_DAGGER_WANDB_PYTHON}
WANDB_PROJECT=${ROUND_DAGGER_WANDB_PROJECT:?missing ROUND_DAGGER_WANDB_PROJECT}
WANDB_ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing ROUND_DAGGER_WANDB_ENTITY}
WANDB_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing ROUND_DAGGER_WANDB_EXPECTED_EMAIL}

if [[ ${ROUND_DAGGER_WANDB_RECOVERY_IN_CONTAINER:-0} == 1 ]]; then
    case "$WANDB_PYTHON" in /*) ;; *) fatal "W&B Python must be absolute" ;; esac
    [[ -x $WANDB_PYTHON ]] || fatal "W&B Python is not executable in the actor image"

    # All mutable W&B state stays on the explicit run-directory mount.  The
    # image root and sealed source remain read-only, and credentials remain in
    # the independently mounted mode-0600 netrc rather than Docker argv.
    RECOVERY_STATE="$RUN_DIR/.wandb_recovery"
    mkdir -p "$RECOVERY_STATE/home" "$RECOVERY_STATE/config" \
        "$RECOVERY_STATE/cache" "$RECOVERY_STATE/data"
    export HOME="$RECOVERY_STATE/home"
    export NETRC="$NETRC_PATH"
    export WANDB_CONFIG_DIR="$RECOVERY_STATE/config"
    export WANDB_CACHE_DIR="$RECOVERY_STATE/cache"
    export WANDB_DATA_DIR="$RECOVERY_STATE/data"
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
    unset SLURM_PROCID

    command=(
        "$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_run_watcher.py"
        --run-dir "$RUN_DIR"
        --project "$WANDB_PROJECT"
        --entity "$WANDB_ENTITY"
        --expected-email "$WANDB_EMAIL"
        --authorization-scope team --min-age-s 0 --once
    )
    [[ -z ${ROUND_DAGGER_WANDB_NAME:-} ]] || \
        command+=(--name "$ROUND_DAGGER_WANDB_NAME")
    [[ -z ${ROUND_DAGGER_WANDB_GROUP:-} ]] || \
        command+=(--group "$ROUND_DAGGER_WANDB_GROUP")
    "${command[@]}"

    # A zero watcher exit is insufficient by itself.  Deeply rediscover every
    # current artifact and require the durable receipt/state coverage before
    # the actor pair may accept its already-sealed cohort.
    "$WANDB_PYTHON" -m round_dagger.wandb_delivery verify --run-dir "$RUN_DIR"
    exit 0
fi
[[ ${ROUND_DAGGER_WANDB_RECOVERY_IN_CONTAINER:-0} == 0 ]] || \
    fatal "container-entry flag must be exactly 0 or 1"

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
CONTAINER="rd-${CAMPAIGN}-m${MACRO_ROUND}-c${COHORT_INDEX}-wandb-recovery-$$"

# Forward environment names, never credential values.  Deliberately do not
# forward WANDB_API_KEY: authentication is through the read-only netrc mount.
ENV_ARGS=(
    --env "ROUND_DAGGER_WANDB_RECOVERY_IN_CONTAINER=1"
    --env ROUND_DAGGER_TRUSTED_COLLECT_DISPATCH
    --env ROUND_DAGGER_ACTOR_SOURCE_ROOT
    --env ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256
    --env TRAIN_RUN_DIR
    --env ROUND_DAGGER_WANDB_NETRC
    --env ROUND_DAGGER_WANDB_PYTHON
    --env ROUND_DAGGER_WANDB_PROJECT
    --env ROUND_DAGGER_WANDB_ENTITY
    --env ROUND_DAGGER_WANDB_EXPECTED_EMAIL
)
[[ -z ${ROUND_DAGGER_WANDB_NAME:-} ]] || ENV_ARGS+=(--env ROUND_DAGGER_WANDB_NAME)
[[ -z ${ROUND_DAGGER_WANDB_GROUP:-} ]] || ENV_ARGS+=(--env ROUND_DAGGER_WANDB_GROUP)

# W&B upload and local receipt verification need network and CPU only.  No GPU
# device is exposed to this recovery container.
exec "$DOCKER" run --rm \
    --name "$CONTAINER" \
    --user "$(id -u):$(id -g)" \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=256m \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --ipc=private \
    --network=host \
    --workdir "$ROOT" \
    --mount "type=bind,src=$SOURCE_ROOT,dst=$SOURCE_ROOT,readonly" \
    --mount "type=bind,src=$RUN_DIR,dst=$RUN_DIR" \
    --mount "type=bind,src=$NETRC_PATH,dst=$NETRC_PATH,readonly" \
    "${ENV_ARGS[@]}" \
    "$PINNED_IMAGE" \
    /usr/bin/env bash "$ROOT/scripts/round_dagger/run_wandb_recovery_docker.sh"
