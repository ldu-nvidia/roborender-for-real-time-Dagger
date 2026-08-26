#!/usr/bin/env bash
# Run one content-bound 1-rank x 1-env x 2-step train-replay smoke cohort.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 2 ]] || {
    echo "usage: run_actor_train_smoke_docker.sh ABSOLUTE_PLAN_JSON COHORT_INDEX" >&2
    exit 2
}
PLAN=$1
COHORT_INDEX=$2
RANK_SCRIPT=$ROOT/scripts/round_dagger/run_actor_validation.sh

case "$PLAN:$RANK_SCRIPT" in
    /*:/*) ;;
    *) echo "train-smoke plan and actor primitive paths must be absolute" >&2; exit 2 ;;
esac
[[ -f $PLAN && ! -L $PLAN && -f $RANK_SCRIPT && ! -L $RANK_SCRIPT ]] || {
    echo "unsafe or missing train-smoke plan or actor primitive" >&2; exit 2;
}
[[ $COHORT_INDEX =~ ^[0-3]$ ]] || {
    echo "train-smoke cohort index must be exactly 0, 1, 2, or 3" >&2; exit 2;
}
[[ ${ROUND_DAGGER_VALIDATION_TRAIN_ACK:-} == bounded-train-replay ]] || {
    echo "train-smoke launch requires explicit ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay" >&2
    exit 2
}

STATUS_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    -m round_dagger.train_smoke status --plan "$PLAN" --cohort-index "$COHORT_INDEX")
STATE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' \
    <<<"$STATUS_JSON")
if [[ $STATE == sealed ]]; then
    printf '%s\n' "$STATUS_JSON"
    echo "ACTOR_TRAIN_SMOKE_ALREADY_SEALED cohort=$COHORT_INDEX"
    exit 0
fi
[[ $STATE == empty ]] || {
    echo "train-smoke cohort contains immutable partial or complete evidence; refusing to overwrite" >&2
    exit 2
}

# Consume a NUL-delimited key/value stream; no shell evaluation is used.
while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    printf -v "$name" '%s' "$value"
    export "$name"
done < <(
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.train_smoke environment \
        --plan "$PLAN" --cohort-index "$COHORT_INDEX" --null
)

# The plan is authoritative for the actor runtime.  Ambient distributed
# identities and actor-rank identities must never turn this into DDP or into a
# partition of another cohort.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT ROUND_DAGGER_ACTOR_RANK ROUND_DAGGER_ACTOR_WORLD_SIZE
unset ROUND_DAGGER_VALIDATION_ACTOR_RANK

DOCKER=${ROUND_DAGGER_DOCKER_BIN:-$(command -v docker || true)}
IMAGE=${ROUND_DAGGER_RTX_IMAGE:?missing ROUND_DAGGER_RTX_IMAGE}
IPC_MODE=${ROUND_DAGGER_ACTOR_DOCKER_IPC:-private}
SHM_SIZE=${ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE:-32g}
MOUNT_ROOTS=${ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS:-/mnt/data1,/mnt/data2}
GPU=${ROUND_DAGGER_ACTOR_DOCKER_GPU_ID:?missing planned train-smoke GPU ID}

case "$DOCKER" in /*) ;; *) echo "Docker executable must be absolute" >&2; exit 2 ;; esac
[[ -x $DOCKER && ! -L $DOCKER ]] || {
    echo "Docker executable is unsafe or unavailable" >&2; exit 2;
}
[[ $IPC_MODE == private ]] || {
    echo "train-smoke actor containers require private IPC; host IPC is forbidden" >&2
    exit 2
}
[[ $SHM_SIZE =~ ^([1-9][0-9]*)(g|G)$ ]] || {
    echo "train-smoke Docker shm size must be an integral number of GiB" >&2
    exit 2
}
SHM_GIB=${BASH_REMATCH[1]}
((SHM_GIB >= 16)) || {
    echo "train-smoke Docker shm size must be at least 16 GiB" >&2; exit 2;
}

IFS=, read -r -a RAW_MOUNTS <<<"$MOUNT_ROOTS"
[[ ${#RAW_MOUNTS[@]} -ge 1 ]] || { echo "no Docker mount roots" >&2; exit 2; }
MOUNTS=()
for mount_root in "${RAW_MOUNTS[@]}"; do
    case "$mount_root" in /*) ;; *) echo "Docker mount roots must be absolute" >&2; exit 2 ;; esac
    [[ -d $mount_root && ! -L $mount_root ]] || {
        echo "unsafe or missing Docker mount root: $mount_root" >&2; exit 2;
    }
    MOUNTS+=(--mount "type=bind,src=$mount_root,dst=$mount_root")
done

[[ $ROUND_DAGGER_CAMPAIGN_ID == rtx-train-smoke-v1 && \
   $ROUND_DAGGER_REPOSITORY_ROOT == "$ROOT" && \
   $ROUND_DAGGER_MACRO_ROUND == 0 && \
   $ROUND_DAGGER_COHORT_INDEX == "$COHORT_INDEX" && \
   $ROUND_DAGGER_VALIDATION_RANKS == 1 && \
   $ROUND_DAGGER_VALIDATION_ENVS_PER_RANK == 1 && \
   $ROUND_DAGGER_VALIDATION_STEPS == 2 && \
   $ROUND_DAGGER_VALIDATION_PURPOSE == train && \
   $ROUND_DAGGER_VALIDATION_TRAIN_ACK == bounded-train-replay && \
   $GPU == $((COHORT_INDEX % 2)) ]] || {
    echo "train-smoke environment differs from the exact 1x1x2 train contract" >&2
    exit 2
}
CONTAINER="rdt-rtx-train-smoke-v1-m0-c${COHORT_INDEX}-r0"

# This whitelist deliberately excludes W&B credentials: actor-side W&B is
# disabled by the frozen validation primitive, and the later learner/W&B gate
# has its own bounded credential path.
ENV_NAMES=(
    ACCELERATE ANCHORS DRY_RUN GR00T_PYTHON GR00T_ROOT
    IMAGE3F_ENDPOINT_MANIFEST ROUND_DAGGER_BETA
    ROUND_DAGGER_BETA_SHA256 ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_COHORT_INDEX
    ROUND_DAGGER_CONFIG_SHA256 ROUND_DAGGER_ENVS_PER_SHARD
    ROUND_DAGGER_EXPECTED_RENDERERS ROUND_DAGGER_GLOBAL_ENV_OFFSET
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_MIN_ANCHORS
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_MANIFEST
    ROUND_DAGGER_POLICY_SHA256 ROUND_DAGGER_POLICY_SPEC
    ROUND_DAGGER_RENDERER_SHA256 ROUND_DAGGER_ROLLOUT_ID
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_TEACHER_CHECKPOINT
    ROUND_DAGGER_TEACHER_SHA256 ROUND_DAGGER_VALIDATION_ENVS_PER_RANK
    ROUND_DAGGER_VALIDATION_PURPOSE ROUND_DAGGER_VALIDATION_RANKS
    ROUND_DAGGER_VALIDATION_STEPS ROUND_DAGGER_VALIDATION_TRAIN_ACK
    ROBORENDER_TELEMETRY_DIR ROLLOUT_STEPS TRAIN_RUN_DIR
)
ENV_ARGS=()
for name in "${ENV_NAMES[@]}"; do
    case "$name" in
        ROUND_DAGGER_RUNTIME_ADAPTER_*|ROUND_DAGGER_RUNTIME_AB_*)
            echo "runtime adapter/AB variables must not enter the train-smoke allowlist: $name" >&2
            exit 2
            ;;
    esac
    [[ -n ${!name:-} ]] || { echo "empty exact train-smoke environment: $name" >&2; exit 2; }
    ENV_ARGS+=(--env "$name")
done
# This train-replay smoke validates the production/default path.  Do not
# inherit an ambient shadow selection, and clear any acknowledgement baked
# into the image environment.
ENV_ARGS+=(
    --env "ROUND_DAGGER_RUNTIME_ADAPTER_MODE=legacy"
    --env "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK="
    --env "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK="
    --env "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE="
    --env "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE="
    --env "ROUND_DAGGER_RUNTIME_AB_PLAN="
    --env "ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256="
    --env "ROUND_DAGGER_RUNTIME_AB_ARM="
    --env "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT="
    --env "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256="
    --env "ROUND_DAGGER_RUNTIME_AB_EVIDENCE="
)

exec "$DOCKER" run --rm \
    --name "$CONTAINER" \
    --gpus "\"device=$GPU\"" \
    --ipc=private \
    --shm-size="$SHM_SIZE" \
    --network=host \
    --workdir "$ROOT" \
    "${MOUNTS[@]}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" \
    /usr/bin/env bash "$RANK_SCRIPT"
