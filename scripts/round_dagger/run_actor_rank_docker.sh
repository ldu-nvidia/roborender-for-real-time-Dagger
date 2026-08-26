#!/usr/bin/env bash
# Single-rank Docker isolation executor for run_actor_cohort_pair.sh.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $# == 2 ]] || {
    echo "usage: run_actor_rank_docker.sh RANK ABSOLUTE_RANK_SCRIPT" >&2; exit 2;
}
RANK=$1
RANK_SCRIPT=$2
WORLD_SIZE=${ROUND_DAGGER_ACTOR_WORLD_SIZE:-}
[[ "$RANK" == 0 || "$RANK" == 1 ]] || {
    echo "Docker actor rank must be exactly 0 or 1" >&2; exit 2;
}
[[ ${ROUND_DAGGER_ACTOR_RANK:-} == "$RANK" && "$WORLD_SIZE" == 2 ]] || {
    echo "Docker actor identity differs from outer pair identity" >&2; exit 2;
}
[[ "$RANK_SCRIPT" == "$ROOT/scripts/round_dagger/run_actor_cohort.sh" || \
   "$RANK_SCRIPT" == "$ROOT/scripts/round_dagger/run_actor_cohort_with_wandb.sh" ]] || {
    echo "Docker executor received an unknown actor primitive" >&2; exit 2;
}
[ -f "$RANK_SCRIPT" ] && [ ! -L "$RANK_SCRIPT" ] || {
    echo "unsafe or missing Docker actor primitive" >&2; exit 2;
}

DOCKER=${ROUND_DAGGER_DOCKER_BIN:-$(command -v docker || true)}
IMAGE=${ROUND_DAGGER_RTX_IMAGE:?missing ROUND_DAGGER_RTX_IMAGE}
IPC_MODE=${ROUND_DAGGER_ACTOR_DOCKER_IPC:-private}
SHM_SIZE=${ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE:-32g}
GPU_IDS=${ROUND_DAGGER_ACTOR_DOCKER_GPU_IDS:-0,1}
MOUNT_ROOTS=${ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS:-/mnt/data1,/mnt/data2}

case "$DOCKER" in /*) ;; *) echo "Docker executable must be absolute" >&2; exit 2 ;; esac
[ -x "$DOCKER" ] && [ ! -L "$DOCKER" ] || {
    echo "Docker executable is unsafe or unavailable" >&2; exit 2;
}
# Never share the host IPC namespace.  Two containers can have the same inner
# PID, and Carb names shared-memory objects using that PID.  --ipc=host lets
# those objects collide and can crash both plugin loaders before Isaac starts.
[[ "$IPC_MODE" == private ]] || {
    echo "actor containers require private IPC; host IPC is forbidden" >&2; exit 2;
}
[[ "$SHM_SIZE" =~ ^([1-9][0-9]*)(g|G)$ ]] || {
    echo "actor Docker shm size must be an integral number of GiB" >&2; exit 2;
}
SHM_GIB=${BASH_REMATCH[1]}
[[ "$SHM_GIB" -ge 16 ]] || {
    echo "actor Docker shm size must be at least 16 GiB" >&2; exit 2;
}
[[ "$GPU_IDS" =~ ^([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || {
    echo "actor Docker GPU IDs must be two distinct comma-separated integers" >&2; exit 2;
}
GPU_MAP=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
GPU=${GPU_MAP[$RANK]}

IFS=, read -r -a RAW_MOUNTS <<<"$MOUNT_ROOTS"
[[ ${#RAW_MOUNTS[@]} -ge 1 ]] || { echo "no Docker mount roots" >&2; exit 2; }
MOUNTS=()
for mount_root in "${RAW_MOUNTS[@]}"; do
    case "$mount_root" in /*) ;; *) echo "Docker mount roots must be absolute" >&2; exit 2 ;; esac
    [ -d "$mount_root" ] && [ ! -L "$mount_root" ] || {
        echo "unsafe or missing Docker mount root: $mount_root" >&2; exit 2;
    }
    MOUNTS+=(--mount "type=bind,src=$mount_root,dst=$mount_root")
done

CAMPAIGN=${ROUND_DAGGER_CAMPAIGN_ID:?missing ROUND_DAGGER_CAMPAIGN_ID}
MACRO_ROUND=${ROUND_DAGGER_MACRO_ROUND:?missing ROUND_DAGGER_MACRO_ROUND}
COHORT_INDEX=${ROUND_DAGGER_COHORT_INDEX:?missing ROUND_DAGGER_COHORT_INDEX}
[[ "$CAMPAIGN" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "campaign ID is unsafe for a Docker container name" >&2; exit 2;
}
[[ "$MACRO_ROUND" =~ ^(0|[1-9][0-9]*)$ && "$COHORT_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "round/cohort identity must use canonical integers" >&2; exit 2;
}
CONTAINER="rd-${CAMPAIGN}-m${MACRO_ROUND}-c${COHORT_INDEX}-r${RANK}"

# Forward names, never values, on Docker argv.  This keeps W&B credentials out
# of process listings while preserving the exact common telemetry directory.
ENV_ARGS=()
while IFS= read -r name; do
    case "$name" in
        WANDB_API_KEY) ;;
        # The production container receives a literal legacy selection below.
        # Never forward adapter/AB selectors, plans, evidence, or any future
        # variables in those namespaces from the operator shell.
        ROUND_DAGGER_RUNTIME_ADAPTER_*|ROUND_DAGGER_RUNTIME_AB_*) ;;
        ROUND_DAGGER_*|IMAGE3F_*|IMAGE_DAGGER_*|WANDB_*|ROBORENDER_*|\
        GR00T_ROOT|GR00T_PYTHON|ANCHORS|TRAIN_RUN_DIR|PYTHONPATH|NETRC|DRY_RUN|ROLLOUT_STEPS)
            ENV_ARGS+=(--env "$name")
            ;;
    esac
done < <(compgen -e | LC_ALL=C sort)
# Clear a possible image-level acknowledgement as well as rejecting ambient
# host values.  Shadow validation has its own launcher and cannot enter this
# production executor accidentally.
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
