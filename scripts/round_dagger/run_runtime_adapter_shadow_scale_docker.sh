#!/usr/bin/env bash
# Sequential one-rank, two-step metadata-only shadow scale ladder.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 1 || $# == 2 ]] || {
    echo "usage: run_runtime_adapter_shadow_scale_docker.sh ABSOLUTE_PLAN_JSON [2|4|8|16|32|all]" >&2
    exit 2
}
PLAN=$1
SELECTION=${2:-all}
case "$PLAN:$SELECTION" in
    /*:2|/*:4|/*:8|/*:16|/*:32|/*:all) ;;
    *) echo "scale plan must be absolute and selection must be 2,4,8,16,32,or all" >&2; exit 2 ;;
esac
[[ -f $PLAN && ! -L $PLAN ]] || { echo "unsafe or missing scale plan" >&2; exit 2; }
[[ ${ROUND_DAGGER_RUNTIME_SCALE_LAUNCH_ACK:-} == bounded-runtime-adapter-shadow-scale-v1 ]] || {
    echo "scale launch requires ROUND_DAGGER_RUNTIME_SCALE_LAUNCH_ACK=bounded-runtime-adapter-shadow-scale-v1" >&2
    exit 2
}

PLAN_SHA256=$(sha256sum "$PLAN" | awk '{print $1}')
REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    -m round_dagger.runtime_scale validate --plan "$PLAN")
BOUND_ROOT=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["source"]["repository_root"])' <<<"$REPORT")
[[ $BOUND_ROOT == "$ROOT" ]] || {
    echo "runtime scale plan is bound to a different repository root" >&2
    exit 2
}
[[ ! -L $ROOT && $(stat -c '%a' "$ROOT") == 555 ]] || {
    echo "runtime scale requires an immutable 0555 sealed repository root" >&2
    exit 2
}

DOCKER=${ROUND_DAGGER_DOCKER_BIN:-$(command -v docker || true)}
IPC_MODE=${ROUND_DAGGER_ACTOR_DOCKER_IPC:-private}
SHM_SIZE=${ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE:-32g}
MOUNT_ROOTS=${ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS:-/mnt/data1,/mnt/data2}
case "$DOCKER" in /*) ;; *) echo "Docker executable must be absolute" >&2; exit 2 ;; esac
[[ -x $DOCKER && ! -L $DOCKER ]] || { echo "Docker executable is unsafe or unavailable" >&2; exit 2; }
[[ $IPC_MODE == private ]] || { echo "runtime scale requires private Docker IPC" >&2; exit 2; }
[[ $SHM_SIZE =~ ^([1-9][0-9]*)(g|G)$ ]] || { echo "scale shm size must be integral GiB" >&2; exit 2; }
((BASH_REMATCH[1] >= 16)) || { echo "scale shm size must be at least 16 GiB" >&2; exit 2; }

IFS=, read -r -a RAW_MOUNTS <<<"$MOUNT_ROOTS"
MOUNTS=()
ROOT_MOUNTED_EXACT=0
for mount_root in "${RAW_MOUNTS[@]}"; do
    case "$mount_root" in /*) ;; *) echo "Docker mount roots must be absolute" >&2; exit 2 ;; esac
    [[ -d $mount_root && ! -L $mount_root ]] || { echo "unsafe Docker mount root: $mount_root" >&2; exit 2; }
    if [[ $mount_root == "$ROOT" ]]; then
        MOUNTS+=(--mount "type=bind,src=$mount_root,dst=$mount_root,readonly")
        ROOT_MOUNTED_EXACT=1
    else
        MOUNTS+=(--mount "type=bind,src=$mount_root,dst=$mount_root")
    fi
done
if ((!ROOT_MOUNTED_EXACT)); then
    MOUNTS+=(--mount "type=bind,src=$ROOT,dst=$ROOT,readonly")
fi

ACTOR_IMAGE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["actor_image"])' <<<"$REPORT")
ACTOR_IMAGE_ID_SHA256=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["actor_image_id_sha256"])' <<<"$REPORT")
[[ $ACTOR_IMAGE_ID_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "malformed actor image identity" >&2; exit 2; }
IMMUTABLE_IMAGE="sha256:$ACTOR_IMAGE_ID_SHA256"
IMAGE_ID=$($DOCKER image inspect "$ACTOR_IMAGE" --format '{{.Id}}')
[[ $IMAGE_ID == "$IMMUTABLE_IMAGE" ]] || {
    echo "runtime scale actor image ID differs from immutable plan" >&2
    exit 2
}

SCALE_COMPONENT_LABEL=runtime-adapter-shadow-scale
ENV_FILE=
ACTIVE_ENV_NAMES=()
CURRENT_CONTAINER=
CURRENT_CONTAINER_ACTIVE=0
CURRENT_RUNG=

inspect_scale_container() {
    "$DOCKER" container inspect "$1" --format \
        '{{.Id}}|{{.State.Status}}|{{index .Config.Labels "ai.openai.round-dagger.component"}}|{{index .Config.Labels "ai.openai.round-dagger.plan-sha256"}}|{{index .Config.Labels "ai.openai.round-dagger.rung"}}|{{index .Config.Labels "ai.openai.round-dagger.launcher-pid"}}' \
        2>/dev/null
}

cleanup_scale_launcher() {
    status=$?
    trap - EXIT INT TERM HUP
    if [[ -n $ENV_FILE ]]; then
        rm -f -- "$ENV_FILE"
        ENV_FILE=
    fi
    for name in "${ACTIVE_ENV_NAMES[@]}"; do
        unset "$name" || true
    done
    if ((CURRENT_CONTAINER_ACTIVE)) && [[ -n $CURRENT_CONTAINER ]]; then
        metadata=$(inspect_scale_container "$CURRENT_CONTAINER" || true)
        IFS='|' read -r container_id container_status component plan_sha rung launcher_pid \
            <<<"$metadata"
        if [[ -n $container_id && \
              $component == "$SCALE_COMPONENT_LABEL" && \
              $plan_sha == "$PLAN_SHA256" && \
              $rung == "$CURRENT_RUNG" && \
              $launcher_pid == "$$" ]]; then
            echo "cooperatively stopping launcher-owned scale container $CURRENT_CONTAINER after interruption" >&2
            "$DOCKER" stop --signal=SIGTERM --timeout=30 "$container_id" >/dev/null 2>&1 || {
                echo "launcher-owned scale container did not stop cleanly: $CURRENT_CONTAINER" >&2
            }
        else
            echo "scale container ownership changed; refusing cleanup: name=$CURRENT_CONTAINER metadata=${metadata:-unavailable}" >&2
        fi
    fi
    exit "$status"
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap cleanup_scale_launcher EXIT

if [[ $SELECTION == all ]]; then RUNGS=(2 4 8 16 32); else RUNGS=("$SELECTION"); fi
for RUNG in "${RUNGS[@]}"; do
    REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_scale validate --plan "$PLAN")
    STATE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["rungs"][sys.argv[1]]["state"])' "$RUNG" <<<"$REPORT")
    if [[ $STATE == sealed ]]; then
        echo "RUNTIME_ADAPTER_SCALE_ALREADY_SEALED envs=$RUNG"
        continue
    fi
    if [[ $STATE == complete ]]; then
        echo "RUNTIME_ADAPTER_SCALE_RESUME_SEAL envs=$RUNG"
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
            -m round_dagger.runtime_scale seal --plan "$PLAN" --rung "$RUNG" >/dev/null
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
            -m round_dagger.runtime_scale verify --plan "$PLAN" --rung "$RUNG"
        continue
    fi
    [[ $STATE == empty ]] || {
        echo "runtime scale rung $RUNG contains immutable partial evidence; refusing overwrite" >&2
        exit 2
    }

    for name in "${ACTIVE_ENV_NAMES[@]}"; do
        unset "$name"
    done
    ACTIVE_ENV_NAMES=()
    ENV_NAMES=()
    ENV_FILE=$(mktemp "$(dirname "$PLAN")/.runtime-scale-env-e${RUNG}.XXXXXX")
    chmod 600 "$ENV_FILE"
    if ! PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_scale environment \
        --plan "$PLAN" --rung "$RUNG" --null >"$ENV_FILE"; then
        echo "failed to derive immutable runtime scale environment for rung $RUNG" >&2
        exit 2
    fi
    ENV_FIELDS=()
    mapfile -d '' -t ENV_FIELDS <"$ENV_FILE"
    rm -f -- "$ENV_FILE"
    ENV_FILE=
    ((${#ENV_FIELDS[@]} > 0 && ${#ENV_FIELDS[@]} % 2 == 0)) || {
        echo "runtime scale environment stream is empty or truncated" >&2
        exit 2
    }
    declare -A SEEN_ENV=()
    for ((index = 0; index < ${#ENV_FIELDS[@]}; index += 2)); do
        name=${ENV_FIELDS[index]}
        value=${ENV_FIELDS[index + 1]}
        [[ $name =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "unsafe scale environment name" >&2; exit 2; }
        [[ -z ${SEEN_ENV[$name]+present} ]] || { echo "duplicate scale environment name: $name" >&2; exit 2; }
        SEEN_ENV[$name]=1
        printf -v "$name" '%s' "$value"
        export "$name"
        ENV_NAMES+=("$name")
        ACTIVE_ENV_NAMES+=("$name")
    done
    unset SEEN_ENV

    [[ $ROUND_DAGGER_RUNTIME_SCALE_PLAN == "$PLAN" && \
       $ROUND_DAGGER_RUNTIME_SCALE_PLAN_SHA256 == "$PLAN_SHA256" && \
       $ROUND_DAGGER_RUNTIME_ADAPTER_MODE == shadow && \
       $ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK == metadata-only-v1 && \
       $ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK == shadow-scale-2step-v1 && \
       $ROUND_DAGGER_VALIDATION_RANKS == 1 && \
       $ROUND_DAGGER_VALIDATION_ENVS_PER_RANK == "$RUNG" && \
       $ROUND_DAGGER_VALIDATION_STEPS == 2 && \
       $ROUND_DAGGER_VALIDATION_PURPOSE == train && \
       $ROUND_DAGGER_VALIDATION_TRAIN_ACK == bounded-train-replay && \
       $ROUND_DAGGER_VALIDATION_ACTOR_RANK == 0 && \
       $ROUND_DAGGER_RUNTIME_SCALE_EVIDENCE == "$ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE" ]] || {
        echo "runtime scale environment differs from bounded shadow contract" >&2
        exit 2
    }
    case "$ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE" in
        live-isaac-v1) ;;
        canonical-zero-robot-mask-deterministic-cuda-v1)
            [[ ${CUBLAS_WORKSPACE_CONFIG:-} == :4096:8 ]] || {
                echo "deterministic runtime scale requires CUBLAS_WORKSPACE_CONFIG=:4096:8" >&2
                exit 2
            }
            ;;
        *) echo "runtime scale input profile is outside bounded contract" >&2; exit 2 ;;
    esac
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_scale authorize \
        --plan "$PLAN" \
        --expected-plan-sha256 "$PLAN_SHA256" \
        --expected-rung-contract-sha256 "$ROUND_DAGGER_RUNTIME_SCALE_RUNG_CONTRACT_SHA256" \
        >/dev/null

    GPU=$ROUND_DAGGER_ACTOR_DOCKER_GPU_ID
    [[ $GPU =~ ^(0|[1-9][0-9]*)$ ]] || { echo "runtime scale GPU ID is malformed" >&2; exit 2; }
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
    unset ROUND_DAGGER_ACTOR_RANK ROUND_DAGGER_ACTOR_WORLD_SIZE
    ENV_ARGS=()
    for name in "${ENV_NAMES[@]}"; do ENV_ARGS+=(--env "$name"); done
    CONTAINER="rd-runtime-scale-e${RUNG}-${PLAN_SHA256:0:12}"
    if metadata=$(inspect_scale_container "$CONTAINER"); then
        echo "runtime scale container name already exists; refusing to touch it: name=$CONTAINER metadata=$metadata" >&2
        exit 2
    fi
    CURRENT_CONTAINER=$CONTAINER
    CURRENT_RUNG=$RUNG
    CURRENT_CONTAINER_ACTIVE=1
    "$DOCKER" run --rm \
        --name "$CONTAINER" \
        --label "ai.openai.round-dagger.component=$SCALE_COMPONENT_LABEL" \
        --label "ai.openai.round-dagger.plan-sha256=$PLAN_SHA256" \
        --label "ai.openai.round-dagger.rung=$RUNG" \
        --label "ai.openai.round-dagger.launcher-pid=$$" \
        --gpus "\"device=$GPU\"" \
        --ipc=private \
        --shm-size="$SHM_SIZE" \
        --network=host \
        --workdir "$ROOT" \
        "${MOUNTS[@]}" \
        --env "PYTHONDONTWRITEBYTECODE=1" \
        "${ENV_ARGS[@]}" \
        "$IMMUTABLE_IMAGE" \
        /usr/bin/env bash "$ROOT/scripts/round_dagger/run_actor_validation.sh" &
    DOCKER_CLIENT_PID=$!
    if wait "$DOCKER_CLIENT_PID"; then
        docker_status=0
    else
        docker_status=$?
    fi
    CURRENT_CONTAINER_ACTIVE=0
    CURRENT_CONTAINER=
    CURRENT_RUNG=
    ((docker_status == 0)) || {
        echo "runtime scale actor container failed for rung $RUNG with status $docker_status" >&2
        exit "$docker_status"
    }

    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_scale seal --plan "$PLAN" --rung "$RUNG" >/dev/null
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_scale verify --plan "$PLAN" --rung "$RUNG"
done
