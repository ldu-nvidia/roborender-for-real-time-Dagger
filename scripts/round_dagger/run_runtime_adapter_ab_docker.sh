#!/usr/bin/env bash
# Sequential, validation-only legacy/shadow actor A/B executor.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 1 || $# == 2 ]] || {
    echo "usage: run_runtime_adapter_ab_docker.sh ABSOLUTE_PLAN_JSON [legacy|shadow|all]" >&2
    exit 2
}
PLAN=$1
SELECTION=${2:-all}
case "$PLAN:$SELECTION" in
    /*:legacy|/*:shadow|/*:all) ;;
    *) echo "runtime A/B plan must be absolute and selection must be legacy, shadow, or all" >&2; exit 2 ;;
esac
[[ -f $PLAN && ! -L $PLAN ]] || { echo "unsafe or missing runtime A/B plan" >&2; exit 2; }
[[ ${ROUND_DAGGER_RUNTIME_AB_LAUNCH_ACK:-} == bounded-runtime-adapter-ab-v1 ]] || {
    echo "runtime A/B launch requires ROUND_DAGGER_RUNTIME_AB_LAUNCH_ACK=bounded-runtime-adapter-ab-v1" >&2
    exit 2
}

PLAN_SHA256=$(sha256sum "$PLAN" | awk '{print $1}')
REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    -m round_dagger.runtime_ab validate --plan "$PLAN")
BOUND_REPOSITORY_ROOT=$(python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(d["source"]["repository_root"])' \
    <<<"$REPORT")
[[ $BOUND_REPOSITORY_ROOT == "$ROOT" ]] || {
    echo "runtime A/B plan is bound to a different repository root" >&2
    exit 2
}

DOCKER=${ROUND_DAGGER_DOCKER_BIN:-$(command -v docker || true)}
IPC_MODE=${ROUND_DAGGER_ACTOR_DOCKER_IPC:-private}
SHM_SIZE=${ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE:-32g}
MOUNT_ROOTS=${ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS:-/mnt/data1,/mnt/data2}
case "$DOCKER" in /*) ;; *) echo "Docker executable must be absolute" >&2; exit 2 ;; esac
[[ -x $DOCKER && ! -L $DOCKER ]] || { echo "Docker executable is unsafe or unavailable" >&2; exit 2; }
[[ $IPC_MODE == private ]] || { echo "runtime A/B requires private Docker IPC" >&2; exit 2; }
[[ $SHM_SIZE =~ ^([1-9][0-9]*)(g|G)$ ]] || { echo "runtime A/B shm size must be integral GiB" >&2; exit 2; }
((BASH_REMATCH[1] >= 16)) || { echo "runtime A/B shm size must be at least 16 GiB" >&2; exit 2; }

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
    # A more-specific read-only bind shadows any broad writable /mnt/data*
    # parent mount. Container root must not mutate the attested source tree.
    MOUNTS+=(--mount "type=bind,src=$ROOT,dst=$ROOT,readonly")
fi

read_warmup_report() {
    local payload=$1
    mapfile -d '' -t WARMUP_FIELDS < <(
        python3 -c '
import json
import sys

document = json.load(sys.stdin)
for name in (
    "state", "endpoint", "anchor", "output", "renderer_sha256",
    "stream_id", "seed", "reference_lag", "actor_image",
    "actor_image_id_sha256", "gr00t_python",
):
    value = document[name]
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise SystemExit(f"malformed warm-up field: {name}")
    sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")
' <<<"$payload"
    )
    [[ ${#WARMUP_FIELDS[@]} == 11 ]] || {
        echo "runtime A/B warm-up report is incomplete" >&2
        exit 2
    }
    WARMUP_STATE=${WARMUP_FIELDS[0]}
    WARMUP_ENDPOINT=${WARMUP_FIELDS[1]}
    WARMUP_ANCHOR=${WARMUP_FIELDS[2]}
    WARMUP_OUTPUT=${WARMUP_FIELDS[3]}
    WARMUP_RENDERER_SHA256=${WARMUP_FIELDS[4]}
    WARMUP_STREAM_ID=${WARMUP_FIELDS[5]}
    WARMUP_SEED=${WARMUP_FIELDS[6]}
    WARMUP_REFERENCE_LAG=${WARMUP_FIELDS[7]}
    ACTOR_IMAGE=${WARMUP_FIELDS[8]}
    ACTOR_IMAGE_ID_SHA256=${WARMUP_FIELDS[9]}
    WARMUP_PYTHON=${WARMUP_FIELDS[10]}
}

IMAGE_VERIFIED=0
verify_immutable_image() {
    if ((IMAGE_VERIFIED)); then
        return
    fi
    [[ $ACTOR_IMAGE_ID_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
        echo "runtime A/B actor image ID in plan is malformed" >&2
        exit 2
    }
    IMMUTABLE_IMAGE="sha256:$ACTOR_IMAGE_ID_SHA256"
    IMAGE_ID=$($DOCKER image inspect "$ACTOR_IMAGE" --format '{{.Id}}')
    [[ $IMAGE_ID == "$IMMUTABLE_IMAGE" ]] || {
        echo "runtime A/B actor image ID differs from immutable plan" >&2
        exit 2
    }
    IMAGE_VERIFIED=1
}

LEGACY_STATE=$(python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(d["arms"]["legacy"]["state"])' \
    <<<"$REPORT")
SHADOW_STATE=$(python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(d["arms"]["shadow"]["state"])' \
    <<<"$REPORT")
if [[ $SELECTION == shadow && $LEGACY_STATE != sealed ]]; then
    echo "runtime A/B shadow requires a sealed legacy arm" >&2
    exit 2
fi

WARMUP_REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    -m round_dagger.runtime_ab warmup --plan "$PLAN")
read_warmup_report "$WARMUP_REPORT"
[[ $WARMUP_SEED == 42 && $WARMUP_REFERENCE_LAG == 256 ]] || {
    echo "runtime A/B warm-up differs from the fixed seed/lag contract" >&2
    exit 2
}
case "$WARMUP_ENDPOINT:$WARMUP_ANCHOR:$WARMUP_OUTPUT:$WARMUP_PYTHON" in
    /*:/*:/*:/*) ;;
    *) echo "runtime A/B warm-up paths must be absolute" >&2; exit 2 ;;
esac

case "$WARMUP_STATE" in
    pending)
        [[ $LEGACY_STATE == empty && $SHADOW_STATE == empty && \
           $SELECTION != shadow ]] || {
            echo "runtime A/B warm-up may only be created before an empty legacy arm" >&2
            exit 2
        }
        verify_immutable_image
        "$DOCKER" run --rm \
            --name "rd-runtime-ab-warmup-${PLAN_SHA256:0:12}" \
            --ipc=private \
            --shm-size="$SHM_SIZE" \
            --network=host \
            --workdir "$ROOT" \
            "${MOUNTS[@]}" \
            --env "PYTHONDONTWRITEBYTECODE=1" \
            --env "PYTHONPATH=$ROOT:$ROOT/GR00T-VisualSim2Real" \
            "$IMMUTABLE_IMAGE" \
            "$WARMUP_PYTHON" -m scripts.round_dagger.warm_image3f_renderer \
            --endpoint "$WARMUP_ENDPOINT" \
            --anchor "$WARMUP_ANCHOR" \
            --output "$WARMUP_OUTPUT" \
            --renderer-sha256 "$WARMUP_RENDERER_SHA256" \
            --stream-id "$WARMUP_STREAM_ID"
        WARMUP_REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
            -m round_dagger.runtime_ab warmup --plan "$PLAN")
        read_warmup_report "$WARMUP_REPORT"
        ;;
    validated) ;;
    *) echo "runtime A/B warm-up has an unknown state: $WARMUP_STATE" >&2; exit 2 ;;
esac
[[ $WARMUP_STATE == validated ]] || {
    echo "runtime A/B warm-up did not publish validated immutable evidence" >&2
    exit 2
}

if [[ $SELECTION == all ]]; then
    ARMS=(legacy shadow)
else
    ARMS=("$SELECTION")
fi

for ARM in "${ARMS[@]}"; do
    REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_ab validate --plan "$PLAN")
    STATE=$(python3 -c \
        'import json,sys; d=json.load(sys.stdin); print(d["arms"][sys.argv[1]]["state"])' \
        "$ARM" <<<"$REPORT")
    if [[ $ARM == shadow ]]; then
        LEGACY_STATE=$(python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(d["arms"]["legacy"]["state"])' \
            <<<"$REPORT")
        [[ $LEGACY_STATE == sealed ]] || {
            echo "runtime A/B shadow requires a sealed legacy arm" >&2
            exit 2
        }
    fi
    if [[ $STATE == sealed ]]; then
        echo "RUNTIME_ADAPTER_AB_ALREADY_SEALED arm=$ARM"
        continue
    fi
    [[ $STATE == empty ]] || {
        echo "runtime A/B arm $ARM contains immutable partial evidence; refusing overwrite" >&2
        exit 2
    }

    ENV_NAMES=()
    while IFS= read -r -d '' name && IFS= read -r -d '' value; do
        [[ $name =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "unsafe runtime A/B environment name" >&2; exit 2; }
        printf -v "$name" '%s' "$value"
        export "$name"
        ENV_NAMES+=("$name")
    done < <(
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
            -m round_dagger.runtime_ab environment \
            --plan "$PLAN" --arm "$ARM" --null
    )

    [[ $ROUND_DAGGER_REPOSITORY_ROOT == "$ROOT" && \
       $ROUND_DAGGER_RUNTIME_AB_PLAN == "$PLAN" && \
       $ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256 == "$PLAN_SHA256" && \
       $ROUND_DAGGER_RUNTIME_AB_ARM == "$ARM" && \
       $ROUND_DAGGER_RTX_IMAGE == "$ACTOR_IMAGE" && \
       $ROUND_DAGGER_RTX_IMAGE_ID_SHA256 == "$ACTOR_IMAGE_ID_SHA256" && \
       $ROUND_DAGGER_VALIDATION_RANKS == 1 && \
       $ROUND_DAGGER_VALIDATION_ENVS_PER_RANK == 1 && \
       $ROUND_DAGGER_VALIDATION_STEPS == 2 && \
       $ROUND_DAGGER_VALIDATION_PURPOSE == train && \
       $ROUND_DAGGER_VALIDATION_TRAIN_ACK == bounded-train-replay && \
       $ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK == paired-1x2-v1 && \
       $ROUND_DAGGER_SIMULATOR_SEED == 0 && \
       $ROUND_DAGGER_IMAGE3F_SEED == 42 ]] || {
        echo "runtime A/B environment differs from the exact paired 1x1x2 contract" >&2
        exit 2
    }
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_ab authorize \
        --plan "$PLAN" --arm "$ARM" \
        --expected-plan-sha256 "$PLAN_SHA256" >/dev/null

    verify_immutable_image
    GPU=$ROUND_DAGGER_ACTOR_DOCKER_GPU_ID
    [[ $GPU =~ ^(0|[1-9][0-9]*)$ ]] || { echo "runtime A/B GPU ID is malformed" >&2; exit 2; }

    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
    unset MASTER_ADDR MASTER_PORT ROUND_DAGGER_ACTOR_RANK ROUND_DAGGER_ACTOR_WORLD_SIZE
    unset ROUND_DAGGER_VALIDATION_ACTOR_RANK
    ENV_ARGS=()
    for name in "${ENV_NAMES[@]}"; do
        ENV_ARGS+=(--env "$name")
    done
    CONTAINER="rd-runtime-ab-${ARM}-${PLAN_SHA256:0:12}"
    "$DOCKER" run --rm \
        --name "$CONTAINER" \
        --gpus "\"device=$GPU\"" \
        --ipc=private \
        --shm-size="$SHM_SIZE" \
        --network=host \
        --workdir "$ROOT" \
        "${MOUNTS[@]}" \
        --env "PYTHONDONTWRITEBYTECODE=1" \
        "${ENV_ARGS[@]}" \
        "$IMMUTABLE_IMAGE" \
        /usr/bin/env bash "$ROOT/scripts/round_dagger/run_actor_validation.sh"

    POST_REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_ab validate --plan "$PLAN")
    POST_STATE=$(python3 -c \
        'import json,sys; d=json.load(sys.stdin); print(d["arms"][sys.argv[1]]["state"])' \
        "$ARM" <<<"$POST_REPORT")
    [[ $POST_STATE == sealed ]] || {
        echo "runtime A/B arm $ARM exited without publishing a sealed cohort" >&2
        exit 2
    }
done

if [[ $SELECTION == all ]]; then
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.runtime_ab compare --plan "$PLAN"
fi
