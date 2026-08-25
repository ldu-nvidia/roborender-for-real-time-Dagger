#!/usr/bin/env bash
# Launch one bounded validation cohort as two independent single-GPU actors.
#
# This is intentionally separate from run_actor_cohort_pair.sh.  Production is
# fixed to 2 x 32 x 300 train transitions; this launcher derives a bounded
# 2 x (1..32) x (1..300) eval or acknowledged-train shape from contract.json.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
DRY_RUN=${DRY_RUN:-0}
COHORT_DIR=${ROUND_DAGGER_COHORT_DIR:?missing ROUND_DAGGER_COHORT_DIR}
TRAIN_DIR=${TRAIN_RUN_DIR:?missing TRAIN_RUN_DIR}
RANK_SCRIPT=$ROOT/scripts/round_dagger/run_actor_validation.sh
if [[ -n ${ROUND_DAGGER_VALIDATION_PAIR_EXECUTOR:-} ]]; then
    EXECUTOR=$ROUND_DAGGER_VALIDATION_PAIR_EXECUTOR
elif [[ $DRY_RUN == 1 ]]; then
    # The rank-local launcher has its own fail-closed dry-run and does not
    # import Isaac or touch a GPU.
    EXECUTOR=
else
    EXECUTOR=$ROOT/scripts/round_dagger/run_actor_validation_rank_docker.sh
fi
STAGGER_SECONDS=${ROUND_DAGGER_VALIDATION_PAIR_STAGGER_SECONDS:-10}

case "$COHORT_DIR:$TRAIN_DIR:$RANK_SCRIPT" in
    /*:/*:/*) ;;
    *) echo "validation pair paths must be absolute" >&2; exit 2 ;;
esac
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || {
    echo "DRY_RUN must be exactly 0 or 1" >&2; exit 2;
}
[[ $STAGGER_SECONDS =~ ^(0|[1-9][0-9]?)$ ]] || {
    echo "validation pair stagger must be an integer from 0 through 99 seconds" >&2
    exit 2
}
[[ -f $RANK_SCRIPT && ! -L $RANK_SCRIPT ]] || {
    echo "unsafe or missing validation rank-local actor script" >&2; exit 2;
}
[[ $(realpath -m -- "$TRAIN_DIR") == "$TRAIN_DIR" ]] || {
    echo "TRAIN_RUN_DIR must be a canonical absolute path" >&2; exit 2;
}
if [[ -n $EXECUTOR ]]; then
    case "$EXECUTOR" in /*) ;; *) echo "validation pair executor must be absolute" >&2; exit 2 ;; esac
    [[ -x $EXECUTOR && ! -L $EXECUTOR ]] || {
        echo "validation pair executor is unsafe or not executable" >&2; exit 2;
    }
elif [[ $DRY_RUN != 1 ]]; then
    echo "non-dry validation pair requires an isolation executor" >&2
    exit 2
fi

# The immutable cohort is the sole authority for the rung shape.  Ambient
# rank/env/step variables are overwritten before either child is launched.
CONTRACT_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m round_dagger.validation_pair contract --cohort-dir "$COHORT_DIR")
read -r PURPOSE ENVS_PER_RANK STEPS GLOBAL_ENV_OFFSET EXPECTED_TRANSITIONS < <(
    python3 -c \
        'import json,sys; d=json.load(sys.stdin); print(d["purpose"], d["environments_per_rank"], d["steps_per_environment"], d["global_env_offset"], d["expected_transitions"])' \
        <<<"$CONTRACT_JSON"
)
if [[ -n ${ROUND_DAGGER_VALIDATION_PURPOSE:-} && \
      $ROUND_DAGGER_VALIDATION_PURPOSE != "$PURPOSE" ]]; then
    echo "requested validation purpose differs from immutable cohort contract" >&2
    exit 2
fi
if [[ $PURPOSE == train && \
      ${ROUND_DAGGER_VALIDATION_TRAIN_ACK:-} != bounded-train-replay ]]; then
    echo "bounded train validation requires ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay" >&2
    exit 2
fi

export ROUND_DAGGER_VALIDATION_RANKS=2
export ROUND_DAGGER_VALIDATION_ENVS_PER_RANK=$ENVS_PER_RANK
export ROUND_DAGGER_VALIDATION_STEPS=$STEPS
export ROUND_DAGGER_VALIDATION_PURPOSE=$PURPOSE
export ROUND_DAGGER_GLOBAL_ENV_OFFSET=$GLOBAL_ENV_OFFSET

mkdir -p "$TRAIN_DIR/logs" "$TRAIN_DIR/validation_pair_attempts"
[[ ! -L $TRAIN_DIR && ! -L $TRAIN_DIR/logs && \
   ! -L $TRAIN_DIR/validation_pair_attempts ]] || {
    echo "validation pair run directories must not be symbolic links" >&2; exit 2;
}
ATTEMPT_DIR="$TRAIN_DIR/validation_pair_attempts/attempt_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$ATTEMPT_DIR"

STATUS_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m round_dagger.validation_pair status --cohort-dir "$COHORT_DIR")
read -r SEALED STATE0 STATE1 < <(
    python3 -c \
        'import json,sys; d=json.load(sys.stdin); print(int(d["sealed"]), d["rank_states"]["0"], d["rank_states"]["1"])' \
        <<<"$STATUS_JSON"
)
if [[ $SEALED == 1 ]]; then
    printf '%s\n' "$STATUS_JSON"
    echo "ACTOR_VALIDATION_PAIR_ALREADY_SEALED"
    exit 0
fi
for rank_state in "$STATE0" "$STATE1"; do
    [[ $rank_state == empty || $rank_state == complete ]] || {
        echo "validation pair contains a partial rank; refusing to overwrite immutable evidence" >&2
        exit 2
    }
done

launch_rank() {
    local rank=$1
    local log=$2
    (
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
        unset MASTER_ADDR MASTER_PORT
        export ROUND_DAGGER_VALIDATION_ACTOR_RANK=$rank
        # The executor also binds this external identity before entering the
        # container; run_actor_validation.sh reasserts it inside the process.
        export ROUND_DAGGER_ACTOR_RANK=$rank
        export ROUND_DAGGER_ACTOR_WORLD_SIZE=2
        if [[ -n $EXECUTOR ]]; then
            exec "$EXECUTOR" "$rank" "$RANK_SCRIPT"
        fi
        exec /usr/bin/env bash "$RANK_SCRIPT"
    ) >"$log" 2>&1 &
    LAUNCHED_PID=$!
}

PIDS=("" "")
LOGS=("$ATTEMPT_DIR/rank_0000.log" "$ATTEMPT_DIR/rank_0001.log")
STATES=("$STATE0" "$STATE1")
for rank in 0 1; do
    if [[ ${STATES[$rank]} == complete ]]; then
        echo "validation actor rank $rank is already exactly complete; skipping relaunch"
        continue
    fi
    launch_rank "$rank" "${LOGS[$rank]}"
    PIDS[$rank]=$LAUNCHED_PID
    if [[ $rank == 0 && $DRY_RUN != 1 && $STATE1 == empty && \
          $STAGGER_SECONDS -gt 0 ]]; then
        sleep "$STAGGER_SECONDS"
    fi
done

# Reap every launched child.  One actor failing never causes this launcher to
# signal, abandon, or hide the other actor's immutable evidence.
STATUSES=(0 0)
set +e
for rank in 0 1; do
    if [[ -n ${PIDS[$rank]} ]]; then
        wait "${PIDS[$rank]}"
        STATUSES[$rank]=$?
    fi
done
set -e

if [[ $DRY_RUN == 1 ]]; then
    for rank in 0 1; do
        [[ ! -f ${LOGS[$rank]} ]] || {
            echo "--- dry-run logical validation rank $rank ---"
            sed -n '1,80p' "${LOGS[$rank]}"
        }
    done
fi
if [[ ${STATUSES[0]} -ne 0 || ${STATUSES[1]} -ne 0 ]]; then
    echo "validation actor pair failed: rank0=${STATUSES[0]} rank1=${STATUSES[1]}; cohort was not sealed" >&2
    echo "validation actor pair logs: ${LOGS[0]} ${LOGS[1]}" >&2
    exit 2
fi
if [[ $DRY_RUN == 1 ]]; then
    echo "DRY_RUN_ACTOR_VALIDATION_PAIR_OK ranks=0,1 envs_per_rank=$ENVS_PER_RANK steps=$STEPS purpose=$PURPOSE transitions=$EXPECTED_TRANSITIONS private_actor_processes=1"
    exit 0
fi

# A trainer can emit a fatal traceback yet return zero.  Independently require
# each exact immutable rank payload here, even though the rank-local primitive
# performs the same gate, before publishing the sole cohort seal.
for rank in 0 1; do
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
        -m round_dagger.validation_pair rank \
        --cohort-dir "$COHORT_DIR" --rank "$rank" >/dev/null
done
SEALED_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m round_dagger.validation_pair seal --cohort-dir "$COHORT_DIR")
printf '%s\n' "$SEALED_JSON"
echo "ACTOR_VALIDATION_PAIR_SEALED purpose=$PURPOSE transitions=$EXPECTED_TRANSITIONS logs=${LOGS[0]},${LOGS[1]}"
