#!/usr/bin/env bash
# Collect one exact production cohort with two independent single-GPU actors.
#
# Non-dry runs default to run_actor_rank_docker.sh; an alternate absolute
# ROUND_DAGGER_ACTOR_PAIR_EXECUTOR can supply cluster-specific isolation.  The
# executor is invoked as:
#
#   EXECUTOR <logical-rank: 0|1> <absolute-rank-local-script>
#
# It must place that process in a distinct single-GPU execution context (for
# example one Docker container per physical GPU), preserve the inherited
# artifact/telemetry mounts, and exec the supplied rank-local script.  It must
# not configure torchrun, Accelerate multi-GPU, or NCCL.  This control-plane
# script itself never signals either actor: it waits for both and fails closed.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DRY_RUN=${DRY_RUN:-0}
COHORT_DIR=${ROUND_DAGGER_COHORT_DIR:?missing ROUND_DAGGER_COHORT_DIR}
TRAIN_DIR=${TRAIN_RUN_DIR:?missing TRAIN_RUN_DIR}
ACTOR_RUN_DIR=${ROUND_DAGGER_ACTOR_RUN_DIR:?missing ROUND_DAGGER_ACTOR_RUN_DIR}
ENDPOINT_MANIFEST=${IMAGE3F_ENDPOINT_MANIFEST:?missing IMAGE3F_ENDPOINT_MANIFEST}
RENDERER_SHA256=${ROUND_DAGGER_RENDERER_SHA256:?missing ROUND_DAGGER_RENDERER_SHA256}
EXPECTED_RENDERERS=${ROUND_DAGGER_EXPECTED_RENDERERS:-2}
LIFECYCLE_MODULE=round_dagger.image3f_lifecycle
RANK_SCRIPT=${ROUND_DAGGER_ACTOR_PAIR_RANK_SCRIPT:-$ROOT/scripts/round_dagger/run_actor_cohort_with_wandb.sh}
if [ -n "${ROUND_DAGGER_ACTOR_PAIR_EXECUTOR:-}" ]; then
    EXECUTOR=$ROUND_DAGGER_ACTOR_PAIR_EXECUTOR
elif [ "$DRY_RUN" = 1 ]; then
    # Unit/preflight validation invokes both rank-local scripts directly; they
    # print their commands and never import Isaac or allocate a GPU.
    EXECUTOR=
else
    EXECUTOR=$ROOT/scripts/round_dagger/run_actor_rank_docker.sh
fi
STAGGER_SECONDS=${ROUND_DAGGER_ACTOR_PAIR_STAGGER_SECONDS:-10}
WANDB_ENABLED=${ROUND_DAGGER_WANDB_ENABLED:-0}
case "${WANDB_ENABLED,,}" in
    0|false|no) WANDB_REQUESTED=0; WANDB_PAIR=0 ;;
    1|true|yes)
        WANDB_REQUESTED=1
        if [[ "$RANK_SCRIPT" != "$ROOT/scripts/round_dagger/run_actor_cohort_with_wandb.sh" ]]; then
            echo "W&B-enabled actor pair requires the repository W&B rank primitive" >&2
            exit 2
        elif [ "$DRY_RUN" != 1 ]; then
            WANDB_PAIR=1
        else
            WANDB_PAIR=0
        fi
        ;;
    *) echo "ROUND_DAGGER_WANDB_ENABLED must be boolean" >&2; exit 2 ;;
esac

case "$COHORT_DIR:$TRAIN_DIR:$ACTOR_RUN_DIR:$RANK_SCRIPT:$ENDPOINT_MANIFEST" in
    /*:/*:/*:/*:/*) ;;
    *) echo "actor pair paths must be absolute" >&2; exit 2 ;;
esac
[[ "$TRAIN_DIR" == "$ACTOR_RUN_DIR" ]] || {
    echo "ROUND_DAGGER_ACTOR_RUN_DIR must equal TRAIN_RUN_DIR" >&2; exit 2;
}
[[ "$RANK_SCRIPT" == "$ROOT/scripts/round_dagger/run_actor_cohort.sh" || \
   "$RANK_SCRIPT" == "$ROOT/scripts/round_dagger/run_actor_cohort_with_wandb.sh" ]] || {
    echo "actor pair must use a repository rank-local actor primitive" >&2; exit 2;
}
[ -f "$RANK_SCRIPT" ] && [ ! -L "$RANK_SCRIPT" ] || {
    echo "unsafe or missing rank-local actor script" >&2; exit 2;
}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
    echo "DRY_RUN must be exactly 0 or 1" >&2; exit 2;
}
[[ "$STAGGER_SECONDS" =~ ^(0|[1-9][0-9]?)$ ]] || {
    echo "actor pair stagger must be an integer from 0 through 99 seconds" >&2; exit 2;
}
[[ ${ROUND_DAGGER_LOGICAL_ENV_COUNT:-} == 64 ]] || {
    echo "actor pair requires exactly 64 logical environments" >&2; exit 2;
}
[[ ${ROUND_DAGGER_STEPS_PER_ENVIRONMENT:-} == 512 ]] || {
    echo "actor pair requires exactly 512 steps per environment" >&2; exit 2;
}
[[ ${ROUND_DAGGER_EXPECTED_TRANSITIONS:-} == 32768 ]] || {
    echo "actor pair requires exactly 32768 transitions" >&2; exit 2;
}
[[ "$EXPECTED_RENDERERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "actor pair expected renderer count must be positive" >&2; exit 2;
}
[[ "$RENDERER_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "actor pair renderer SHA-256 is malformed" >&2; exit 2;
}
[ -f "$ENDPOINT_MANIFEST" ] && [ ! -L "$ENDPOINT_MANIFEST" ] || {
    echo "actor pair endpoint manifest is unsafe or missing" >&2; exit 2;
}

if [ -n "$EXECUTOR" ]; then
    case "$EXECUTOR" in /*) ;; *) echo "actor pair executor must be absolute" >&2; exit 2 ;; esac
    [ -x "$EXECUTOR" ] && [ ! -L "$EXECUTOR" ] || {
        echo "actor pair executor is unsafe or not executable" >&2; exit 2;
    }
elif [ "$DRY_RUN" != 1 ]; then
    echo "non-dry actor pair requires ROUND_DAGGER_ACTOR_PAIR_EXECUTOR" >&2
    exit 2
fi

mkdir -p "$TRAIN_DIR/logs" "$TRAIN_DIR/actor_pair_attempts"
[ ! -L "$TRAIN_DIR" ] && [ ! -L "$TRAIN_DIR/logs" ] && \
    [ ! -L "$TRAIN_DIR/actor_pair_attempts" ] || {
        echo "actor pair run directories must not be symbolic links" >&2; exit 2;
    }

# Serialize lifecycle ownership for this cohort.  If the outer launcher dies,
# a surviving rank retains the descriptor until it exits; a retry therefore
# cannot overlap and claim the same immutable shard or stream ownership.
# Lock the already-validated directory inode itself.  This avoids creating or
# following a mutable lock-file symlink in a cross-container run directory.
exec {PAIR_LOCK_FD}<"$TRAIN_DIR"
flock -n "$PAIR_LOCK_FD" || {
    echo "another actor-pair attempt is still active for this cohort" >&2; exit 2;
}

STATUS_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m round_dagger.actor_pair status --cohort-dir "$COHORT_DIR")
read -r SEALED STATE0 STATE1 < <(python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(int(d["sealed"]), d["rank_states"]["0"], d["rank_states"]["1"])' \
    <<<"$STATUS_JSON")

for rank_state in "$STATE0" "$STATE1"; do
    [[ "$rank_state" == empty || "$rank_state" == complete ]] || {
        echo "actor pair contains a partial rank; refusing to overwrite immutable evidence" >&2
        exit 2
    }
done

COMPLETED_RANKS=()
[ "$STATE0" != complete ] || COMPLETED_RANKS+=(0)
[ "$STATE1" != complete ] || COMPLETED_RANKS+=(1)
COMPLETED_RANKS_CSV=$(IFS=,; printf '%s' "${COMPLETED_RANKS[*]}")
# Reconcile every older plan, even when neither rank has committed a shard.
# Otherwise two supervisor deaths before failure cleanup can leave multiple
# unreceipted owners and make a later shard-to-receipt recovery ambiguous.
if ! PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m "$LIFECYCLE_MODULE" recover-ownership \
        --train-dir "$TRAIN_DIR" --cohort-dir "$COHORT_DIR" \
        --endpoint-manifest "$ENDPOINT_MANIFEST" \
        --expected-renderers "$EXPECTED_RENDERERS" \
        --renderer-sha256 "$RENDERER_SHA256" \
        --completed-ranks "$COMPLETED_RANKS_CSV" >/dev/null; then
    echo "actor pair could not reconcile Image3F stream ownership" >&2
    exit 2
fi
# A unique namespace prevents collision, but it does not expand renderer
# capacity.  Reset every fully reaped prior plan and require durable ACKs
# before a retry can allocate another 64 streams.
if [ "$DRY_RUN" != 1 ] && ! PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m "$LIFECYCLE_MODULE" cleanup-prior \
        --train-dir "$TRAIN_DIR" \
        --timeout-s "${ROUND_DAGGER_IMAGE3F_RESET_TIMEOUT_S:-10}" \
        --retries "${ROUND_DAGGER_IMAGE3F_RESET_RETRIES:-3}" >/dev/null; then
    echo "actor pair could not reset and acknowledge prior Image3F streams" >&2
    exit 2
fi

if [ "$SEALED" = 1 ]; then
    if [ "$DRY_RUN" != 1 ]; then
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m "$LIFECYCLE_MODULE" cleanup \
            --train-dir "$TRAIN_DIR" --cohort-dir "$COHORT_DIR" \
            --timeout-s "${ROUND_DAGGER_IMAGE3F_RESET_TIMEOUT_S:-10}" \
            --retries "${ROUND_DAGGER_IMAGE3F_RESET_RETRIES:-3}"
    fi
    printf '%s\n' "$STATUS_JSON"
    echo "ACTOR_PAIR_ALREADY_SEALED_STREAMS_CLEAN"
    exit 0
fi

LAUNCH_RANKS=()
for rank in 0 1; do
    if [ "${rank}" = 0 ] && [ "$STATE0" = empty ]; then
        LAUNCH_RANKS+=(0)
    elif [ "${rank}" = 1 ] && [ "$STATE1" = empty ]; then
        LAUNCH_RANKS+=(1)
    fi
done
if [ "${#LAUNCH_RANKS[@]}" -eq 0 ]; then
    if [ "$DRY_RUN" = 1 ]; then
        printf '%s\n' "$STATUS_JSON"
        echo "DRY_RUN_ACTOR_PAIR_ALREADY_COMPLETE no_cleanup=1 no_seal=1"
        exit 0
    fi
    if [ "$WANDB_REQUESTED" = 1 ]; then
        echo "unsealed W&B cohort has no incomplete actor available to own telemetry recovery" >&2
        exit 2
    fi
    CLEANUP_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m "$LIFECYCLE_MODULE" cleanup \
            --train-dir "$TRAIN_DIR" --cohort-dir "$COHORT_DIR" \
            --timeout-s "${ROUND_DAGGER_IMAGE3F_RESET_TIMEOUT_S:-10}" \
            --retries "${ROUND_DAGGER_IMAGE3F_RESET_RETRIES:-3}")
    printf '%s\n' "$CLEANUP_JSON"
    SEALED_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m round_dagger.actor_pair seal --cohort-dir "$COHORT_DIR")
    printf '%s\n' "$SEALED_JSON"
    echo "ACTOR_PAIR_RECOVERED_COMPLETE_STREAMS_CLEAN"
    exit 0
fi

# mktemp gives every physical retry a unique directory even when the same PID
# is rapidly recycled.  The namespace is then deterministically derived from
# this immutable attempt path by the lifecycle module.
ATTEMPT_DIR=$(mktemp -d \
    "$TRAIN_DIR/actor_pair_attempts/attempt_$(date -u +%Y%m%dT%H%M%SZ)_$$_XXXXXXXX")
ATTEMPT_PLAN="$ATTEMPT_DIR/image3f_stream_plan.json"
LAUNCH_RANKS_CSV=$(IFS=,; printf '%s' "${LAUNCH_RANKS[*]}")
if ! PLAN_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m "$LIFECYCLE_MODULE" plan \
        --output "$ATTEMPT_PLAN" \
        --campaign-id "${ROUND_DAGGER_CAMPAIGN_ID:?missing ROUND_DAGGER_CAMPAIGN_ID}" \
        --macro-round "${ROUND_DAGGER_MACRO_ROUND:?missing ROUND_DAGGER_MACRO_ROUND}" \
        --cohort-index "${ROUND_DAGGER_COHORT_INDEX:?missing ROUND_DAGGER_COHORT_INDEX}" \
        --attempt-dir "$ATTEMPT_DIR" --cohort-dir "$COHORT_DIR" \
        --endpoint-manifest "$ENDPOINT_MANIFEST" \
        --expected-renderers "$EXPECTED_RENDERERS" \
        --renderer-sha256 "$RENDERER_SHA256" \
        --local-environments 32 --actor-world-size 2 \
        --launched-ranks "$LAUNCH_RANKS_CSV"); then
    echo "actor pair could not publish an immutable Image3F stream plan" >&2
    exit 2
fi
ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE=$(python3 -c \
    'import json,sys; print(json.load(sys.stdin)["attempt_namespace"])' <<<"$PLAN_JSON")
export ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE
export ROUND_DAGGER_IMAGE3F_STREAM_PLAN="$ATTEMPT_PLAN"
echo "ACTOR_PAIR_STREAM_PLAN namespace=$ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE plan=$ATTEMPT_PLAN"

# A W&B owner waits for the peer to finish before draining shared telemetry.
# Give it a cross-container lease so SIGKILL of this outer supervisor cannot
# strand the owner/watcher (and the retry lock) forever.
PAIR_HEARTBEAT="$ATTEMPT_DIR/pair_supervisor_heartbeat"
: >"$PAIR_HEARTBEAT"
[ -f "$PAIR_HEARTBEAT" ] && [ ! -L "$PAIR_HEARTBEAT" ] || {
    echo "actor pair heartbeat path is unsafe" >&2; exit 2;
}
PAIR_SUPERVISOR_PID=$BASHPID
PAIR_SUPERVISOR_START=$(awk '{print $22}' "/proc/$PAIR_SUPERVISOR_PID/stat")
(
    exec {PAIR_LOCK_FD}>&-
    while [ -r "/proc/$PAIR_SUPERVISOR_PID/stat" ] && \
          [ "$(awk '{print $22}' "/proc/$PAIR_SUPERVISOR_PID/stat" 2>/dev/null)" = \
            "$PAIR_SUPERVISOR_START" ]; do
        touch -- "$PAIR_HEARTBEAT"
        sleep 2
    done
) &
PAIR_HEARTBEAT_PID=$!
stop_pair_heartbeat() {
    kill "$PAIR_HEARTBEAT_PID" 2>/dev/null || true
    wait "$PAIR_HEARTBEAT_PID" 2>/dev/null || true
}
trap stop_pair_heartbeat EXIT
export ROUND_DAGGER_WANDB_PAIR_HEARTBEAT="$PAIR_HEARTBEAT"
export ROUND_DAGGER_WANDB_PAIR_ORPHAN_TIMEOUT_SECONDS=${ROUND_DAGGER_WANDB_PAIR_ORPHAN_TIMEOUT_SECONDS:-30}

# The W&B run ID and upload state live in the common run directory, so a retry
# resumes the same remote run.  Its sole sidecar must, however, belong to a
# rank that this attempt actually launches.  Prefer rank zero on a fresh
# attempt and recover on rank one when rank zero is already immutable.
WANDB_OWNER=
WANDB_ACTOR_DONE=
WANDB_PAIR_FINISH_RELATIVE=
WANDB_PAIR_FINISH=
if [ "$WANDB_PAIR" = 1 ]; then
    if [ "$STATE0" = empty ]; then
        WANDB_OWNER=0
    elif [ "$STATE1" = empty ]; then
        WANDB_OWNER=1
    else
        echo "unsealed W&B cohort has no incomplete actor available to own telemetry recovery" >&2
        exit 2
    fi
    printf -v WANDB_OWNER_PADDED '%04d' "$WANDB_OWNER"
    WANDB_ACTOR_DONE="$ATTEMPT_DIR/rank_${WANDB_OWNER_PADDED}_actor_done"
    WANDB_PAIR_FINISH_RELATIVE=${ATTEMPT_DIR#"$TRAIN_DIR"/}/wandb_pair_finish
    WANDB_PAIR_FINISH="$TRAIN_DIR/$WANDB_PAIR_FINISH_RELATIVE"
fi

launch_rank() {
    local rank=$1
    local log=$2
    (
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
        unset MASTER_ADDR MASTER_PORT
        export ROUND_DAGGER_ACTOR_RANK="$rank"
        export ROUND_DAGGER_ACTOR_WORLD_SIZE=2
        if [ "$WANDB_PAIR" = 1 ] && [ "$rank" = "$WANDB_OWNER" ]; then
            export ROUND_DAGGER_WANDB_SIDECAR_OWNER=1
            export ROUND_DAGGER_WANDB_PAIR_LIFECYCLE=1
            export ROUND_DAGGER_WANDB_ACTOR_DONE_MARKER="$WANDB_ACTOR_DONE"
            export ROUND_DAGGER_WANDB_PAIR_FINISH_MARKER="$WANDB_PAIR_FINISH_RELATIVE"
        else
            if [ "$WANDB_REQUESTED" = 1 ] && \
               [ "$RANK_SCRIPT" = "$ROOT/scripts/round_dagger/run_actor_cohort_with_wandb.sh" ]; then
                export ROUND_DAGGER_WANDB_SIDECAR_OWNER=0
            else
                unset ROUND_DAGGER_WANDB_SIDECAR_OWNER
            fi
            unset ROUND_DAGGER_WANDB_PAIR_LIFECYCLE
            unset ROUND_DAGGER_WANDB_ACTOR_DONE_MARKER
            unset ROUND_DAGGER_WANDB_PAIR_FINISH_MARKER
        fi
        if [ -n "$EXECUTOR" ]; then
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
    if [ "${STATES[$rank]}" = complete ]; then
        echo "actor rank $rank is already exactly complete; skipping relaunch"
        continue
    fi
    launch_rank "$rank" "${LOGS[$rank]}"
    PIDS[$rank]=$LAUNCHED_PID
    # Private IPC is the hard isolation boundary.  A small stagger additionally
    # avoids making both Isaac/Carb plugin loaders and GPU contexts peak at the
    # exact same instant.  Dry-run validation is intentionally immediate.
    if [ "$rank" = 0 ] && [ "$DRY_RUN" != 1 ] && [ "$STATE1" = empty ] && \
       [ "$STAGGER_SECONDS" -gt 0 ]; then
        sleep "$STAGGER_SECONDS"
    fi
done

# Always reap both actor processes.  A failure on one rank never causes this
# launcher to signal or abandon the other rank.
STATUSES=(0 0)
set +e
if [ "$WANDB_PAIR" = 1 ]; then
    OWNER_PID=${PIDS[$WANDB_OWNER]}
    # Reap every launched non-owner without signaling it.  The owner publishes
    # ACTOR_DONE after its trainer and exact-rank gate finish, then remains
    # alive until the outer pair authorizes a final stability-window drain.
    for rank in 0 1; do
        if [ "$rank" != "$WANDB_OWNER" ] && [ -n "${PIDS[$rank]}" ]; then
            wait "${PIDS[$rank]}"
            STATUSES[$rank]=$?
        fi
    done
    while [ ! -f "$WANDB_ACTOR_DONE" ] && kill -0 "$OWNER_PID" 2>/dev/null; do
        sleep 1
    done
    if [ -f "$WANDB_ACTOR_DONE" ] && [ ! -L "$WANDB_ACTOR_DONE" ]; then
        : >"$WANDB_PAIR_FINISH"
    fi
    wait "$OWNER_PID"
    STATUSES[$WANDB_OWNER]=$?
else
    for rank in 0 1; do
        if [ -n "${PIDS[$rank]}" ]; then
            wait "${PIDS[$rank]}"
            STATUSES[$rank]=$?
        fi
    done
fi
set -e

if [ "$DRY_RUN" = 1 ]; then
    for rank in 0 1; do
        [ ! -f "${LOGS[$rank]}" ] || {
            echo "--- dry-run logical actor rank $rank ---"
            sed -n '1,80p' "${LOGS[$rank]}"
        }
    done
fi

# The rank-local primitive normally publishes this receipt immediately after
# its exact immutable shard gate.  Repeating publication here closes the
# custom-executor/test path and is byte-idempotent.
LIFECYCLE_STATUS=0
for rank in 0 1; do
    if [ "$DRY_RUN" != 1 ] && [ -n "${PIDS[$rank]}" ] && \
       [ "${STATUSES[$rank]}" -eq 0 ]; then
        printf -v COMPLETION_PATH '%s/rank_%04d_stream_completion.json' "$ATTEMPT_DIR" "$rank"
        if ! PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -m "$LIFECYCLE_MODULE" complete-rank \
                --output "$COMPLETION_PATH" --plan "$ATTEMPT_PLAN" \
                --cohort-dir "$COHORT_DIR" --rank "$rank" >/dev/null; then
            echo "actor rank $rank exited zero without valid stream ownership evidence" >&2
            LIFECYCLE_STATUS=1
        fi
    fi
done

if [ "${STATUSES[0]}" -ne 0 ] || [ "${STATUSES[1]}" -ne 0 ] || \
   [ "$LIFECYCLE_STATUS" -ne 0 ]; then
    if [ "$DRY_RUN" != 1 ] && ! PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m "$LIFECYCLE_MODULE" cleanup-attempt \
            --train-dir "$TRAIN_DIR" --plan "$ATTEMPT_PLAN" \
            --timeout-s "${ROUND_DAGGER_IMAGE3F_RESET_TIMEOUT_S:-10}" \
            --retries "${ROUND_DAGGER_IMAGE3F_RESET_RETRIES:-3}" >/dev/null; then
        echo "failed actor attempt also left unacknowledged stream cleanup" >&2
    fi
    echo "actor pair failed: rank0=${STATUSES[0]} rank1=${STATUSES[1]}; cohort was not sealed" >&2
    echo "actor pair logs: ${LOGS[0]} ${LOGS[1]}" >&2
    exit 2
fi
if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN_ACTOR_PAIR_OK ranks=0,1 world_size=2 no_torch_distributed=1"
    exit 0
fi

CLEANUP_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m "$LIFECYCLE_MODULE" cleanup \
        --train-dir "$TRAIN_DIR" --cohort-dir "$COHORT_DIR" \
        --timeout-s "${ROUND_DAGGER_IMAGE3F_RESET_TIMEOUT_S:-10}" \
        --retries "${ROUND_DAGGER_IMAGE3F_RESET_RETRIES:-3}")
printf '%s\n' "$CLEANUP_JSON"
SEALED_JSON=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m round_dagger.actor_pair seal --cohort-dir "$COHORT_DIR")
printf '%s\n' "$SEALED_JSON"
echo "ACTOR_PAIR_SEALED_STREAMS_CLEAN logs=${LOGS[0]},${LOGS[1]}"
