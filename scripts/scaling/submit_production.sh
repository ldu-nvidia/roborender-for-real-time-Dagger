#!/bin/bash
# Submit the renderer fleet and the distributed 256-env x 300-step DAgger job.
# Set DRY_RUN=1 to print the exact sbatch commands without changing Slurm state.
set -Eeuo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}
TRAINER_NODES=${TRAINER_NODES:-8}
MIX=${MIX:-0.0}
REFERENCE_LAG=${REFERENCE_LAG:-256}
ENVS_PER_RENDERER=${ENVS_PER_RENDERER:-5}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
if [ -n "${NUM_TOTAL_BATCHES:-}" ] && [ -n "${NUM_ROLLOUT_ITERATIONS:-}" ] && \
   [ "$NUM_TOTAL_BATCHES" != "$NUM_ROLLOUT_ITERATIONS" ]; then
    echo "FATAL: NUM_TOTAL_BATCHES and NUM_ROLLOUT_ITERATIONS disagree"
    exit 2
fi
NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-${NUM_TOTAL_BATCHES:-1}}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}
BEHAVIOR_SOURCE=${BEHAVIOR_SOURCE:-student}
TEACHER_ROLLOUT_RATIO=${TEACHER_ROLLOUT_RATIO:-0.0}
VG_PRIMARY_VIEW=${VG_PRIMARY_VIEW:-head_color}
RUN_WALLTIME=${RUN_WALLTIME:-08:00:00}
RENDER_WALLTIME=${RENDER_WALLTIME:-24:00:00}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}
FIXED_EVAL_SEEDS=${FIXED_EVAL_SEEDS:-1729,3253,4519,5987,7411,8863,10193,12203}
FIXED_EVAL_NUM_ENVS=${FIXED_EVAL_NUM_ENVS:-32}
FIXED_EVAL_MAX_PARALLEL=${FIXED_EVAL_MAX_PARALLEL:-4}
FIXED_EVAL_MIN_SUCCESS_DELTA=${FIXED_EVAL_MIN_SUCCESS_DELTA:-0.02}
ENABLE_FIXED_EVAL_GATE=${ENABLE_FIXED_EVAL_GATE:-auto}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_PYTHON=${WANDB_PYTHON:-/svl/u/ravenh/miniconda3/envs/viral/bin/python}
EXPECTED_IMAGE3F_RELEASE_ID=${EXPECTED_IMAGE3F_RELEASE_ID:-roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0}
EXPECTED_IMAGE3F_ADAPTER_SHA256=${EXPECTED_IMAGE3F_ADAPTER_SHA256:-c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588}
EXPECTED_IMAGE3F_INFERENCE_STEPS=${EXPECTED_IMAGE3F_INFERENCE_STEPS:-5}
EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=${EXPECTED_IMAGE3F_TEACACHE_THRESHOLD:-0.3}
EXPECTED_IMAGE3F_CFG_SCALE=${EXPECTED_IMAGE3F_CFG_SCALE:-1.0}
EXPECTED_IMAGE3F_CUDA_GRAPH=${EXPECTED_IMAGE3F_CUDA_GRAPH:-disabled}
PROJ=${PROJ:-roborender_dagger_scale}
EXP=${EXP:-dagger_${GLOBAL_NUM_ENVS}env_${ROLLOUT_STEPS}step_mix${MIX}}
RUN_ID=${RUN_ID:-scale_$(date +%Y%m%d_%H%M%S)}
FLEET_DIR=${FLEET_DIR:-$ROOT/runtime/$RUN_ID}
DRY_RUN=${DRY_RUN:-0}
EXPECTED_FAST_ROBORENDER_REVISION=${EXPECTED_FAST_ROBORENDER_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}
EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-$(git -C "$GR00T_ROOT" rev-parse HEAD)}
CKPT=${CKPT:-$GR00T_ROOT/logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/last.pt}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$GR00T_ROOT/logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447/config.yaml}

python3 - \
    "$EXPECTED_IMAGE3F_RELEASE_ID" \
    "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    "$EXPECTED_IMAGE3F_INFERENCE_STEPS" \
    "$EXPECTED_IMAGE3F_TEACACHE_THRESHOLD" \
    "$EXPECTED_IMAGE3F_CFG_SCALE" \
    "$EXPECTED_IMAGE3F_CUDA_GRAPH" <<'PY'
import math
import re
import sys

release_id, adapter_sha256, steps_text, tea_text, cfg_text, cuda_graph = sys.argv[1:]
if not release_id.strip():
    raise SystemExit("FATAL: EXPECTED_IMAGE3F_RELEASE_ID must be non-empty")
if not re.fullmatch(r"[0-9a-f]{64}", adapter_sha256):
    raise SystemExit("FATAL: EXPECTED_IMAGE3F_ADAPTER_SHA256 must be 64 lowercase hex characters")
try:
    steps = int(steps_text)
    tea = float(tea_text)
    cfg = float(cfg_text)
except ValueError as exc:
    raise SystemExit(f"FATAL: malformed Image3F generation contract: {exc}")
if steps < 1 or not math.isfinite(tea) or tea < 0 or not math.isfinite(cfg) or cfg < 0:
    raise SystemExit("FATAL: invalid Image3F steps, TeaCache threshold, or CFG scale")
if cuda_graph not in {"enabled", "disabled"}:
    raise SystemExit("FATAL: EXPECTED_IMAGE3F_CUDA_GRAPH must be enabled or disabled")
PY

read -r FIXED_EVAL_SEEDS_JSON FIXED_EVAL_SEEDS_COLON FIXED_EVAL_SEED_COUNT < <(
    python3 - "$FIXED_EVAL_SEEDS" "$FIXED_EVAL_NUM_ENVS" "$FIXED_EVAL_MIN_SUCCESS_DELTA" "$FIXED_EVAL_MAX_PARALLEL" "$SAVE_FREQUENCY" <<'PY'
import json
import sys

try:
    seeds = [int(piece.strip()) for piece in sys.argv[1].split(",") if piece.strip()]
    num_envs = int(sys.argv[2])
    delta = float(sys.argv[3])
    max_parallel = int(sys.argv[4])
    save_frequency = int(sys.argv[5])
except ValueError as exc:
    raise SystemExit(f"FATAL: invalid fixed-seed evaluation setting: {exc}")
if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
    raise SystemExit("FATAL: FIXED_EVAL_SEEDS needs at least two unique non-negative integers")
if num_envs < 1:
    raise SystemExit("FATAL: FIXED_EVAL_NUM_ENVS must be positive")
if not 0.0 <= delta <= 1.0:
    raise SystemExit("FATAL: FIXED_EVAL_MIN_SUCCESS_DELTA must be in [0, 1]")
if not 1 <= max_parallel <= 4:
    raise SystemExit("FATAL: FIXED_EVAL_MAX_PARALLEL must be in [1, 4]")
if save_frequency < 1:
    raise SystemExit("FATAL: SAVE_FREQUENCY must be positive")
print(json.dumps(seeds, separators=(",", ":")), ":".join(map(str, seeds)), len(seeds))
PY
)
FIXED_EVAL_EPISODES_PER_CHECKPOINT=$((FIXED_EVAL_SEED_COUNT * FIXED_EVAL_NUM_ENVS))
[[ "$NUM_ROLLOUT_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: NUM_ROLLOUT_ITERATIONS must be a positive integer"
    exit 2
}
if [ "$ENABLE_FIXED_EVAL_GATE" = "auto" ]; then
    if [ "$NUM_ROLLOUT_ITERATIONS" -ge $((3 * SAVE_FREQUENCY)) ]; then
        ENABLE_FIXED_EVAL_GATE=1
    else
        ENABLE_FIXED_EVAL_GATE=0
    fi
fi
case "$ENABLE_FIXED_EVAL_GATE" in
    0)
        EVALUATION_ENABLED_JSON=false
        EVALUATION_DISABLED_REASON="fewer than three configured checkpoint saves"
        ;;
    1)
        EVALUATION_ENABLED_JSON=true
        EVALUATION_DISABLED_REASON=""
        ;;
    *) echo "FATAL: ENABLE_FIXED_EVAL_GATE must be auto, 0, or 1"; exit 2 ;;
esac
BASELINE_CKPT_SHA256=DISABLED
REFERENCE_CONFIG_SHA256=DISABLED

FAST_DIRTY=$(git -C "$ROOT" status --porcelain)
if [ -n "$FAST_DIRTY" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN WARNING: fast_roborender is dirty; an actual submission will refuse it."
    elif [ "${ALLOW_DIRTY_FAST_ROBORENDER:-0}" != "1" ]; then
        echo "FATAL: commit the scaling branch before production submission."
        exit 2
    fi
fi
if [ "$DRY_RUN" != "1" ]; then
    [ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 2; }
    BASELINE_CKPT_SHA256=$(sha256sum "$CKPT" | awk '{print $1}')
    if [ "$ENABLE_FIXED_EVAL_GATE" = "1" ]; then
        [ -f "$REFERENCE_CONFIG" ] || {
            echo "FATAL: fixed-seed reference config not found: $REFERENCE_CONFIG"
            exit 2
        }
        REFERENCE_CONFIG_SHA256=$(sha256sum "$REFERENCE_CONFIG" | awk '{print $1}')
    fi
    INSTALL_CHECK=(python3 "$ROOT/scripts/scaling/install_gr00t_integration.py"
        --gr00t-root "$GR00T_ROOT" --expected-revision "$EXPECTED_GR00T_REVISION")
    if [ "${ALLOW_DIRTY_GR00T:-0}" = "1" ]; then
        INSTALL_CHECK+=(--allow-dirty)
    fi
    "${INSTALL_CHECK[@]}"
    [ -x "$WANDB_PYTHON" ] || {
        echo "FATAL: trainer W&B Python is not executable: $WANDB_PYTHON"
        exit 2
    }
    WANDB_PREFLIGHT=("$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py"
        --expected-email "$WANDB_EXPECTED_EMAIL" --format shell)
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_PREFLIGHT+=(--entity "$WANDB_ENTITY")
    fi
    WANDB_IDENTITY=$(WANDB_MODE="$WANDB_MODE" "${WANDB_PREFLIGHT[@]}")
    case "$WANDB_IDENTITY" in
        WANDB_ENTITY=?*) WANDB_ENTITY=${WANDB_IDENTITY#WANDB_ENTITY=} ;;
        *) echo "FATAL: W&B preflight did not return a personal entity"; exit 2 ;;
    esac
else
    WANDB_ENTITY=${WANDB_ENTITY:-DRY_RUN_UNVERIFIED}
    BASELINE_CKPT_SHA256=DRY_RUN_UNVERIFIED
    if [ "$ENABLE_FIXED_EVAL_GATE" = "1" ]; then
        REFERENCE_CONFIG_SHA256=DRY_RUN_UNVERIFIED
    fi
fi

PLAN=(
    --global-envs "$GLOBAL_NUM_ENVS"
    --rollout-steps "$ROLLOUT_STEPS"
    --num-rollout-iterations "$NUM_ROLLOUT_ITERATIONS"
    --trainer-nodes "$TRAINER_NODES"
    --raytraced-mix "$MIX"
    --envs-per-renderer "$ENVS_PER_RENDERER"
    --reference-lag "$REFERENCE_LAG"
    --num-mini-batches "$NUM_MINI_BATCHES"
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --num-learning-epochs "$NUM_LEARNING_EPOCHS"
    --behavior-source "$BEHAVIOR_SOURCE"
    --teacher-rollout-ratio "$TEACHER_ROLLOUT_RATIO"
    --primary-view "$VG_PRIMARY_VIEW"
)
RENDERERS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field required_renderers)
LOCAL_ENVS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field local_envs)

mkdir -p "$FLEET_DIR/endpoints" "$FLEET_DIR/queues" "$ROOT/logs_archive"
python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --format json \
    > "$FLEET_DIR/submission_plan.json"

RENDER_JOB_ID=""
TRAIN_JOB_ID=""
EVAL_ORCHESTRATOR_JOB_ID=""
EVAL_ARRAY_JOB_ID=""
EVAL_REPORT_JOB_ID=""
RENDER_CLEANUP_WATCHDOG_JOB_ID=""
TRAIN_RUN_DIR_FOR_JOB=""
SUBMISSION_OWNS_RENDERER=0
SUBMISSION_OWNS_TRAINER=0
SUBMISSION_OWNS_WATCHDOG=0
FAILURE_STATUS=launcher_failed
if [ "$ENABLE_FIXED_EVAL_GATE" = "1" ]; then
    EVALUATION_SUBMISSION_STATUS=not_submitted
else
    EVALUATION_SUBMISSION_STATUS=disabled_short_run
fi

write_job_manifest() {
    local submission_status=$1
    local renderer_json=null
    local trainer_json=null
    local eval_orchestrator_json=null
    local eval_array_json=null
    local eval_report_json=null
    local cleanup_watchdog_json=null
    local train_run_dir_json=null
    local target="$FLEET_DIR/slurm_jobs.json"
    local temporary="${target}.tmp.$$"
    if [ -n "$RENDER_JOB_ID" ]; then
        renderer_json="\"$RENDER_JOB_ID\""
    fi
    if [ -n "$TRAIN_JOB_ID" ]; then
        trainer_json="\"$TRAIN_JOB_ID\""
    fi
    if [ -n "$EVAL_ORCHESTRATOR_JOB_ID" ]; then
        eval_orchestrator_json="\"$EVAL_ORCHESTRATOR_JOB_ID\""
    fi
    if [ -n "$EVAL_ARRAY_JOB_ID" ]; then
        eval_array_json="\"$EVAL_ARRAY_JOB_ID\""
    fi
    if [ -n "$EVAL_REPORT_JOB_ID" ]; then
        eval_report_json="\"$EVAL_REPORT_JOB_ID\""
    fi
    if [ -n "$RENDER_CLEANUP_WATCHDOG_JOB_ID" ]; then
        cleanup_watchdog_json="\"$RENDER_CLEANUP_WATCHDOG_JOB_ID\""
    fi
    if [ -n "$TRAIN_RUN_DIR_FOR_JOB" ]; then
        train_run_dir_json="\"$TRAIN_RUN_DIR_FOR_JOB\""
    fi
    cat > "$temporary" <<EOF
{
  "run_id": "$RUN_ID",
  "submission_status": "$submission_status",
  "renderer_array_job_id": $renderer_json,
  "trainer_job_id": $trainer_json,
  "eval_orchestrator_job_id": $eval_orchestrator_json,
  "eval_array_job_id": $eval_array_json,
  "eval_report_job_id": $eval_report_json,
  "renderer_cleanup_watchdog_job_id": $cleanup_watchdog_json,
  "train_run_dir": $train_run_dir_json,
  "renderer_count": $RENDERERS,
  "trainer_nodes": $TRAINER_NODES,
  "global_environments": $GLOBAL_NUM_ENVS,
  "local_environments": $LOCAL_ENVS,
  "rollout_steps": $ROLLOUT_STEPS,
  "walltime": "$RUN_WALLTIME",
  "trainer_walltime": "$RUN_WALLTIME",
  "renderer_walltime": "$RENDER_WALLTIME",
  "pre_timeout_signal": "B:USR1@300",
  "rollout_iterations": $NUM_ROLLOUT_ITERATIONS,
  "trainer_num_total_batches": $NUM_ROLLOUT_ITERATIONS,
  "num_learning_epochs": $NUM_LEARNING_EPOCHS,
  "behavior_source": "$BEHAVIOR_SOURCE",
  "rollout_with_teacher_num_steps": 0,
  "teacher_rollout_ratio": $TEACHER_ROLLOUT_RATIO,
  "value_loss_coef": 0.0,
  "entropy_coef": 0.0,
  "primary_view": "$VG_PRIMARY_VIEW",
  "checkpoint_frequency_rollout_iterations": $SAVE_FREQUENCY,
  "wandb_expected_email": "$WANDB_EXPECTED_EMAIL",
  "wandb_entity": "$WANDB_ENTITY",
  "wandb_project": "$PROJ",
  "renderer_contract": {
    "release_id": "$EXPECTED_IMAGE3F_RELEASE_ID",
    "adapter_sha256": "$EXPECTED_IMAGE3F_ADAPTER_SHA256",
    "inference_steps": $EXPECTED_IMAGE3F_INFERENCE_STEPS,
    "teacache_threshold": $EXPECTED_IMAGE3F_TEACACHE_THRESHOLD,
    "cfg_scale": $EXPECTED_IMAGE3F_CFG_SCALE,
    "cuda_graph": "$EXPECTED_IMAGE3F_CUDA_GRAPH",
    "backend": "tensorrt",
    "capability_report": "$FLEET_DIR/renderer_capabilities.json"
  },
  "baseline_checkpoint_sha256": "$BASELINE_CKPT_SHA256",
  "reference_config_sha256": "$REFERENCE_CONFIG_SHA256",
  "evaluation_submission_status": "$EVALUATION_SUBMISSION_STATUS",
  "evaluation_gate": {
    "enabled": $EVALUATION_ENABLED_JSON,
    "disabled_reason": "$EVALUATION_DISABLED_REASON",
    "observation_mode": "raytraced",
    "checkpoint_roles": ["baseline", "early", "mid", "final"],
    "fixed_seed_blocks": $FIXED_EVAL_SEEDS_JSON,
    "num_envs_per_seed": $FIXED_EVAL_NUM_ENVS,
    "one_episode_per_environment": true,
    "episodes_per_checkpoint": $FIXED_EVAL_EPISODES_PER_CHECKPOINT,
    "reference_config": "$REFERENCE_CONFIG",
    "min_absolute_success_delta": $FIXED_EVAL_MIN_SUCCESS_DELTA,
    "pass_rule": "paired final-minus-baseline success delta >= minimum and paired seed-block bootstrap 95% CI lower bound > 0",
    "reward_is_diagnostic_not_a_gate": true,
    "training_loss_used_as_evidence": false,
    "orchestrator_job_id": $eval_orchestrator_json,
    "evaluation_array_job_id": $eval_array_json,
    "report_job_id": $eval_report_json
  }
}
EOF
    mv "$temporary" "$target"
}

cancel_new_renderer_array() {
    if [ "$SUBMISSION_OWNS_RENDERER" != "1" ]; then
        return 0
    fi
    if [[ ! "$RENDER_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "FATAL: refusing to scancel malformed renderer job id: $RENDER_JOB_ID"
        return 1
    fi
    echo "Cancelling newly submitted renderer array $RENDER_JOB_ID"
    if scancel "$RENDER_JOB_ID"; then
        SUBMISSION_OWNS_RENDERER=0
        return 0
    fi
    return 1
}

cancel_new_trainer_job() {
    if [ "$SUBMISSION_OWNS_TRAINER" != "1" ]; then
        return 0
    fi
    if [[ ! "$TRAIN_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "FATAL: refusing to scancel malformed trainer job id: $TRAIN_JOB_ID"
        return 1
    fi
    echo "Cancelling newly submitted trainer $TRAIN_JOB_ID"
    if scancel "$TRAIN_JOB_ID"; then
        SUBMISSION_OWNS_TRAINER=0
        return 0
    fi
    return 1
}

cancel_new_cleanup_watchdog() {
    if [ "$SUBMISSION_OWNS_WATCHDOG" != "1" ]; then
        return 0
    fi
    if [[ ! "$RENDER_CLEANUP_WATCHDOG_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "FATAL: refusing to scancel malformed cleanup watchdog job id: $RENDER_CLEANUP_WATCHDOG_JOB_ID"
        return 1
    fi
    echo "Cancelling newly submitted renderer-cleanup watchdog $RENDER_CLEANUP_WATCHDOG_JOB_ID"
    if scancel "$RENDER_CLEANUP_WATCHDOG_JOB_ID"; then
        SUBMISSION_OWNS_WATCHDOG=0
        return 0
    fi
    return 1
}

cleanup_failed_submission() {
    local rc=$?
    local status=$FAILURE_STATUS
    trap - EXIT
    if [ "$SUBMISSION_OWNS_WATCHDOG" = "1" ]; then
        if cancel_new_cleanup_watchdog; then
            status="${status}_watchdog_cancelled"
        else
            status="${status}_watchdog_cancel_failed"
        fi
    fi
    if [ "$SUBMISSION_OWNS_TRAINER" = "1" ]; then
        if cancel_new_trainer_job; then
            status="${status}_trainer_cancelled"
        else
            status="${status}_trainer_cancel_failed"
        fi
    fi
    if [ "$SUBMISSION_OWNS_RENDERER" = "1" ]; then
        if cancel_new_renderer_array; then
            status="${status}_renderer_cancelled"
        else
            status="${status}_renderer_cancel_failed"
        fi
    fi
    write_job_manifest "$status" || true
    exit "$rc"
}
trap cleanup_failed_submission EXIT
write_job_manifest planned

EXPORTS="ALL,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,FLEET_DIR=$FLEET_DIR,IMAGE3F_RUN_ID=$RUN_ID,GLOBAL_NUM_ENVS=$GLOBAL_NUM_ENVS,ROLLOUT_STEPS=$ROLLOUT_STEPS,MIX=$MIX,REFERENCE_LAG=$REFERENCE_LAG,ENVS_PER_RENDERER=$ENVS_PER_RENDERER,NUM_MINI_BATCHES=$NUM_MINI_BATCHES,PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE,NUM_ROLLOUT_ITERATIONS=$NUM_ROLLOUT_ITERATIONS,NUM_LEARNING_EPOCHS=$NUM_LEARNING_EPOCHS,BEHAVIOR_SOURCE=$BEHAVIOR_SOURCE,TEACHER_ROLLOUT_RATIO=$TEACHER_ROLLOUT_RATIO,VG_PRIMARY_VIEW=$VG_PRIMARY_VIEW,SAVE_FREQUENCY=$SAVE_FREQUENCY,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_MODE=$WANDB_MODE,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PYTHON=$WANDB_PYTHON,PROJ=$PROJ,EXP=$EXP,CKPT=$CKPT,EXPECTED_TRAIN_CHECKPOINT_SHA256=$BASELINE_CKPT_SHA256,EXPECTED_FAST_ROBORENDER_REVISION=$EXPECTED_FAST_ROBORENDER_REVISION,EXPECTED_GR00T_REVISION=$EXPECTED_GR00T_REVISION,EXPECTED_IMAGE3F_RELEASE_ID=$EXPECTED_IMAGE3F_RELEASE_ID,EXPECTED_IMAGE3F_ADAPTER_SHA256=$EXPECTED_IMAGE3F_ADAPTER_SHA256,EXPECTED_IMAGE3F_INFERENCE_STEPS=$EXPECTED_IMAGE3F_INFERENCE_STEPS,EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=$EXPECTED_IMAGE3F_TEACACHE_THRESHOLD,EXPECTED_IMAGE3F_CFG_SCALE=$EXPECTED_IMAGE3F_CFG_SCALE,EXPECTED_IMAGE3F_CUDA_GRAPH=$EXPECTED_IMAGE3F_CUDA_GRAPH"

echo "Submitting $RUN_ID: $TRAINER_NODES trainer nodes x $LOCAL_ENVS envs, $RENDERERS renderers"
echo "Runtime manifest: $FLEET_DIR/submission_plan.json"

if [ "$RENDERERS" -gt 0 ]; then
    RENDER_CMD=(sbatch --parsable --time="$RENDER_WALLTIME" --array="0-$((RENDERERS - 1))" \
        --export="$EXPORTS" "$ROOT/scripts/server/image3f_server_array.sbatch")
    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY RUN:'; printf ' %q' "${RENDER_CMD[@]}"; printf '\n'
        RENDER_JOB_ID=DRY_RENDER_JOB
    else
        RENDER_JOB_ID=$("${RENDER_CMD[@]}")
        RENDER_JOB_ID=${RENDER_JOB_ID%%;*}
        [[ "$RENDER_JOB_ID" =~ ^[0-9]+$ ]] || {
            echo "FATAL: sbatch returned malformed renderer job id: $RENDER_JOB_ID"
            exit 2
        }
        SUBMISSION_OWNS_RENDERER=1
        write_job_manifest renderer_submitted
        echo "Renderer array job: $RENDER_JOB_ID"
    fi
fi

TRAIN_EXPORTS="$EXPORTS,RENDER_JOB_ID=$RENDER_JOB_ID"
TRAIN_DEPENDENCY_ARGS=()
if [ "$RENDERERS" -gt 0 ]; then
    # ArrayJobID means every renderer task has started. The trainer still runs
    # the active ping/provenance gate before allocating any rollout work.
    TRAIN_DEPENDENCY_ARGS+=(--dependency="after:$RENDER_JOB_ID")
fi
TRAIN_CMD=(sbatch --parsable --time="$RUN_WALLTIME" --signal=B:USR1@300 \
    "${TRAIN_DEPENDENCY_ARGS[@]}" --nodes="$TRAINER_NODES" --export="$TRAIN_EXPORTS" \
    "$ROOT/scripts/train/dagger_scale_256x300.sbatch")
if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN:'; printf ' %q' "${TRAIN_CMD[@]}"; printf '\n'
    TRAIN_JOB_ID=DRY_TRAIN_JOB
    write_job_manifest dry_run
else
    set +e
    TRAIN_JOB_ID=$("${TRAIN_CMD[@]}")
    TRAIN_SUBMIT_RC=$?
    set -e
    if [ "$TRAIN_SUBMIT_RC" -ne 0 ]; then
        TRAIN_JOB_ID=""
        FAILURE_STATUS=trainer_submission_failed
        write_job_manifest trainer_submission_failed
        echo "FATAL: trainer sbatch submission failed (rc=$TRAIN_SUBMIT_RC)"
        exit "$TRAIN_SUBMIT_RC"
    fi
    TRAIN_JOB_ID=${TRAIN_JOB_ID%%;*}
    [[ "$TRAIN_JOB_ID" =~ ^[0-9]+$ ]] || {
        FAILURE_STATUS=trainer_submission_malformed
        echo "FATAL: sbatch returned malformed trainer job id: $TRAIN_JOB_ID"
        exit 2
    }
    SUBMISSION_OWNS_TRAINER=1
    TRAIN_RUN_DIR_FOR_JOB=${TRAIN_RUN_DIR:-$GR00T_ROOT/logs_rl/$PROJ/${EXP}-${TRAIN_JOB_ID}}
    write_job_manifest trainer_submitted_cleanup_and_eval_pending
    echo "Trainer job: $TRAIN_JOB_ID (waits for $RENDERERS healthy renderer endpoints)"
fi

if [ "$DRY_RUN" = "1" ]; then
    TRAIN_RUN_DIR_FOR_JOB=${TRAIN_RUN_DIR:-$GR00T_ROOT/logs_rl/$PROJ/${EXP}-${TRAIN_JOB_ID}}
fi

CLEANUP_EXPORTS="$EXPORTS,RENDER_JOB_ID=$RENDER_JOB_ID,EXPECTED_TRAIN_JOB_ID=$TRAIN_JOB_ID"
RENDER_CLEANUP_CMD=(sbatch --parsable \
    --dependency="afterany:$TRAIN_JOB_ID" \
    --export="$CLEANUP_EXPORTS" \
    "$ROOT/scripts/scaling/renderer_cleanup_watchdog.sbatch")
if [ "$RENDERERS" -eq 0 ]; then
    echo "Renderer-cleanup watchdog: not needed (ray-traced-only run)"
elif [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN:'; printf ' %q' "${RENDER_CLEANUP_CMD[@]}"; printf '\n'
    RENDER_CLEANUP_WATCHDOG_JOB_ID=DRY_RENDER_CLEANUP_WATCHDOG_JOB
else
    set +e
    RENDER_CLEANUP_WATCHDOG_JOB_ID=$("${RENDER_CLEANUP_CMD[@]}")
    CLEANUP_SUBMIT_RC=$?
    set -e
    if [ "$CLEANUP_SUBMIT_RC" -ne 0 ]; then
        RENDER_CLEANUP_WATCHDOG_JOB_ID=""
        FAILURE_STATUS=renderer_cleanup_watchdog_submission_failed
        echo "FATAL: renderer-cleanup watchdog submission failed (rc=$CLEANUP_SUBMIT_RC)"
        exit "$CLEANUP_SUBMIT_RC"
    fi
    RENDER_CLEANUP_WATCHDOG_JOB_ID=${RENDER_CLEANUP_WATCHDOG_JOB_ID%%;*}
    if [[ ! "$RENDER_CLEANUP_WATCHDOG_JOB_ID" =~ ^[0-9]+$ ]]; then
        FAILURE_STATUS=renderer_cleanup_watchdog_submission_malformed
        echo "FATAL: sbatch returned malformed cleanup watchdog id: $RENDER_CLEANUP_WATCHDOG_JOB_ID"
        exit 2
    fi
    SUBMISSION_OWNS_WATCHDOG=1
    write_job_manifest trainer_and_cleanup_submitted_eval_gate_pending
fi

if [ "$ENABLE_FIXED_EVAL_GATE" = "1" ]; then
    EVAL_EXPORTS="$EXPORTS,TRAIN_RUN_DIR=$TRAIN_RUN_DIR_FOR_JOB,BASELINE_CKPT=$CKPT,REFERENCE_CONFIG=$REFERENCE_CONFIG,EXPECTED_BASELINE_CKPT_SHA256=$BASELINE_CKPT_SHA256,EXPECTED_REFERENCE_CONFIG_SHA256=$REFERENCE_CONFIG_SHA256,FIXED_EVAL_SEEDS_COLON=$FIXED_EVAL_SEEDS_COLON,FIXED_EVAL_NUM_ENVS=$FIXED_EVAL_NUM_ENVS,FIXED_EVAL_MAX_PARALLEL=$FIXED_EVAL_MAX_PARALLEL,MIN_SUCCESS_DELTA=$FIXED_EVAL_MIN_SUCCESS_DELTA,JOB_MANIFEST=$FLEET_DIR/slurm_jobs.json"
    EVAL_ORCHESTRATOR_CMD=(sbatch --parsable \
        --dependency="afterany:$TRAIN_JOB_ID" \
        --export="$EVAL_EXPORTS" \
        "$ROOT/scripts/eval/orchestrate_fixed_seed_gate.sbatch")
    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY RUN:'; printf ' %q' "${EVAL_ORCHESTRATOR_CMD[@]}"; printf '\n'
        EVAL_ORCHESTRATOR_JOB_ID=DRY_EVAL_ORCHESTRATOR_JOB
        EVAL_ARRAY_JOB_ID=DRY_DEFERRED_EVAL_ARRAY_JOB
        EVAL_REPORT_JOB_ID=DRY_DEFERRED_EVAL_REPORT_JOB
        EVALUATION_SUBMISSION_STATUS=dry_run
        write_job_manifest dry_run
    else
        set +e
        EVAL_ORCHESTRATOR_JOB_ID=$("${EVAL_ORCHESTRATOR_CMD[@]}")
        EVAL_SUBMIT_RC=$?
        set -e
        if [ "$EVAL_SUBMIT_RC" -ne 0 ]; then
            EVAL_ORCHESTRATOR_JOB_ID=""
            FAILURE_STATUS=eval_orchestrator_submission_failed
            EVALUATION_SUBMISSION_STATUS=orchestrator_submission_failed
            echo "FATAL: fixed-seed evaluation orchestrator submission failed (rc=$EVAL_SUBMIT_RC)"
            exit "$EVAL_SUBMIT_RC"
        fi
        EVAL_ORCHESTRATOR_JOB_ID=${EVAL_ORCHESTRATOR_JOB_ID%%;*}
        if [[ ! "$EVAL_ORCHESTRATOR_JOB_ID" =~ ^[0-9]+$ ]]; then
            FAILURE_STATUS=eval_orchestrator_submission_malformed
            EVALUATION_SUBMISSION_STATUS=orchestrator_submission_malformed
            echo "FATAL: sbatch returned malformed eval orchestrator job id: $EVAL_ORCHESTRATOR_JOB_ID"
            exit 2
        fi
        # Only after trainer, cleanup watchdog, and dependent evaluation gate are
        # accepted does the trainer own primary renderer cleanup. Before this point
        # launcher rollback cancels every submitted production job.
        SUBMISSION_OWNS_WATCHDOG=0
        SUBMISSION_OWNS_TRAINER=0
        SUBMISSION_OWNS_RENDERER=0
        EVALUATION_SUBMISSION_STATUS=orchestrator_submitted
        write_job_manifest production_and_eval_gate_submitted
        echo "Fixed-seed eval orchestrator: $EVAL_ORCHESTRATOR_JOB_ID (afterany:$TRAIN_JOB_ID)"
    fi
else
    # Bounded smoke launches cannot produce three distinct snapshots. The
    # eight-hour entry point pins ENABLE_FIXED_EVAL_GATE=1.
    SUBMISSION_OWNS_WATCHDOG=0
    SUBMISSION_OWNS_TRAINER=0
    SUBMISSION_OWNS_RENDERER=0
    write_job_manifest production_submitted_eval_gate_disabled_short_run
    echo "Fixed-seed eval gate: disabled (run too short for early/mid/final snapshots)"
fi
trap - EXIT

echo "Monitor: squeue -j ${RENDER_JOB_ID:-none},$TRAIN_JOB_ID,$RENDER_CLEANUP_WATCHDOG_JOB_ID,$EVAL_ORCHESTRATOR_JOB_ID"
echo "Fleet:   $FLEET_DIR"
