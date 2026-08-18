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
NUM_TOTAL_BATCHES=${NUM_TOTAL_BATCHES:-1}
RUN_ID=${RUN_ID:-scale_$(date +%Y%m%d_%H%M%S)}
FLEET_DIR=${FLEET_DIR:-$ROOT/runtime/$RUN_ID}
DRY_RUN=${DRY_RUN:-0}
EXPECTED_FAST_ROBORENDER_REVISION=${EXPECTED_FAST_ROBORENDER_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}
EXPECTED_GR00T_REVISION=${EXPECTED_GR00T_REVISION:-$(git -C "$GR00T_ROOT" rev-parse HEAD)}
CKPT=${CKPT:-$GR00T_ROOT/logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/last.pt}

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
    INSTALL_CHECK=(python3 "$ROOT/scripts/scaling/install_gr00t_integration.py"
        --gr00t-root "$GR00T_ROOT" --expected-revision "$EXPECTED_GR00T_REVISION")
    if [ "${ALLOW_DIRTY_GR00T:-0}" = "1" ]; then
        INSTALL_CHECK+=(--allow-dirty)
    fi
    "${INSTALL_CHECK[@]}"
fi

PLAN=(
    --global-envs "$GLOBAL_NUM_ENVS"
    --rollout-steps "$ROLLOUT_STEPS"
    --trainer-nodes "$TRAINER_NODES"
    --raytraced-mix "$MIX"
    --envs-per-renderer "$ENVS_PER_RENDERER"
    --reference-lag "$REFERENCE_LAG"
    --num-mini-batches "$NUM_MINI_BATCHES"
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
)
RENDERERS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field required_renderers)
LOCAL_ENVS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field local_envs)

mkdir -p "$FLEET_DIR/endpoints" "$FLEET_DIR/queues" "$ROOT/logs_archive"
python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --format json \
    > "$FLEET_DIR/submission_plan.json"

EXPORTS="ALL,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,FLEET_DIR=$FLEET_DIR,IMAGE3F_RUN_ID=$RUN_ID,GLOBAL_NUM_ENVS=$GLOBAL_NUM_ENVS,ROLLOUT_STEPS=$ROLLOUT_STEPS,MIX=$MIX,REFERENCE_LAG=$REFERENCE_LAG,ENVS_PER_RENDERER=$ENVS_PER_RENDERER,NUM_MINI_BATCHES=$NUM_MINI_BATCHES,PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE,NUM_TOTAL_BATCHES=$NUM_TOTAL_BATCHES,CKPT=$CKPT,EXPECTED_FAST_ROBORENDER_REVISION=$EXPECTED_FAST_ROBORENDER_REVISION,EXPECTED_GR00T_REVISION=$EXPECTED_GR00T_REVISION"

echo "Submitting $RUN_ID: $TRAINER_NODES trainer nodes x $LOCAL_ENVS envs, $RENDERERS renderers"
echo "Runtime manifest: $FLEET_DIR/submission_plan.json"

RENDER_JOB_ID=""
if [ "$RENDERERS" -gt 0 ]; then
    RENDER_CMD=(sbatch --parsable --array="0-$((RENDERERS - 1))" \
        --export="$EXPORTS" "$ROOT/scripts/server/image3f_server_array.sbatch")
    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY RUN:'; printf ' %q' "${RENDER_CMD[@]}"; printf '\n'
        RENDER_JOB_ID=DRY_RENDER_JOB
    else
        RENDER_JOB_ID=$("${RENDER_CMD[@]}")
        echo "Renderer array job: $RENDER_JOB_ID"
    fi
fi

TRAIN_CMD=(sbatch --parsable --nodes="$TRAINER_NODES" --export="$EXPORTS" \
    "$ROOT/scripts/train/dagger_scale_256x300.sbatch")
if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN:'; printf ' %q' "${TRAIN_CMD[@]}"; printf '\n'
    TRAIN_JOB_ID=DRY_TRAIN_JOB
else
    TRAIN_JOB_ID=$("${TRAIN_CMD[@]}")
    echo "Trainer job: $TRAIN_JOB_ID (waits for $RENDERERS healthy renderer endpoints)"
fi

cat > "$FLEET_DIR/slurm_jobs.json" <<EOF
{
  "run_id": "$RUN_ID",
  "renderer_array_job_id": "$RENDER_JOB_ID",
  "trainer_job_id": "$TRAIN_JOB_ID",
  "renderer_count": $RENDERERS,
  "trainer_nodes": $TRAINER_NODES,
  "global_environments": $GLOBAL_NUM_ENVS,
  "local_environments": $LOCAL_ENVS,
  "rollout_steps": $ROLLOUT_STEPS,
  "optimizer_batches": $NUM_TOTAL_BATCHES
}
EOF

echo "Monitor: squeue -j ${RENDER_JOB_ID:-none},$TRAIN_JOB_ID"
echo "Fleet:   $FLEET_DIR"
