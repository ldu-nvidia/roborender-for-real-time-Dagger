#!/bin/bash
# Redundant fleet cleanup for the case where a pending trainer is cancelled and
# therefore never executes its own EXIT trap. Submit with afterany:<trainer_job>.
#SBATCH --partition=backfill_block1
#SBATCH --account=general_cs_infra
#SBATCH --job-name=rr_fleet_cleanup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
set -Eeuo pipefail

RENDER_JOB_ID=${RENDER_JOB_ID:?RENDER_JOB_ID is required}
EXPECTED_TRAIN_JOB_ID=${EXPECTED_TRAIN_JOB_ID:?EXPECTED_TRAIN_JOB_ID is required}
FLEET_DIR=${FLEET_DIR:?FLEET_DIR is required}

[[ "$RENDER_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "FATAL: refusing malformed renderer array job id: $RENDER_JOB_ID"
    exit 2
}
[[ "$EXPECTED_TRAIN_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "FATAL: refusing malformed trainer job id: $EXPECTED_TRAIN_JOB_ID"
    exit 2
}

mkdir -p "$FLEET_DIR"
if squeue -h -j "$RENDER_JOB_ID" 2>/dev/null | grep -q .; then
    scancel "$RENDER_JOB_ID"
    OUTCOME=renderer_cancelled
else
    OUTCOME=renderer_already_terminal
fi
printf '%s\n' \
    "watchdog_job_id=${SLURM_JOB_ID:-manual}" \
    "trainer_job_id=$EXPECTED_TRAIN_JOB_ID" \
    "renderer_array_job_id=$RENDER_JOB_ID" \
    "outcome=$OUTCOME" \
    > "$FLEET_DIR/renderer_cleanup_watchdog.txt"
echo "RENDERER_CLEANUP_WATCHDOG_DONE outcome=$OUTCOME renderer=$RENDER_JOB_ID"
