#!/bin/bash
# Validate the authenticated personal W&B workspace and media transport.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=rr_gb200_wandb_validate
set -Eeuo pipefail

STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
CONTAINER=${CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
PROJECT=${WANDB_PROJECT:-roborender_dagger_gb200_validation}
RUN_ID=${WANDB_RUN_ID:-gb200_wandb_smoke_${SLURM_JOB_ID}}
OUTPUT=${OUTPUT:-$STAGE_ROOT/validation/wandb_online_smoke_${SLURM_JOB_ID}.json}
WANDB_HOME=${WANDB_HOME:-$STAGE_ROOT/runtime/home}
export STAGE_ROOT ROOT CONTAINER EXPECTED_EMAIL PROJECT RUN_ID OUTPUT WANDB_HOME

[ -s "$CONTAINER" ] || { echo "FATAL: trainer container missing: $CONTAINER" >&2; exit 2; }
[ -s "$WANDB_HOME/.netrc" ] || { echo "FATAL: W&B credential missing" >&2; exit 2; }
mkdir -p "$STAGE_ROOT/validation"

srun --nodes=1 --ntasks=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT" \
    --container-workdir="$ROOT/scripts/scaling" \
    env HOME="$WANDB_HOME" WANDB_MODE=online \
        PYTHONPATH="$ROOT/scripts/scaling" \
        /usr/local/bin/roborender-trainer-sm100 \
        python "$ROOT/scripts/scaling/wandb_online_smoke.py" \
            --expected-email "$EXPECTED_EMAIL" \
            --project "$PROJECT" --run-id "$RUN_ID" --output "$OUTPUT"

grep -q '"status": "PASS"' "$OUTPUT"
echo "HSG_GB200_WANDB_ONLINE_PASS"
