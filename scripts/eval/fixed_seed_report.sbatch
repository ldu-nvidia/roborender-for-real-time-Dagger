#!/bin/bash
# Aggregate all fixed-seed blocks, upload comparable metrics to the same personal
# W&B project, and make the Slurm job red unless the policy-improvement gate passes.
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=rr_eval_report
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/rr_eval_report_%j.log
set -Eeuo pipefail

ROOT=${ROOT:-/svl/u/ravenh/fast_roborender}
EVAL_MANIFEST=${EVAL_MANIFEST:?EVAL_MANIFEST is required}
EVAL_DIR=${EVAL_DIR:?EVAL_DIR is required}
PROJ=${PROJ:-roborender_dagger_scale}
EXP=${EXP:-dagger_fixed_seed_gate}
MIN_SUCCESS_DELTA=${MIN_SUCCESS_DELTA:-0.02}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_PYTHON=${WANDB_PYTHON:-/svl/u/ravenh/miniconda3/envs/viral/bin/python}
SUMMARY="$EVAL_DIR/fixed_seed_improvement_report.json"

set +e
"$WANDB_PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" summarize \
    --manifest "$EVAL_MANIFEST" \
    --results-dir "$EVAL_DIR/results" \
    --min-success-delta "$MIN_SUCCESS_DELTA" \
    --output "$SUMMARY"
GATE_RC=$?
set -e

# Incomplete evaluation is never uploaded as if it were a scientific result.
if [ "$GATE_RC" -eq 2 ]; then
    echo "FATAL: fixed-seed evaluation is incomplete; see $SUMMARY"
    exit 2
fi

WANDB_PREFLIGHT=("$WANDB_PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py"
    --expected-email "$WANDB_EXPECTED_EMAIL" --format shell)
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_PREFLIGHT+=(--entity "$WANDB_ENTITY")
fi
WANDB_IDENTITY=$(WANDB_MODE="$WANDB_MODE" "${WANDB_PREFLIGHT[@]}")
case "$WANDB_IDENTITY" in
    WANDB_ENTITY=?*) WANDB_ENTITY=${WANDB_IDENTITY#WANDB_ENTITY=} ;;
    *) echo "FATAL: W&B identity preflight did not return a personal entity"; exit 2 ;;
esac

WANDB_MODE="$WANDB_MODE" "$WANDB_PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" upload \
    --manifest "$EVAL_MANIFEST" \
    --summary "$SUMMARY" \
    --project "$PROJ" \
    --entity "$WANDB_ENTITY" \
    --run-name "${EXP}-fixed-seed-gate"

if [ "$GATE_RC" -ne 0 ]; then
    echo "POLICY_IMPROVEMENT_GATE_FAIL report=$SUMMARY"
    exit "$GATE_RC"
fi
echo "POLICY_IMPROVEMENT_GATE_PASS report=$SUMMARY"
