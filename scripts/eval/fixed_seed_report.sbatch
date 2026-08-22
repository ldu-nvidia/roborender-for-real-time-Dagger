#!/bin/bash
# Aggregate all fixed-seed blocks, upload comparable metrics to the authorized
# W&B team project, and make the job red unless the policy-improvement gate passes.
#SBATCH --partition=cpu_short
#SBATCH --account=general_cs_infra
#SBATCH --job-name=rr_eval_report
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
EVAL_MANIFEST=${EVAL_MANIFEST:?EVAL_MANIFEST is required}
EVAL_DIR=${EVAL_DIR:?EVAL_DIR is required}
PROJ=${PROJ:-roborender_dagger_scale}
EXP=${EXP:-dagger_fixed_seed_gate}
MIN_SUCCESS_DELTA=${MIN_SUCCESS_DELTA:-0.02}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}
WANDB_PYTHON=${WANDB_PYTHON:-$STAGE_ROOT/runtime/envs/wandb-host/bin/python}
export HOME=${HOME_OVERRIDE:-$STAGE_ROOT/runtime/home}
SUMMARY="$EVAL_DIR/fixed_seed_improvement_report.json"

[ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || { echo "FATAL: W&B account binding changed"; exit 2; }
[ "$WANDB_ENTITY" = nvidia ] || { echo "FATAL: W&B team entity binding changed"; exit 2; }
[ "$WANDB_AUTHORIZATION_SCOPE" = team ] || { echo "FATAL: W&B authorization scope changed"; exit 2; }

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
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY"
    --authorization-scope "$WANDB_AUTHORIZATION_SCOPE" --format shell)
WANDB_IDENTITY=$(WANDB_MODE="$WANDB_MODE" "${WANDB_PREFLIGHT[@]}")
case "$WANDB_IDENTITY" in
    WANDB_ENTITY=?*) WANDB_ENTITY=${WANDB_IDENTITY#WANDB_ENTITY=} ;;
    *) echo "FATAL: W&B identity preflight did not return the authorized team"; exit 2 ;;
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
