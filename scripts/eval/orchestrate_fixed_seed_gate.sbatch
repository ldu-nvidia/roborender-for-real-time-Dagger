#!/bin/bash
# Run after the trainer (normally dependency=afterany:<trainer job>).  It discovers
# the snapshots that actually exist, then submits four checkpoint evaluations and
# a fail-closed report job.
#SBATCH --partition=cpu_short
#SBATCH --account=general_cs_infra
#SBATCH --job-name=rr_eval_orchestrate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:?TRAIN_RUN_DIR is required}
BASELINE_CKPT=${BASELINE_CKPT:?BASELINE_CKPT is required}
REFERENCE_CONFIG=${REFERENCE_CONFIG:?REFERENCE_CONFIG must identify the staged OCI evaluation config}
EXPECTED_BASELINE_CKPT_SHA256=${EXPECTED_BASELINE_CKPT_SHA256:?EXPECTED_BASELINE_CKPT_SHA256 is required}
EXPECTED_REFERENCE_CONFIG_SHA256=${EXPECTED_REFERENCE_CONFIG_SHA256:?EXPECTED_REFERENCE_CONFIG_SHA256 is required}
FIXED_EVAL_SEEDS=${FIXED_EVAL_SEEDS:-}
if [ -z "$FIXED_EVAL_SEEDS" ] && [ -n "${FIXED_EVAL_SEEDS_COLON:-}" ]; then
    FIXED_EVAL_SEEDS=${FIXED_EVAL_SEEDS_COLON//:/,}
fi
FIXED_EVAL_SEEDS=${FIXED_EVAL_SEEDS:-1729,3253,4519,5987,7411,8863,10193,12203}
FIXED_EVAL_NUM_ENVS=${FIXED_EVAL_NUM_ENVS:-32}
FIXED_EVAL_MAX_PARALLEL=${FIXED_EVAL_MAX_PARALLEL:-4}
EVAL_DIR=${EVAL_DIR:-$TRAIN_RUN_DIR/fixed_seed_eval}
EVAL_MANIFEST="$EVAL_DIR/manifest.json"
JOB_MANIFEST=${JOB_MANIFEST:-}
PYTHON=${PYTHON:-$STAGE_ROOT/runtime/envs/wandb-host/bin/python}
PROJ=${PROJ:-roborender_dagger_scale}
EXP=${EXP:-dagger_fixed_seed_gate}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}
WANDB_PYTHON=${WANDB_PYTHON:-$PYTHON}
MIN_SUCCESS_DELTA=${MIN_SUCCESS_DELTA:-0.02}

[ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || { echo "FATAL: W&B account binding changed"; exit 2; }
[ "$WANDB_ENTITY" = nvidia ] || { echo "FATAL: W&B team entity binding changed"; exit 2; }
[ "$WANDB_AUTHORIZATION_SCOPE" = team ] || { echo "FATAL: W&B authorization scope changed"; exit 2; }

EVAL_JOB_ID=""
REPORT_JOB_ID=""
OWNS_EVAL_ARRAY=0
OWNS_REPORT_JOB=0

update_job_manifest() {
    local status=$1
    [ -n "$JOB_MANIFEST" ] || return 0
    "$PYTHON" - "$JOB_MANIFEST" "$SLURM_JOB_ID" "$EVAL_JOB_ID" "$REPORT_JOB_ID" "$status" <<'PY'
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
payload = json.loads(target.read_text(encoding="utf-8"))
expected = str(payload.get("eval_orchestrator_job_id"))
if expected != sys.argv[2]:
    raise SystemExit(
        f"FATAL: job manifest belongs to eval orchestrator {expected}, not {sys.argv[2]}"
    )
payload["eval_array_job_id"] = sys.argv[3] or None
payload["eval_report_job_id"] = sys.argv[4] or None
payload["evaluation_submission_status"] = sys.argv[5]
evaluation = payload.setdefault("evaluation_gate", {})
evaluation["orchestrator_job_id"] = sys.argv[2]
evaluation["evaluation_array_job_id"] = sys.argv[3] or None
evaluation["report_job_id"] = sys.argv[4] or None
temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

cleanup_failed_orchestration() {
    local rc=$?
    trap - EXIT
    local rolled_back=0
    if [ "$OWNS_REPORT_JOB" = "1" ] && [[ "$REPORT_JOB_ID" =~ ^[0-9]+$ ]]; then
        scancel "$REPORT_JOB_ID" || true
        rolled_back=1
    fi
    if [ "$OWNS_EVAL_ARRAY" = "1" ] && [[ "$EVAL_JOB_ID" =~ ^[0-9]+$ ]]; then
        scancel "$EVAL_JOB_ID" || true
        rolled_back=1
    fi
    if [ "$rolled_back" = "1" ]; then
        update_job_manifest eval_submission_rolled_back || true
    else
        update_job_manifest eval_orchestrator_failed || true
    fi
    exit "$rc"
}
trap cleanup_failed_orchestration EXIT

mkdir -p "$EVAL_DIR" "$ROOT/logs_archive"
TASK_COUNT=$("$PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" prepare \
    --train-run-dir "$TRAIN_RUN_DIR" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --reference-config "$REFERENCE_CONFIG" \
    --seeds "$FIXED_EVAL_SEEDS" \
    --num-envs "$FIXED_EVAL_NUM_ENVS" \
    --expected-baseline-sha256 "$EXPECTED_BASELINE_CKPT_SHA256" \
    --expected-reference-config-sha256 "$EXPECTED_REFERENCE_CONFIG_SHA256" \
    --output "$EVAL_MANIFEST")
[[ "$TASK_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: invalid task count: $TASK_COUNT"; exit 2; }

EXPORTS="ALL,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,EVAL_MANIFEST=$EVAL_MANIFEST,EVAL_DIR=$EVAL_DIR,PROJ=$PROJ,EXP=$EXP,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_MODE=$WANDB_MODE,WANDB_ENTITY=$WANDB_ENTITY,WANDB_AUTHORIZATION_SCOPE=$WANDB_AUTHORIZATION_SCOPE,WANDB_PYTHON=$WANDB_PYTHON,MIN_SUCCESS_DELTA=$MIN_SUCCESS_DELTA"
EVAL_JOB_ID=$(sbatch --parsable \
    --array="0-$((TASK_COUNT - 1))%$FIXED_EVAL_MAX_PARALLEL" \
    --export="$EXPORTS" \
    "$ROOT/scripts/eval/fixed_seed_policy_eval.sbatch")
EVAL_JOB_ID=${EVAL_JOB_ID%%;*}
[[ "$EVAL_JOB_ID" =~ ^[0-9]+$ ]] || { echo "FATAL: malformed eval job id: $EVAL_JOB_ID"; exit 2; }
OWNS_EVAL_ARRAY=1
update_job_manifest eval_array_submitted

REPORT_JOB_ID=$(sbatch --parsable \
    --dependency="afterany:$EVAL_JOB_ID" \
    --export="$EXPORTS" \
    "$ROOT/scripts/eval/fixed_seed_report.sbatch")
REPORT_JOB_ID=${REPORT_JOB_ID%%;*}
[[ "$REPORT_JOB_ID" =~ ^[0-9]+$ ]] || { echo "FATAL: malformed report job id: $REPORT_JOB_ID"; exit 2; }
OWNS_REPORT_JOB=1
update_job_manifest eval_and_report_submitted

"$PYTHON" - "$EVAL_DIR/jobs.json" "$SLURM_JOB_ID" "$EVAL_JOB_ID" "$REPORT_JOB_ID" <<'PY'
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "orchestrator_job_id": sys.argv[2],
    "evaluation_array_job_id": sys.argv[3],
    "report_job_id": sys.argv[4],
}
temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

# Ownership transfers only after both the shared job manifest and the local
# jobs record are atomically written and fsynced.  Any earlier failure rolls
# back the report first and then its evaluation-array dependency.
OWNS_REPORT_JOB=0
OWNS_EVAL_ARRAY=0

echo "Fixed-seed evaluation array: $EVAL_JOB_ID"
echo "Policy-improvement report:   $REPORT_JOB_ID"
echo "Evaluation manifest:         $EVAL_MANIFEST"
trap - EXIT
