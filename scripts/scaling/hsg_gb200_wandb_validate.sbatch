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
OUTPUT=${OUTPUT:-$STAGE_ROOT/runtime/validation/wandb_online_smoke_${SLURM_JOB_ID}.json}
WANDB_ACCEPTANCE=${WANDB_ACCEPTANCE:-$STAGE_ROOT/runtime/validation/wandb_online_smoke_acceptance.json}
WANDB_HOME=${WANDB_HOME:-$STAGE_ROOT/runtime/home}
WANDB_SCRATCH=${WANDB_SCRATCH:-$STAGE_ROOT/runtime/wandb-validation/$SLURM_JOB_ID}
export STAGE_ROOT ROOT CONTAINER EXPECTED_EMAIL PROJECT RUN_ID OUTPUT WANDB_ACCEPTANCE WANDB_HOME WANDB_SCRATCH

[ -s "$CONTAINER" ] || { echo "FATAL: trainer container missing: $CONTAINER" >&2; exit 2; }
[ -s "$WANDB_HOME/.netrc" ] || { echo "FATAL: W&B credential missing" >&2; exit 2; }
mkdir -p "$STAGE_ROOT/runtime/validation" "$STAGE_ROOT/runtime/wandb-validation"
readarray -t CANONICAL_PATHS < <(python3 - "$STAGE_ROOT" "$ROOT" \
    "$WANDB_SCRATCH" "$OUTPUT" "$WANDB_ACCEPTANCE" <<'PY'
import os
import sys
from pathlib import Path

stage = Path(sys.argv[1]).resolve(strict=True)
source = Path(sys.argv[2]).resolve(strict=True)
scratch_root = (stage / "runtime" / "wandb-validation").resolve(strict=True)
output_root = (stage / "runtime" / "validation").resolve(strict=True)
for allowed in (scratch_root, output_root):
    if allowed.is_symlink():
        raise SystemExit("FATAL: W&B writable root must not be a symlink")

def validate(raw: str, allowed: Path, label: str, *, allow_file: bool) -> Path:
    lexical = Path(os.path.abspath(raw))
    try:
        lexical_relative = lexical.relative_to(allowed)
    except ValueError:
        raise SystemExit(f"FATAL: {label} is not lexically below its runtime root") from None
    resolved = lexical.resolve(strict=False)
    try:
        relative = resolved.relative_to(allowed)
    except ValueError:
        raise SystemExit(f"FATAL: {label} escaped its writable runtime root") from None
    if not relative.parts:
        raise SystemExit(f"FATAL: {label} must be below, not equal to, its writable root")
    probe = allowed
    components = lexical_relative.parts[:-1] if allow_file else lexical_relative.parts
    for component in components:
        probe = probe / component
        if probe.exists() and probe.is_symlink():
            raise SystemExit(f"FATAL: {label} has a symlink ancestor")
    if lexical.exists() and lexical.is_symlink():
        raise SystemExit(f"FATAL: {label} itself is a symlink")
    return resolved

scratch = validate(sys.argv[3], scratch_root, "W&B scratch", allow_file=False)
output = validate(sys.argv[4], output_root, "W&B output", allow_file=True)
acceptance = validate(sys.argv[5], output_root, "W&B acceptance", allow_file=True)
if output.suffix != ".json" or acceptance.suffix != ".json":
    raise SystemExit("FATAL: W&B output and acceptance must be JSON files")
if output == acceptance:
    raise SystemExit("FATAL: job-scoped W&B output and campaign acceptance must differ")
if source == scratch or source in scratch.parents:
    raise SystemExit("FATAL: W&B scratch may not target the sealed source tree")
print(scratch)
print(output)
print(acceptance)
PY
)
[ "${#CANONICAL_PATHS[@]}" -eq 3 ] || { echo "FATAL: W&B path validation failed" >&2; exit 2; }
WANDB_SCRATCH=${CANONICAL_PATHS[0]}
OUTPUT=${CANONICAL_PATHS[1]}
WANDB_ACCEPTANCE=${CANONICAL_PATHS[2]}
export OUTPUT WANDB_ACCEPTANCE WANDB_SCRATCH
mkdir -p "$WANDB_SCRATCH/work" \
    "$WANDB_SCRATCH/wandb" "$WANDB_SCRATCH/cache" \
    "$WANDB_SCRATCH/config" "$WANDB_SCRATCH/data" "$WANDB_SCRATCH/tmp"

srun --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=8 --mem=32G \
    --exact --exclusive --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$ROOT:$ROOT:ro" \
    --container-workdir="$WANDB_SCRATCH/work" \
    env HOME="$WANDB_HOME" WANDB_MODE=online \
        WANDB_DIR="$WANDB_SCRATCH/wandb" \
        WANDB_CACHE_DIR="$WANDB_SCRATCH/cache" \
        WANDB_CONFIG_DIR="$WANDB_SCRATCH/config" \
        WANDB_DATA_DIR="$WANDB_SCRATCH/data" \
        TMPDIR="$WANDB_SCRATCH/tmp" \
        PYTHONPATH="$ROOT/scripts/scaling" \
        /usr/local/bin/roborender-trainer-sm100 \
        python "$ROOT/scripts/scaling/wandb_online_smoke.py" \
            --expected-email "$EXPECTED_EMAIL" \
            --project "$PROJECT" --run-id "$RUN_ID" --output "$OUTPUT"

grep -q '"status": "PASS"' "$OUTPUT"
python3 "$ROOT/scripts/scaling/promote_wandb_acceptance.py" \
    --source "$OUTPUT" --destination "$WANDB_ACCEPTANCE" \
    --expected-email "$EXPECTED_EMAIL"
echo "HSG_GB200_WANDB_ONLINE_PASS"
