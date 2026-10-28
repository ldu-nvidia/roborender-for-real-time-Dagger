#!/bin/bash
# Raven-equivalent three-view offline BC, without starting Isaac or a renderer.
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$PROJECT_ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
# New runs have no held-out MSE pass; callers must supply --max-epochs explicitly.
# Historical protocol reproduction requires an explicit compatibility choice.
# Selection considers exact CLI overrides too; a partial override must not mix
# the published union with historical split/root defaults. This is metadata
# only: the trainer still performs full data admission and strict resume checks.
selection=$("$GR00T_PYTHON" "$PROJECT_ROOT/pipeline/bc_training/launcher_data.py" \
    --project-root "$PROJECT_ROOT" --asset-root "$ROBORENDER_ASSET_ROOT" -- "$@")
mapfile -t selected_data <<< "$selection"
[[ ${#selected_data[@]} == 5 ]] || { echo "invalid BC data selection" >&2; exit 2; }
TRAINING_PROTOCOL=${selected_data[0]}
DATASET_ROOT=${selected_data[1]}
TRAIN_SPLIT=${selected_data[2]}
VALIDATION_SPLIT=${selected_data[3]}
DATA_ADMISSION_JSON=${selected_data[4]}
# Resolve the default only after CLI/environment protocol selection. Historical
# validation keeps B16; new fixed-final replay uses one complete B32 population.
if [[ $TRAINING_PROTOCOL == fixed_final ]]; then
    CHUNK_BATCH_SIZE=${CHUNK_BATCH_SIZE:-32}
else
    CHUNK_BATCH_SIZE=${CHUNK_BATCH_SIZE:-16}
fi
# Match argparse's last exact CLI value before metadata/divisibility checks.
# Do not consume "$@": the learner must still validate the original arguments.
batch_value_pending=0
for argument in "$@"; do
    if [[ $batch_value_pending == 1 ]]; then
        CHUNK_BATCH_SIZE=$argument
        batch_value_pending=0
    elif [[ $argument == --chunk-batch-size ]]; then
        batch_value_pending=1
    elif [[ $argument == --chunk-batch-size=* ]]; then
        CHUNK_BATCH_SIZE=${argument#*=}
    elif [[ $argument == --chunk-* ]]; then
        echo "spell the full --chunk-batch-size option" >&2; exit 2
    fi
done
[[ $batch_value_pending == 0 ]] || { echo "--chunk-batch-size requires a value" >&2; exit 2; }
STUDENT_BOOTSTRAP=${STUDENT_BOOTSTRAP:-$ROBORENDER_ASSET_ROOT/checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt}
POLICY_SPEC=${POLICY_SPEC:-$ROBORENDER_ASSET_ROOT/checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json}
REQUIRE_PRODUCTION_DATA_ADMISSION=${REQUIRE_PRODUCTION_DATA_ADMISSION:-1}
ALLOW_QUALIFICATION_DATA=${ALLOW_QUALIFICATION_DATA:-0}
export WANDB_MODE=${WANDB_MODE:-online}

admission_args=()
if [[ -f $DATA_ADMISSION_JSON || $REQUIRE_PRODUCTION_DATA_ADMISSION == 1 ]]; then
    admission_args+=(--data-admission-json "$DATA_ADMISSION_JSON")
fi
case "$REQUIRE_PRODUCTION_DATA_ADMISSION" in
    1) admission_args+=(--require-production-data-admission) ;;
    0) admission_args+=(--diagnostic-without-production-admission)
       echo "WARNING: diagnostic-only BC; production data admission explicitly disabled" >&2 ;;
    *) echo "REQUIRE_PRODUCTION_DATA_ADMISSION must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALLOW_QUALIFICATION_DATA" in
    1) admission_args+=(--allow-qualification-data) ;;
    0) ;;
    *) echo "ALLOW_QUALIFICATION_DATA must be 0 or 1" >&2; exit 2 ;;
esac

[[ $CHUNK_BATCH_SIZE =~ ^[1-9][0-9]*$ ]] || { echo "CHUNK_BATCH_SIZE must be positive" >&2; exit 2; }
[[ $CHUNK_BATCH_SIZE == 16 || $CHUNK_BATCH_SIZE == 32 ]] || {
    echo "BC requires explicit full CHUNK_BATCH_SIZE=16 or32" >&2
    exit 2
}
[[ $TRAINING_PROTOCOL == fixed_final || $CHUNK_BATCH_SIZE == 16 ]] || {
    echo "Historical validation BC retains CHUNK_BATCH_SIZE=16" >&2; exit 2
}
[[ -x $GR00T_PYTHON ]] || { echo "GR00T_PYTHON is not executable" >&2; exit 2; }
for required in \
    "$TRAIN_SPLIT" \
    "$VALIDATION_SPLIT" \
    "$STUDENT_BOOTSTRAP" \
    "$POLICY_SPEC"; do
    [[ -f $required ]] || { echo "required BC input is missing: $required" >&2; exit 2; }
done
"$GR00T_PYTHON" -c 'import torch, hydra, wandb' 2>/dev/null || {
    echo "qualified runtime lacks BC training dependencies" >&2
    exit 2
}
"$GR00T_PYTHON" - "$TRAIN_SPLIT" "$VALIDATION_SPLIT" "$CHUNK_BATCH_SIZE" "$TRAINING_PROTOCOL" <<'PY'
import json, sys
from pathlib import Path

train_path, validation_path, batch = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
train = json.loads(train_path.read_text())
validation = json.loads(validation_path.read_text())
assert train["split"] == "train" and train["chunk_count"] == len(train["items"])
assert validation["split"] == "validation" and validation["chunk_count"] == len(validation["items"])
assert not ({item["stream_id"] for item in train["items"]}
            & {item["stream_id"] for item in validation["items"]})
assert len(train["items"]) > 0, "BC requires a nonempty sorted cyclic pool"
cyclic = sys.argv[4] == "fixed_final"
if not cyclic:
    assert len(train["items"]) % batch == 0, (
        f"historical training chunk count must be divisible by {batch}; refusing to drop or pad a partial batch"
    )
steps = (len(train["items"]) + batch - 1) // batch if cyclic else len(train["items"]) // batch
print(f"BC preflight PASS: train={train['chunk_count']} validation={validation['chunk_count']} "
      f"batch_chunks={batch} frames_per_update={batch * 81} "
      f"steps_per_epoch={steps} persistent_cyclic={cyclic}")
PY
if [[ $WANDB_MODE == online ]]; then
    "$GR00T_PYTHON" -c 'import wandb; assert wandb.api.api_key, "run wandb login first"' || {
        echo "W&B online logging requested but no API key is configured" >&2
        exit 2
    }
fi

exec "$GR00T_PYTHON" "$PROJECT_ROOT/pipeline/bc_training/train_bc_h200.py" \
    --training-protocol "$TRAINING_PROTOCOL" \
    --dataset-root "$DATASET_ROOT" \
    --train-split "$TRAIN_SPLIT" \
    --validation-split "$VALIDATION_SPLIT" \
    --bootstrap "$STUDENT_BOOTSTRAP" \
    --policy-spec "$POLICY_SPEC" \
    --output-root "$ROBORENDER_ASSET_ROOT/checkpoints/students/BC" \
    --chunk-batch-size "$CHUNK_BATCH_SIZE" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project dagger \
    --wandb-group BC \
    --wandb-run-name BC \
    "${admission_args[@]}" \
    "$@"
