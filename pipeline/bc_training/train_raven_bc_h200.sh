#!/bin/bash
# Raven-equivalent three-view offline BC, without starting Isaac or a renderer.
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$PROJECT_ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
CHUNK_BATCH_SIZE=${CHUNK_BATCH_SIZE:-4}
TRAIN_SPLIT=$PROJECT_ROOT/data/BC/splits/train.json
VALIDATION_SPLIT=$PROJECT_ROOT/data/BC/splits/validation.json
STUDENT_BOOTSTRAP=$PROJECT_ROOT/checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt
POLICY_SPEC=$PROJECT_ROOT/checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json
export WANDB_MODE=${WANDB_MODE:-online}

[[ $CHUNK_BATCH_SIZE =~ ^[1-9][0-9]*$ ]] || { echo "CHUNK_BATCH_SIZE must be positive" >&2; exit 2; }
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
"$GR00T_PYTHON" - "$TRAIN_SPLIT" "$VALIDATION_SPLIT" "$CHUNK_BATCH_SIZE" <<'PY'
import json, sys
from pathlib import Path

train_path, validation_path, batch = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
train = json.loads(train_path.read_text())
validation = json.loads(validation_path.read_text())
assert train["split"] == "train" and train["chunk_count"] == len(train["items"])
assert validation["split"] == "validation" and validation["chunk_count"] == len(validation["items"])
assert not ({item["stream_id"] for item in train["items"]}
            & {item["stream_id"] for item in validation["items"]})
print(f"BC preflight PASS: train={train['chunk_count']} validation={validation['chunk_count']} "
      f"batch_chunks={batch} steps_per_epoch={(train['chunk_count'] + batch - 1) // batch}")
PY
if [[ $WANDB_MODE == online ]]; then
    "$GR00T_PYTHON" -c 'import wandb; assert wandb.api.api_key, "run wandb login first"' || {
        echo "W&B online logging requested but no API key is configured" >&2
        exit 2
    }
fi

exec "$GR00T_PYTHON" "$PROJECT_ROOT/pipeline/bc_training/train_bc_h200.py" \
    --dataset-root "$PROJECT_ROOT/data/BC" \
    --train-split "$TRAIN_SPLIT" \
    --validation-split "$VALIDATION_SPLIT" \
    --bootstrap "$STUDENT_BOOTSTRAP" \
    --policy-spec "$POLICY_SPEC" \
    --output-root "$PROJECT_ROOT/checkpoints/students/BC" \
    --chunk-batch-size "$CHUNK_BATCH_SIZE" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project dagger \
    --wandb-group BC \
    --wandb-run-name BC \
    "$@"
