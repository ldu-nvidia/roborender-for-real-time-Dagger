#!/usr/bin/env bash
# Container-side online identity gate and writable W&B runtime for OCI learner.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON=${ROUND_DAGGER_H100_PYTHON:-/opt/viral-h100/bin/python}
OUTPUT=${ROUND_DAGGER_LEARNER_OUTPUT_DIR:?missing learner output}
WANDB_RUNTIME=${ROUND_DAGGER_WANDB_RUNTIME_DIR:?missing W&B runtime directory}
EXPECTED_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:?missing W&B expected email}
ENTITY=${ROUND_DAGGER_WANDB_ENTITY:?missing W&B entity}
SCOPE=${ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE:?missing W&B scope}

[[ $EXPECTED_EMAIL == ldu@nvidia.com ]] || fatal "W&B account binding changed"
[[ $ENTITY == nvidia ]] || fatal "W&B team entity binding changed"
[[ $SCOPE == team ]] || fatal "W&B authorization scope changed"
[[ ${WANDB_MODE:-online} == online ]] || fatal "production W&B mode must be online"
[[ $OUTPUT == /* && ! -L $OUTPUT ]] || fatal "learner output is unsafe"
[[ $WANDB_RUNTIME == /round_dagger_job/wandb && ! -L $WANDB_RUNTIME ]] || \
    fatal "W&B runtime must use the isolated job scratch mount"
[[ -x $PYTHON ]] || fatal "H100 Python is unavailable"

mkdir -p "$WANDB_RUNTIME/runtime" "$WANDB_RUNTIME/cache" "$WANDB_RUNTIME/config"
export WANDB_MODE=online
export WANDB_DIR="$WANDB_RUNTIME/runtime"
export WANDB_CACHE_DIR="$WANDB_RUNTIME/cache"
export WANDB_CONFIG_DIR="$WANDB_RUNTIME/config"

"$PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py" \
    --expected-email "$EXPECTED_EMAIL" --entity "$ENTITY" \
    --authorization-scope "$SCOPE" --format json >/dev/null

exec bash "$ROOT/scripts/round_dagger/run_h100_learner.sh"
