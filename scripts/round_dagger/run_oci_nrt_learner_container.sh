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
VERIFY_CAMPAIGN=${ROUND_DAGGER_CONTAINER_VERIFY_CAMPAIGN_ROOT:?missing verification campaign root}
VERIFY_INVENTORY=${ROUND_DAGGER_CONTAINER_VERIFY_CAMPAIGN_INVENTORY:?missing verification inventory}
SUCCESS_MARKER=${ROUND_DAGGER_CONTAINER_SUCCESS_MARKER:?missing container success marker}

[[ $EXPECTED_EMAIL == ldu@nvidia.com ]] || fatal "W&B account binding changed"
[[ $ENTITY == nvidia ]] || fatal "W&B team entity binding changed"
[[ $SCOPE == team ]] || fatal "W&B authorization scope changed"
[[ ${WANDB_MODE:-online} == online ]] || fatal "production W&B mode must be online"
[[ $OUTPUT == /* && ! -L $OUTPUT ]] || fatal "learner output is unsafe"
[[ $WANDB_RUNTIME == /round_dagger_job/wandb && ! -L $WANDB_RUNTIME ]] || \
    fatal "W&B runtime must use the isolated job scratch mount"
[[ $VERIFY_CAMPAIGN == /round_dagger_campaign_input && \
   -d $VERIFY_CAMPAIGN && ! -L $VERIFY_CAMPAIGN ]] || \
    fatal "verification campaign mount is unsafe"
[[ $VERIFY_INVENTORY == /round_dagger_campaign_input.inventory.json && \
   -f $VERIFY_INVENTORY && ! -L $VERIFY_INVENTORY ]] || \
    fatal "verification campaign inventory is unsafe"
[[ $SUCCESS_MARKER == /round_dagger_job/learner_success.json && \
   ! -L $SUCCESS_MARKER ]] || fatal "container success marker is unsafe"
[[ -x $PYTHON ]] || fatal "H100 Python is unavailable"

# Every validator importing round_dagger (and therefore NumPy/PyTorch) runs
# inside the pinned trainer image. The bare OCI compute host performs only
# shell/coreutils checks and has no Python dependency contract.
"$PYTHON" "$ROOT/scripts/round_dagger/verify_sealed_source_snapshot.py" \
    --root "$ROOT" \
    --expected-inventory-sha256 \
    "$ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/remote_campaign_transfer.py" verify \
    --root "$VERIFY_CAMPAIGN" --inventory "$VERIFY_INVENTORY" >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT" "$PYTHON" - \
    "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_POLICY_SPEC" \
    "$ROUND_DAGGER_POLICY_MANIFEST" "$ROUND_DAGGER_POLICY_SHA256" \
    "$ROUND_DAGGER_RUNTIME_POLICY_SHA256" <<'PY'
import sys
from round_dagger.policy_bundle import validate_policy_bundle_identity

validate_policy_bundle_identity(
    checkpoint=sys.argv[1], policy_spec=sys.argv[2], manifest=sys.argv[3],
    expected_checkpoint_sha256=sys.argv[4],
    expected_runtime_policy_sha256=sys.argv[5],
)
PY

mkdir -p "$WANDB_RUNTIME/runtime" "$WANDB_RUNTIME/cache" "$WANDB_RUNTIME/config"
export WANDB_MODE=online
export WANDB_DIR="$WANDB_RUNTIME/runtime"
export WANDB_CACHE_DIR="$WANDB_RUNTIME/cache"
export WANDB_CONFIG_DIR="$WANDB_RUNTIME/config"

"$PYTHON" "$ROOT/scripts/scaling/wandb_preflight.py" \
    --expected-email "$EXPECTED_EMAIL" --entity "$ENTITY" \
    --authorization-scope "$SCOPE" --format json >/dev/null

bash "$ROOT/scripts/round_dagger/run_h100_learner.sh"

printf -v ROUND_PADDED '%06d' "$((10#$ROUND_DAGGER_MACRO_ROUND))"
COMPLETION_ATTESTATION="$OUTPUT/learner_completion_round_${ROUND_PADDED}.json"
"$PYTHON" "$ROOT/scripts/round_dagger/learner_completion_attestation.py" adopt \
    --attestation "$COMPLETION_ATTESTATION" \
    --result "$ROUND_DAGGER_LEARNER_RESULT" \
    --success-marker "$SUCCESS_MARKER" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --parent-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
    --snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
    --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
    --runtime-container-sha256 "$ROUND_DAGGER_REMOTE_CONTAINER_SHA256" \
    --learner-contract-sha256 "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" \
    --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/remote_learner_success.py" verify \
    --marker "$SUCCESS_MARKER" \
    --result "$ROUND_DAGGER_LEARNER_RESULT" \
    --completion-attestation "$COMPLETION_ATTESTATION" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --learner-code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
echo "OCI_NRT_CONTAINER_COMPLETION_ADOPTED marker=$SUCCESS_MARKER"
