#!/bin/bash
# Synchronous RTX coordinator -> OCI-NRT 8xH100 learner bridge.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }
safe_absolute_path() {
    [[ $1 =~ ^/[A-Za-z0-9_./-]+$ && $1 != *//* && \
       $1 != */../* && $1 != */./* && $1 != */.. && $1 != */. ]]
}

if [[ ${ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH:-0} == 1 ]]; then
    ROOT=${ROUND_DAGGER_COORDINATOR_SOURCE_ROOT:?missing trusted coordinator source root}
    [[ $ROOT == /* && -d $ROOT && ! -L $ROOT && $(cd "$ROOT" && pwd -P) == "$ROOT" ]] || \
        fatal "trusted coordinator source root is unsafe"
    [[ ${ROUND_DAGGER_TRUSTED_HOOK_SOURCE_INVENTORY_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || \
        fatal "trusted coordinator source inventory SHA-256 is malformed"
else
    [[ ${DRY_RUN:-0} == 1 ]] || fatal "untrusted train hook dispatch"
    ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
fi
PYTHON=${ROUND_DAGGER_PYTHON:-python3}
REMOTE=${ROUND_DAGGER_REMOTE_HOST:-oci-nrt}
ACCOUNT=${ROUND_DAGGER_REMOTE_ACCOUNT:-general_cs_infra__ngcsre}
PARTITION=${ROUND_DAGGER_REMOTE_PARTITION:-batch_singlenode}
QOS=${ROUND_DAGGER_REMOTE_QOS:-normal}
STAGE_ROOT=${ROUND_DAGGER_REMOTE_STAGE_ROOT:-/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build}
REMOTE_SOURCE=${ROUND_DAGGER_REMOTE_SOURCE_ROOT:?set the exact sealed OCI source root}
REMOTE_SOURCE_INVENTORY_SHA=${ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256:?set the sealed source inventory SHA-256}
REMOTE_CONTAINER=${ROUND_DAGGER_REMOTE_CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh}
REMOTE_CONTAINER_SHA=${ROUND_DAGGER_REMOTE_CONTAINER_SHA256:-f3391b76a66182618f9e71b67e89b0af2d758ffe16afdb164760fcbb4860ebe6}
REMOTE_NETRC=${ROUND_DAGGER_REMOTE_WANDB_NETRC:-/home/ldu/.netrc}
REMOTE_SLURM_BIN=${ROUND_DAGGER_REMOTE_SLURM_BIN:-/cm/shared/apps/slurm/current/bin}
HANDOFF_ROOT=${ROUND_DAGGER_REMOTE_HANDOFF_ROOT:-/mnt/data2/ldu/roborender_round_dagger_handoff}
DRY_RUN=${DRY_RUN:-0}
POLL_SECONDS=${ROUND_DAGGER_REMOTE_POLL_SECONDS:-30}
MAX_ATTEMPTS=${ROUND_DAGGER_REMOTE_MAX_ATTEMPTS:-3}
# OCI allocations are administratively revoked at the four-hour boundary.
# Leave a 15-minute checkpoint/finalization margin and keep this immutable so
# an ambient scheduler option cannot silently recreate the failure mode.
LEARNER_WALLTIME=03:45:00

case "$DRY_RUN" in 0|1) ;; *) fatal "DRY_RUN must be exactly 0 or 1" ;; esac
[[ $REMOTE =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "remote host is malformed"
[[ $ACCOUNT == general_cs_infra__ngcsre ]] || fatal "OCI account is not approved"
[[ $PARTITION == batch_singlenode ]] || \
    fatal "OCI partition must be batch_singlenode"
[[ $QOS == normal ]] || fatal "OCI QOS must be normal"
[[ $REMOTE_SOURCE_INVENTORY_SHA =~ ^[0-9a-f]{64}$ ]] || \
    fatal "sealed source inventory SHA-256 is malformed"
[[ $REMOTE_CONTAINER_SHA =~ ^[0-9a-f]{64}$ ]] || \
    fatal "remote container SHA-256 is malformed"
for value in "$POLL_SECONDS" "$MAX_ATTEMPTS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || fatal "poll/attempt values must be positive integers"
done
[[ $LEARNER_WALLTIME == 03:45:00 ]] || fatal "learner walltime contract changed"
for path in "$STAGE_ROOT" "$REMOTE_SOURCE" "$REMOTE_CONTAINER" "$REMOTE_NETRC" \
            "$REMOTE_SLURM_BIN" "$HANDOFF_ROOT"; do
    safe_absolute_path "$path" || \
        fatal "unsafe absolute bridge path"
done
[[ -x $PYTHON || -x $(command -v "$PYTHON" 2>/dev/null || true) ]] || \
    fatal "round DAgger Python is not executable"

required=(
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_POLICY_CHECKPOINT
    ROUND_DAGGER_POLICY_SPEC ROUND_DAGGER_POLICY_MANIFEST
    ROUND_DAGGER_POLICY_SPEC_SHA256 ROUND_DAGGER_POLICY_MANIFEST_SHA256
    ROUND_DAGGER_POLICY_SHA256 ROUND_DAGGER_RUNTIME_POLICY_SHA256
    ROUND_DAGGER_REPLAY_SNAPSHOT ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256
    ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_OUTPUT_DIR
    ROUND_DAGGER_LEARNER_RESULT ROUND_DAGGER_LEARNER_CODE_SHA256
    ROUND_DAGGER_LEARNER_CONFIG_SHA256 ROUND_DAGGER_LEARNER_CONTRACT_SHA256
    ROUND_DAGGER_LEARNER_IDENTITY_SHA256
    ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256
    ROUND_DAGGER_REMOTE_CONTAINER_SHA256
    ROUND_DAGGER_TARGET_ENVIRONMENTS ROUND_DAGGER_COHORT_COUNT
    ROUND_DAGGER_ENVIRONMENTS_PER_COHORT
    ROUND_DAGGER_STEPS_PER_ENVIRONMENT
    ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND
)
for name in "${required[@]}"; do
    [[ -n ${!name:-} ]] || fatal "missing orchestrator environment: $name"
    [[ ${!name} != *$'\n'* && ${!name} != *$'\r'* && ${!name} != *,* ]] || \
        fatal "unsafe value for $name"
done
heldout_names=(
    ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT
    ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256
    ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256
    ROUND_DAGGER_HELDOUT_EVAL_INTERVAL
)
heldout_present=0
for name in "${heldout_names[@]}"; do
    [[ -z ${!name:-} ]] || heldout_present=$((heldout_present + 1))
done
((heldout_present == 0 || heldout_present == ${#heldout_names[@]})) || \
    fatal "held-out replay environment must be supplied all-or-none"
HELDOUT_ENABLED=0
if ((heldout_present)); then
    HELDOUT_ENABLED=1
    [[ $ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
        fatal "held-out replay snapshot file SHA-256 is malformed"
    [[ $ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
        fatal "held-out replay snapshot semantic SHA-256 is malformed"
    [[ $ROUND_DAGGER_HELDOUT_EVAL_INTERVAL == 100 ]] || \
        fatal "production held-out evaluation interval must be exactly 100"
fi
for name in ROUND_DAGGER_POLICY_SPEC_SHA256 ROUND_DAGGER_POLICY_MANIFEST_SHA256 \
            ROUND_DAGGER_POLICY_SHA256 ROUND_DAGGER_RUNTIME_POLICY_SHA256 \
            ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256 \
            ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_CODE_SHA256 \
            ROUND_DAGGER_LEARNER_CONFIG_SHA256 \
            ROUND_DAGGER_LEARNER_CONTRACT_SHA256 \
            ROUND_DAGGER_LEARNER_IDENTITY_SHA256 \
            ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ $REMOTE_SOURCE_INVENTORY_SHA == "$ROUND_DAGGER_LEARNER_CODE_SHA256" ]] || \
    fatal "learner code identity is not the sealed source inventory"
[[ $ROUND_DAGGER_MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "macro round is invalid"
[[ "$ROUND_DAGGER_TARGET_ENVIRONMENTS:$ROUND_DAGGER_COHORT_COUNT:$ROUND_DAGGER_ENVIRONMENTS_PER_COHORT:$ROUND_DAGGER_STEPS_PER_ENVIRONMENT:$ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND" == \
   "256:4:64:512:131072" ]] || fatal "production topology is not exactly 4x64x512"

CAMPAIGN_ROOT=$ROUND_DAGGER_CAMPAIGN_ROOT
[[ $CAMPAIGN_ROOT == /* && -d $CAMPAIGN_ROOT && ! -L $CAMPAIGN_ROOT ]] || \
    fatal "campaign root is missing or unsafe"
ROUND_DAGGER_LEARNER_CONTRACT=${ROUND_DAGGER_LEARNER_CONTRACT:-$CAMPAIGN_ROOT/learner_contract.json}
[[ $ROUND_DAGGER_LEARNER_CONTRACT == "$CAMPAIGN_ROOT/learner_contract.json" ]] || \
    fatal "learner contract is not the canonical campaign-owned file"
[[ -f $ROUND_DAGGER_LEARNER_CONTRACT && \
   ! -L $ROUND_DAGGER_LEARNER_CONTRACT ]] || \
    fatal "campaign-owned learner contract is missing or unsafe"
[[ $(stat -c %a -- "$ROUND_DAGGER_LEARNER_CONTRACT") == 444 ]] || \
    fatal "campaign-owned learner contract mode must be exactly 0444"
[[ $(sha256sum -- "$ROUND_DAGGER_LEARNER_CONTRACT" | awk '{print $1}') == \
   "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" ]] || \
    fatal "campaign-owned learner contract SHA mismatch"
for path in "$CAMPAIGN_ROOT" "$ROUND_DAGGER_POLICY_CHECKPOINT" \
            "$ROUND_DAGGER_POLICY_SPEC" "$ROUND_DAGGER_POLICY_MANIFEST" \
            "$ROUND_DAGGER_REPLAY_SNAPSHOT" "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" \
            "$ROUND_DAGGER_LEARNER_RESULT"; do
    safe_absolute_path "$path" || fatal "unsafe learner mount/input path"
done
if ((HELDOUT_ENABLED)); then
    safe_absolute_path "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT" || \
        fatal "unsafe held-out replay snapshot path"
    [[ $ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT == "$CAMPAIGN_ROOT"/* ]] || \
        fatal "held-out replay snapshot must be inside the campaign"
    [[ -f $ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT && \
       ! -L $ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT ]] || \
        fatal "held-out replay snapshot is missing or unsafe"
    [[ $(sha256sum "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT" | awk '{print $1}') == \
       "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256" ]] || \
        fatal "held-out replay snapshot file SHA mismatch"
    PYTHONPATH="$ROOT" "$PYTHON" - \
        "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT" \
        "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256" \
        "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256" \
        "$ROUND_DAGGER_CAMPAIGN_ID" "$CAMPAIGN_ROOT" <<'PY'
import sys
from pathlib import Path
from round_dagger.replay import load_heldout_snapshot

path, file_sha, semantic_sha, campaign_id, campaign_root = sys.argv[1:]
snapshot = load_heldout_snapshot(path, expected_file_sha256=file_sha)
if snapshot["snapshot_sha256"] != semantic_sha:
    raise SystemExit("held-out replay snapshot semantic digest mismatch")
if snapshot["campaign_id"] != campaign_id:
    raise SystemExit("held-out replay snapshot campaign mismatch")
root = Path(campaign_root).resolve(strict=True)
referenced = [
    *(Path(item["cohort_manifest_path"]) for item in snapshot["cohorts"]),
    *(Path(item["shard_manifest_path"]) for item in snapshot["resolved_shards"]),
    *(Path(item["payload_path"]) for item in snapshot["resolved_shards"]),
]
for referenced_path in referenced:
    resolved = referenced_path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit("held-out replay artifact is outside the campaign") from exc
    if relative.parts and relative.parts[0] in {"actors", "operator"}:
        raise SystemExit(
            "held-out replay artifact would be excluded from learner projection"
        )
PY
fi
for file in "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_POLICY_SPEC" \
            "$ROUND_DAGGER_POLICY_MANIFEST" "$ROUND_DAGGER_REPLAY_SNAPSHOT"; do
    [[ -f $file && ! -L $file ]] || fatal "required learner input is missing or unsafe"
done
[[ $ROUND_DAGGER_POLICY_SPEC == "${ROUND_DAGGER_POLICY_CHECKPOINT}.policy_spec.json" && \
   $ROUND_DAGGER_POLICY_MANIFEST == "${ROUND_DAGGER_POLICY_CHECKPOINT}.manifest.json" ]] || \
    fatal "active policy sidecars are not canonically co-located"
[[ $(sha256sum "$ROUND_DAGGER_POLICY_CHECKPOINT" | awk '{print $1}') == \
   "$ROUND_DAGGER_POLICY_SHA256" ]] || fatal "active checkpoint SHA mismatch"
[[ $(sha256sum "$ROUND_DAGGER_POLICY_SPEC" | awk '{print $1}') == \
   "$ROUND_DAGGER_POLICY_SPEC_SHA256" ]] || fatal "active policy spec SHA mismatch"
[[ $(sha256sum "$ROUND_DAGGER_POLICY_MANIFEST" | awk '{print $1}') == \
   "$ROUND_DAGGER_POLICY_MANIFEST_SHA256" ]] || fatal "active policy manifest SHA mismatch"
[[ $(sha256sum "$ROUND_DAGGER_REPLAY_SNAPSHOT" | awk '{print $1}') == \
   "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" ]] || fatal "snapshot file SHA mismatch"
[[ $ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256 == "$ROUND_DAGGER_POLICY_SHA256" ]] || \
    fatal "learner lineage is not the active checkpoint"
[[ $ROUND_DAGGER_LEARNER_OUTPUT_DIR == "$CAMPAIGN_ROOT"/* ]] || \
    fatal "learner output must be inside the campaign"
[[ $ROUND_DAGGER_LEARNER_RESULT == "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"/* ]] || \
    fatal "learner result must be inside learner output"
[[ ! -e $ROUND_DAGGER_LEARNER_RESULT && ! -L $ROUND_DAGGER_LEARNER_RESULT ]] || \
    fatal "learner result already exists; coordinator must verify it"

ROUND_DAGGER_BASE_POLICY_SPEC=${ROUND_DAGGER_BASE_POLICY_SPEC:-${ROUND_DAGGER_POLICY_CHECKPOINT}.policy_spec.json}
[[ $ROUND_DAGGER_BASE_POLICY_SPEC == "$ROUND_DAGGER_POLICY_SPEC" ]] || \
    fatal "base policy spec differs from the active bundle"
PYTHONPATH="$ROOT" "$PYTHON" - "$ROUND_DAGGER_POLICY_CHECKPOINT" \
    "$ROUND_DAGGER_POLICY_SPEC" "$ROUND_DAGGER_POLICY_MANIFEST" \
    "$ROUND_DAGGER_POLICY_SHA256" "$ROUND_DAGGER_RUNTIME_POLICY_SHA256" <<'PY'
import sys
from round_dagger.policy_bundle import validate_policy_bundle_identity

validate_policy_bundle_identity(
    checkpoint=sys.argv[1], policy_spec=sys.argv[2], manifest=sys.argv[3],
    expected_checkpoint_sha256=sys.argv[4],
    expected_runtime_policy_sha256=sys.argv[5],
)
PY
ROUND_DAGGER_LEARNER_BATCH_SIZE=${ROUND_DAGGER_LEARNER_BATCH_SIZE:-8}
ROUND_DAGGER_LEARNER_MAX_STEPS=${ROUND_DAGGER_LEARNER_MAX_STEPS:-1200}
ROUND_DAGGER_LEARNER_SAVE_EVERY=${ROUND_DAGGER_LEARNER_SAVE_EVERY:-5}
ROUND_DAGGER_LEARNER_LEARNING_RATE=${ROUND_DAGGER_LEARNER_LEARNING_RATE:-2.5e-5}
ROUND_DAGGER_LEARNER_WEIGHT_DECAY=${ROUND_DAGGER_LEARNER_WEIGHT_DECAY:-0.01}
ROUND_DAGGER_LEARNER_SEED=${ROUND_DAGGER_LEARNER_SEED:-17}
ROUND_DAGGER_LEARNER_PARTITION_SEED=${ROUND_DAGGER_LEARNER_PARTITION_SEED:-23}
[[ "$ROUND_DAGGER_LEARNER_BATCH_SIZE:$ROUND_DAGGER_LEARNER_MAX_STEPS:$ROUND_DAGGER_LEARNER_SAVE_EVERY:$ROUND_DAGGER_LEARNER_LEARNING_RATE:$ROUND_DAGGER_LEARNER_WEIGHT_DECAY:$ROUND_DAGGER_LEARNER_SEED:$ROUND_DAGGER_LEARNER_PARTITION_SEED" == \
   "8:1200:5:2.5e-5:0.01:17:23" ]] || fatal "production learner hyperparameters are immutable"
ROUND_DAGGER_WANDB_PROJECT=${ROUND_DAGGER_WANDB_PROJECT:-roborender-round-dagger}
ROUND_DAGGER_WANDB_ENTITY=${ROUND_DAGGER_WANDB_ENTITY:-nvidia}
ROUND_DAGGER_WANDB_EXPECTED_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE=${ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE:-team}
printf -v ROUND_PADDED '%06d' "$((10#$ROUND_DAGGER_MACRO_ROUND))"
ROUND_DECIMAL=$((10#$ROUND_DAGGER_MACRO_ROUND))
EXPECTED_LEARNER_IDENTITY_SHA256=$(
    printf '%s\n' \
        'fast_roborender.round_dagger.learner_run_identity.v2' \
        "campaign_id=$ROUND_DAGGER_CAMPAIGN_ID" \
        "macro_round=$ROUND_DECIMAL" \
        "learner_code_sha256=$ROUND_DAGGER_LEARNER_CODE_SHA256" \
        "learner_config_sha256=$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
        "replay_snapshot_file_sha256=$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
        "replay_snapshot_sha256=$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
        "parent_checkpoint_sha256=$ROUND_DAGGER_POLICY_SHA256" \
        "parent_runtime_policy_sha256=$ROUND_DAGGER_RUNTIME_POLICY_SHA256" \
        "runtime_container_sha256=$REMOTE_CONTAINER_SHA" \
        "learner_contract_sha256=$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" | \
        sha256sum | awk '{print $1}'
)
[[ $ROUND_DAGGER_LEARNER_IDENTITY_SHA256 == \
   "$EXPECTED_LEARNER_IDENTITY_SHA256" ]] || \
    fatal "learner identity SHA-256 does not bind the exact semantic run"
EXPECTED_WANDB_RUN_ID=rrd-m${ROUND_PADDED}-id${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}
ROUND_DAGGER_WANDB_RUN_ID=${ROUND_DAGGER_WANDB_RUN_ID:-$EXPECTED_WANDB_RUN_ID}
[[ $ROUND_DAGGER_WANDB_ENTITY == nvidia ]] || fatal "W&B entity must be nvidia"
[[ $ROUND_DAGGER_WANDB_EXPECTED_EMAIL == ldu@nvidia.com && \
   $ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE == team ]] || \
    fatal "W&B authorization identity changed"
[[ $ROUND_DAGGER_WANDB_RUN_ID == "$EXPECTED_WANDB_RUN_ID" ]] || \
    fatal "W&B run ID is not bound to the exact learner identity"
for token in "$ROUND_DAGGER_CAMPAIGN_ID" "$ROUND_DAGGER_WANDB_PROJECT" \
             "$ROUND_DAGGER_WANDB_RUN_ID"; do
    [[ $token =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "unsafe campaign/W&B token"
    ((${#token} <= 128)) || fatal "campaign/W&B token exceeds 128 characters"
done

# The production campaign must prove the new fixed-256, optimizer-step-100
# quality telemetry and strict restart path on the exact round-0 replay before
# its 1200-step learner can start.  Keep this inside the existing train hook:
# the orchestrator invokes us only after it has durably committed INGESTED, so
# any validation failure leaves a resumable replay frontier and never forces a
# second actor collection.
HELDOUT100_GATE_ENABLED=0
HELDOUT100_CONTRACT=
HELDOUT100_CONTRACT_SHA256=
HELDOUT100_CONFIG_SHA256=
HELDOUT100_IDENTITY_SHA256=
HELDOUT100_WANDB_RUN_ID=
HELDOUT100_BINDING_SHA256=
HELDOUT100_PRODUCTION_CONTRACT_SHA256=
if [[ ${ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH:-0} == 1 ]]; then
    if [[ ${ROUND_DAGGER_REQUIRE_HELDOUT100_GATE+set} == set ]]; then
        [[ $ROUND_DAGGER_REQUIRE_HELDOUT100_GATE == 1 ]] || \
            fatal "trusted production dispatch cannot disable the heldout100 gate"
    fi
    # This is a production safety/quality gate, not an operator-tunable knob.
    # An inherited environment must never weaken a trusted coordinator dispatch.
    HELDOUT100_REQUIRED=1
else
    HELDOUT100_REQUIRED=${ROUND_DAGGER_REQUIRE_HELDOUT100_GATE:-0}
fi
[[ $HELDOUT100_REQUIRED == 0 || $HELDOUT100_REQUIRED == 1 ]] || \
    fatal "heldout100 gate requirement must be 0 or 1"
if ((HELDOUT100_REQUIRED && ! HELDOUT_ENABLED)); then
    fatal "trusted production dispatch requires fixed-256 held-out evaluation"
fi
if [[ $ROUND_DECIMAL == 0 ]] && ((HELDOUT100_REQUIRED)); then
    HELDOUT100_GATE_ENABLED=1
    HELDOUT100_PRODUCTION_CONTRACT_SHA256=$ROUND_DAGGER_LEARNER_CONTRACT_SHA256
    HELDOUT100_CONTRACT="$ROOT/configs/round_dagger/validation_learner_1x8h100_heldout100.json"
    [[ -f $HELDOUT100_CONTRACT && ! -L $HELDOUT100_CONTRACT ]] || \
        fatal "round-0 heldout100 validation contract is missing or unsafe"
    HELDOUT100_CONTRACT_SHA256=$(sha256sum "$HELDOUT100_CONTRACT" | awk '{print $1}')
    PYTHONPATH="$ROOT" "$PYTHON" \
        "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" contract \
        --path "$HELDOUT100_CONTRACT" \
        --sha256 "$HELDOUT100_CONTRACT_SHA256" \
        --source-root "$ROOT" >/dev/null
    HELDOUT100_BINDING_REPORT=$(PYTHONPATH="$ROOT" "$PYTHON" \
        "$ROOT/scripts/round_dagger/h100_heldout100_validation.py" binding \
        --contract "$HELDOUT100_CONTRACT" \
        --contract-sha256 "$HELDOUT100_CONTRACT_SHA256" \
        --source-root "$ROOT" --campaign-root "$CAMPAIGN_ROOT" \
        --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
        --heldout-snapshot "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT" \
        --heldout-snapshot-file-sha256 \
            "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256" \
        --heldout-snapshot-sha256 \
            "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256" \
        --production-contract-sha256 \
            "$HELDOUT100_PRODUCTION_CONTRACT_SHA256")
    HELDOUT100_BINDING_SHA256=$(printf '%s\n' "$HELDOUT100_BINDING_REPORT" | \
        "$PYTHON" -c 'import json,re,sys; value=json.load(sys.stdin).get("binding_sha256"); assert isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value); print(value)') || \
        fatal "cannot derive the fresh-campaign heldout100 binding identity"

    HELDOUT100_BUNDLE_REPORT=$(PYTHONPATH="$ROOT" "$PYTHON" \
        "$ROOT/scripts/round_dagger/prepare_gr00t_learner_bundle.py" \
        --base-policy-spec "$ROUND_DAGGER_BASE_POLICY_SPEC" \
        --parent-checkpoint "$ROUND_DAGGER_POLICY_CHECKPOINT" \
        --expected-parent-checkpoint-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
        --runtime-policy-sha256 "$ROUND_DAGGER_RUNTIME_POLICY_SHA256" \
        --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
        --macro-round "$ROUND_DECIMAL" \
        --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
        --batch-size 1 --max-steps 100 --save-every 5 \
        --learning-rate 2.5e-5 --weight-decay 0.01 \
        --seed 17 --partition-seed 23 \
        --heldout-snapshot "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT" \
        --heldout-eval-interval 100 \
        --output "$CAMPAIGN_ROOT/rounds/macro_round_${ROUND_PADDED}/heldout100_bundle.dry-run.json" \
        --dry-run)
    HELDOUT100_CONFIG_SHA256=$(printf '%s\n' "$HELDOUT100_BUNDLE_REPORT" | \
        "$PYTHON" -c 'import json,re,sys; value=json.load(sys.stdin).get("bundle_sha256"); assert isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value); print(value)') || \
        fatal "cannot derive the round-0 heldout100 learner bundle identity"
    mapfile -t HELDOUT100_IDENTITY < <(
        "$PYTHON" "$ROOT/round_dagger/learner_identity.py" \
            "$ROUND_DAGGER_CAMPAIGN_ID" "$ROUND_DECIMAL" \
            "$ROUND_DAGGER_LEARNER_CODE_SHA256" "$HELDOUT100_CONFIG_SHA256" \
            "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
            "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
            "$ROUND_DAGGER_POLICY_SHA256" "$ROUND_DAGGER_RUNTIME_POLICY_SHA256" \
            "$REMOTE_CONTAINER_SHA" "$HELDOUT100_CONTRACT_SHA256"
    )
    ((${#HELDOUT100_IDENTITY[@]} == 2)) || \
        fatal "round-0 heldout100 learner identity derivation failed"
    HELDOUT100_IDENTITY_SHA256=${HELDOUT100_IDENTITY[0]}
    HELDOUT100_WANDB_RUN_ID="${HELDOUT100_IDENTITY[1]}-b${HELDOUT100_BINDING_SHA256:0:32}"
    [[ $HELDOUT100_IDENTITY_SHA256 =~ ^[0-9a-f]{64}$ && \
       $HELDOUT100_WANDB_RUN_ID == \
       "rrd-m000000-id${HELDOUT100_IDENTITY_SHA256}-b${HELDOUT100_BINDING_SHA256:0:32}" ]] || \
        fatal "round-0 heldout100 learner identity is malformed"
fi

LEARNER_NAMESPACE=identity_${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}
JOB_NAME="rr-rd-${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}"
LOCAL_HANDOFF="$HANDOFF_ROOT/$ROUND_DAGGER_CAMPAIGN_ID/macro_round_$ROUND_PADDED/$LEARNER_NAMESPACE"
LOCAL_PROJECTION="$LOCAL_HANDOFF/campaign_input"
LOCAL_INVENTORY="$LOCAL_HANDOFF/campaign_input.inventory.json"
REMOTE_RUN="$STAGE_ROOT/round_runs/$ROUND_DAGGER_CAMPAIGN_ID/macro_round_$ROUND_PADDED/$LEARNER_NAMESPACE"
REMOTE_STAGE="$REMOTE_RUN/.incoming-campaign-input"
REMOTE_CAMPAIGN="$REMOTE_RUN/campaign_input"
REMOTE_INVENTORY="$REMOTE_RUN/campaign_input.inventory.json"
REMOTE_OUTPUT="$REMOTE_RUN/learner_output"
REMOTE_OUTPUT_INVENTORY="$REMOTE_RUN/learner_output.inventory.json"
REMOTE_JOB_ROOT="$REMOTE_RUN/job"
REMOTE_SUCCESS_MARKER="$REMOTE_JOB_ROOT/learner_success.json"
LOCAL_SUCCESS_MARKER="$LOCAL_HANDOFF/remote_learner_success.json"
LOCAL_OUTPUT_INVENTORY="$LOCAL_HANDOFF/learner_output.inventory.json"
TRANSFER_TOOL="$ROOT/scripts/round_dagger/remote_campaign_transfer.py"
HELDOUT100_OUTPUT_RELATIVE=-
if ((HELDOUT100_GATE_ENABLED)); then
    HELDOUT100_NAMESPACE=identity_${HELDOUT100_IDENTITY_SHA256}_binding_${HELDOUT100_BINDING_SHA256}
    HELDOUT100_REMOTE_ROOT="$REMOTE_RUN/heldout100_gate/$HELDOUT100_NAMESPACE"
    HELDOUT100_REMOTE_STAGE="$HELDOUT100_REMOTE_ROOT/.incoming-campaign-input"
    HELDOUT100_REMOTE_CAMPAIGN="$HELDOUT100_REMOTE_ROOT/campaign_input"
    HELDOUT100_REMOTE_JOB="$HELDOUT100_REMOTE_ROOT/job"
    HELDOUT100_REMOTE_LOG="$HELDOUT100_REMOTE_ROOT/log"
    HELDOUT100_LOCAL_OUTPUT="$CAMPAIGN_ROOT/rounds/macro_round_${ROUND_PADDED}/heldout100_gate/$HELDOUT100_NAMESPACE"
    HELDOUT100_OUTPUT_RELATIVE=${HELDOUT100_LOCAL_OUTPUT#"$CAMPAIGN_ROOT"/}
    # The dedicated smoke contract requires the writable bind source to be
    # the canonical nested target in the otherwise read-only campaign mount.
    HELDOUT100_REMOTE_OUTPUT="$HELDOUT100_REMOTE_CAMPAIGN/$HELDOUT100_OUTPUT_RELATIVE"
    HELDOUT100_LOCAL_EVIDENCE="$LOCAL_HANDOFF/heldout100_gate_evidence"
    HELDOUT100_JOB_NAME="rr-heldout100-${HELDOUT100_IDENTITY_SHA256:0:12}"
fi

if ((DRY_RUN)); then
    printf 'DRY_RUN identity=%s wandb_run_id=%s job_name=%s local_handoff=%q remote_run=%q\n' \
        "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" "$ROUND_DAGGER_WANDB_RUN_ID" \
        "$JOB_NAME" "$LOCAL_HANDOFF" "$REMOTE_RUN"
    printf 'DRY_RUN local_campaign=%q projection=learner-inputs-only\n' \
        "$CAMPAIGN_ROOT"
    printf 'DRY_RUN remote_host=%s account=%s partition=%s gpus=8 updates=%s save_every=5 walltime=%s\n' \
        "$REMOTE" "$ACCOUNT" "$PARTITION" "$ROUND_DAGGER_LEARNER_MAX_STEPS" \
        "$LEARNER_WALLTIME"
    printf 'DRY_RUN heldout_eval=%s cadence=%s\n' \
        "$HELDOUT_ENABLED" "${ROUND_DAGGER_HELDOUT_EVAL_INTERVAL:-disabled}"
    if ((HELDOUT100_GATE_ENABLED)); then
        printf 'DRY_RUN heldout100_gate=required-before-1200 identity=%s wandb_run_id=%s steps=100 stop=50 heldout_steps=0,100\n' \
            "$HELDOUT100_IDENTITY_SHA256" "$HELDOUT100_WANDB_RUN_ID"
    fi
    printf 'DRY_RUN mount %q at %q read-only; mount %q at %q writable\n' \
        "$REMOTE_CAMPAIGN" "$CAMPAIGN_ROOT" "$REMOTE_OUTPUT" \
        "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"
    echo "OCI_NRT_ROUND_LEARNER_BRIDGE_DRY_RUN_OK"
    exit 0
fi

mkdir -p "$LOCAL_HANDOFF"
[[ -d $LOCAL_HANDOFF && ! -L $LOCAL_HANDOFF ]] || fatal "local handoff root is unsafe"
exec 9>"$LOCAL_HANDOFF/.bridge.lock"
flock -n 9 || fatal "another bridge process owns this campaign/round handoff"
if [[ ! -d $LOCAL_PROJECTION ]]; then
    INCOMING="$LOCAL_HANDOFF/.incoming-campaign-input"
    if [[ -e $INCOMING || -L $INCOMING ]]; then
        [[ -d $INCOMING && ! -L $INCOMING ]] || fatal "unsafe local incoming projection"
    else
        mkdir "$INCOMING"
    fi
    rsync --archive --links --delete --delete-excluded \
        --link-dest="$CAMPAIGN_ROOT" \
        --exclude='/actors/' \
        --exclude='/operator/' \
        --exclude='/.coordinator_state.lock' \
        --exclude='/rounds/*/orchestration_run/' \
        -- "$CAMPAIGN_ROOT/" "$INCOMING/"
    "$PYTHON" "$TRANSFER_TOOL" validate-projection --root "$INCOMING" >/dev/null
    "$PYTHON" "$TRANSFER_TOOL" inventory --root "$INCOMING" --output "$LOCAL_INVENTORY"
    mv "$INCOMING" "$LOCAL_PROJECTION"
fi
"$PYTHON" "$TRANSFER_TOOL" validate-projection --root "$LOCAL_PROJECTION" >/dev/null
"$PYTHON" "$TRANSFER_TOOL" verify --root "$LOCAL_PROJECTION" \
    --inventory "$LOCAL_INVENTORY" >/dev/null
for source in "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_POLICY_SPEC" \
              "$ROUND_DAGGER_POLICY_MANIFEST" "$ROUND_DAGGER_REPLAY_SNAPSHOT" \
              "$ROUND_DAGGER_LEARNER_CONTRACT"; do
    [[ $source == "$CAMPAIGN_ROOT"/* ]] || fatal "learner input is outside campaign"
    relative=${source#"$CAMPAIGN_ROOT"/}
    projected="$LOCAL_PROJECTION/$relative"
    [[ -f $projected && ! -L $projected ]] || fatal "projection omitted learner input"
    [[ $(sha256sum "$source" | awk '{print $1}') == \
       $(sha256sum "$projected" | awk '{print $1}') ]] || \
        fatal "projected learner input differs from canonical campaign"
done
if ((HELDOUT_ENABLED)); then
    source=$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT
    relative=${source#"$CAMPAIGN_ROOT"/}
    projected="$LOCAL_PROJECTION/$relative"
    [[ -f $projected && ! -L $projected ]] || \
        fatal "projection omitted held-out replay snapshot"
    [[ $(sha256sum "$source" | awk '{print $1}') == \
       $(sha256sum "$projected" | awk '{print $1}') ]] || \
        fatal "projected held-out replay snapshot differs from canonical campaign"
fi

ssh -o BatchMode=yes "$REMOTE" env PYTHONDONTWRITEBYTECODE=1 \
    python3 "$REMOTE_SOURCE/scripts/round_dagger/verify_sealed_source_snapshot.py" \
    --root "$REMOTE_SOURCE" \
    --expected-inventory-sha256 "$REMOTE_SOURCE_INVENTORY_SHA" >/dev/null

ssh -o BatchMode=yes "$REMOTE" bash -s -- \
    "$REMOTE_RUN" "$REMOTE_STAGE" "$REMOTE_JOB_ROOT" \
    "$REMOTE_SLURM_BIN" <<'REMOTE_PREPARE'
set -Eeuo pipefail
run=$1; stage=$2; job_root=$3; slurm_bin=$4
case "$stage" in "$run"/.incoming-campaign-input) ;; *) exit 2 ;; esac
case "$job_root" in "$run"/job) ;; *) exit 2 ;; esac
mkdir -p "$run" "$job_root"
[ -x "$slurm_bin/srun" ] && [ -x "$slurm_bin/sbatch" ] && \
    [ -x "$slurm_bin/squeue" ] && [ -x "$slurm_bin/sacct" ] || exit 2
if [ -e "$stage" ]; then
    [ -d "$stage" ] && [ ! -L "$stage" ] || exit 2
else
    mkdir "$stage"
fi
REMOTE_PREPARE
rsync --archive --no-links --protect-args --partial --append-verify \
    --delete-delay -- "$LOCAL_PROJECTION/" "$REMOTE:$REMOTE_STAGE/"
rsync --archive --no-links --protect-args -- "$LOCAL_INVENTORY" \
    "$REMOTE:$REMOTE_INVENTORY"
ssh -o BatchMode=yes "$REMOTE" env PYTHONDONTWRITEBYTECODE=1 \
    python3 "$REMOTE_SOURCE/scripts/round_dagger/remote_campaign_transfer.py" publish \
    --stage "$REMOTE_STAGE" --destination "$REMOTE_CAMPAIGN" \
    --inventory "$REMOTE_INVENTORY" >/dev/null
REMOTE_LEARNER_CONTRACT_SHA=$(
    ssh -o BatchMode=yes "$REMOTE" sha256sum -- \
        "$REMOTE_CAMPAIGN/learner_contract.json" | awk '{print $1}'
) || fatal "cannot hash projected remote learner contract"
[[ $REMOTE_LEARNER_CONTRACT_SHA == "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" ]] || \
    fatal "projected remote learner contract SHA mismatch"
HELDOUT100_NEEDS_SEAL=0
if ((HELDOUT100_GATE_ENABLED)) && \
   [[ ! -e $HELDOUT100_LOCAL_EVIDENCE && ! -L $HELDOUT100_LOCAL_EVIDENCE ]]; then
    # Never write smoke artifacts into the production campaign projection:
    # its immutable inventory is re-verified inside the 1200-step container.
    # Create an identity-scoped hard-link projection for the smoke instead.
    smoke_projection_state=$(ssh -o BatchMode=yes "$REMOTE" bash -s -- \
        "$HELDOUT100_REMOTE_ROOT" "$HELDOUT100_REMOTE_STAGE" \
        "$HELDOUT100_REMOTE_CAMPAIGN" \
        "$HELDOUT100_OUTPUT_RELATIVE" <<'REMOTE_HELDOUT100_PROJECTION_PREPARE'
set -Eeuo pipefail
root=$1; stage=$2; destination=$3; output_relative=$4
case "$stage" in "$root"/.incoming-campaign-input) ;; *) exit 2 ;; esac
case "$destination" in "$root"/campaign_input) ;; *) exit 2 ;; esac
case "$output_relative" in /*|../*|*/../*|*/./*|*/..|*/.) exit 2 ;; esac
mkdir -p "$root"
if [ -e "$destination" ] || [ -L "$destination" ]; then
    [ -d "$destination" ] && [ ! -L "$destination" ] && \
        [ ! -e "$stage" ] && [ ! -L "$stage" ] || exit 2
    if [ -e "$destination/$output_relative" ] || \
       [ -L "$destination/$output_relative" ]; then
        [ -d "$destination/$output_relative" ] && \
            [ ! -L "$destination/$output_relative" ] || exit 2
        printf '%s\n' existing-output
    else
        printf '%s\n' existing-clean
    fi
else
    if [ -e "$stage" ] || [ -L "$stage" ]; then
        [ -d "$stage" ] && [ ! -L "$stage" ] || exit 2
    else
        mkdir "$stage"
    fi
    printf '%s\n' stage
fi
REMOTE_HELDOUT100_PROJECTION_PREPARE
    ) || fatal "cannot prepare isolated heldout100 campaign projection"
    case "$smoke_projection_state" in
        stage)
            rsync --archive --no-links --protect-args --partial --append-verify \
                --delete-delay --link-dest="$REMOTE_CAMPAIGN" -- \
                "$LOCAL_PROJECTION/" "$REMOTE:$HELDOUT100_REMOTE_STAGE/"
            ssh -o BatchMode=yes "$REMOTE" env PYTHONDONTWRITEBYTECODE=1 \
                python3 "$REMOTE_SOURCE/scripts/round_dagger/remote_campaign_transfer.py" \
                publish --stage "$HELDOUT100_REMOTE_STAGE" \
                --destination "$HELDOUT100_REMOTE_CAMPAIGN" \
                --inventory "$REMOTE_INVENTORY" >/dev/null
            HELDOUT100_NEEDS_SEAL=1
            ;;
        existing-clean) HELDOUT100_NEEDS_SEAL=1 ;;
        existing-output) ;;
        *) fatal "isolated heldout100 campaign projection state is malformed" ;;
    esac
fi
LEARNER_OUTPUT_REL=${ROUND_DAGGER_LEARNER_OUTPUT_DIR#"$CAMPAIGN_ROOT"/}
[[ $LEARNER_OUTPUT_REL != "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" && \
   -n $LEARNER_OUTPUT_REL && $LEARNER_OUTPUT_REL != /* && \
   $LEARNER_OUTPUT_REL != ../* && $LEARNER_OUTPUT_REL != */../* ]] || \
    fatal "learner output is not a safe campaign-relative directory"
ssh -o BatchMode=yes "$REMOTE" bash -s -- \
    "$REMOTE_CAMPAIGN" "$LEARNER_OUTPUT_REL" \
    - <<'REMOTE_SEAL'
set -Eeuo pipefail
root=$1; learner_relative=$2; heldout_relative=$3
[ -d "$root" ] && [ ! -L "$root" ] || exit 2
# Pyxis requires a nested writable bind target to exist below the outer
# read-only campaign mount.  Create it only in the remote projection; the
# canonical RTX destination remains absent for atomic publication.
mkdir -p "$root/$learner_relative"
[ -d "$root/$learner_relative" ] && [ ! -L "$root/$learner_relative" ] || exit 2
if [ "$heldout_relative" != - ]; then
    case "$heldout_relative" in
        /*|../*|*/../*|*/./*|*/..|*/.) exit 2 ;;
    esac
    mkdir -p "$root/$heldout_relative"
    [ -d "$root/$heldout_relative" ] && \
        [ ! -L "$root/$heldout_relative" ] || exit 2
fi
find "$root" -type f -exec chmod a-w {} +
find "$root" -type d -exec chmod a-w {} +
# The heldout100 output is both the nested Pyxis target and its writable bind
# source.  Re-open only that empty identity-scoped directory; every input file
# and every other campaign directory remains immutable.
if [ "$heldout_relative" != - ]; then
    chmod u+rwx "$root/$heldout_relative"
    [ -z "$(find "$root" -path "$root/$heldout_relative" -prune -o \
        -perm /222 -print -quit)" ]
else
    [ -z "$(find "$root" -perm /222 -print -quit)" ]
fi
REMOTE_SEAL
if ((HELDOUT100_NEEDS_SEAL)); then
    ssh -o BatchMode=yes "$REMOTE" bash -s -- \
        "$HELDOUT100_REMOTE_CAMPAIGN" \
        "$HELDOUT100_OUTPUT_RELATIVE" <<'REMOTE_HELDOUT100_SEAL'
set -Eeuo pipefail
root=$1; output_relative=$2
[ -d "$root" ] && [ ! -L "$root" ] || exit 2
case "$output_relative" in
    /*|../*|*/../*|*/./*|*/..|*/.) exit 2 ;;
esac
mkdir -p "$root/$output_relative"
[ -d "$root/$output_relative" ] && [ ! -L "$root/$output_relative" ] || exit 2
find "$root" -type f -exec chmod a-w {} +
find "$root" -type d -exec chmod a-w {} +
chmod u+rwx "$root/$output_relative"
[ -z "$(find "$root" -path "$root/$output_relative" -prune -o \
    -perm /222 -print -quit)" ]
REMOTE_HELDOUT100_SEAL
fi

# BEGIN_HELDOUT100_GATE_EVIDENCE_VERIFIER
verify_heldout100_gate_evidence() {
    "$PYTHON" - "$@" <<'PY'
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path


def fail(message):
    raise SystemExit(f"heldout100 gate evidence rejected: {message}")


def strict_object(path, label):
    try:
        info = path.lstat()
    except OSError as exc:
        fail(f"{label} is unavailable: {exc}")
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        fail(f"{label} is not a regular non-symlink file")
    if info.st_mode & 0o222:
        fail(f"{label} is mutable")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value):
        fail(f"{label} contains non-finite constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"{label} is malformed: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object")
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def expect(condition, message):
    if not condition:
        fail(message)


def exact_values(document, expected, label):
    for key, value in expected.items():
        expect(document.get(key) == value, f"{label} {key} differs")


def finite(value, label, *, nonnegative=False):
    expect(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    number = float(value)
    expect(not nonnegative or number >= 0.0, f"{label} is negative")
    return number


(
    evidence_value,
    campaign_id,
    macro_round_value,
    learner_code_sha256,
    learner_config_sha256,
    learner_identity_sha256,
    validation_contract_sha256,
    validation_binding_sha256,
    production_contract_sha256,
    replay_file_sha256,
    replay_sha256,
    parent_checkpoint_sha256,
    parent_runtime_policy_sha256,
    runtime_container_sha256,
    heldout_snapshot_value,
    heldout_file_sha256,
    heldout_sha256,
    wandb_run_id,
) = sys.argv[1:]
expect(macro_round_value == "0", "macro round is not exactly zero")
sha_pattern = re.compile(r"[0-9a-f]{64}\Z")
for label, value in (
    ("learner code", learner_code_sha256),
    ("learner config", learner_config_sha256),
    ("learner identity", learner_identity_sha256),
    ("validation contract", validation_contract_sha256),
    ("validation binding", validation_binding_sha256),
    ("production contract", production_contract_sha256),
    ("training replay file", replay_file_sha256),
    ("training replay semantic", replay_sha256),
    ("parent checkpoint", parent_checkpoint_sha256),
    ("parent runtime policy", parent_runtime_policy_sha256),
    ("runtime container", runtime_container_sha256),
    ("held-out file", heldout_file_sha256),
    ("held-out semantic", heldout_sha256),
):
    expect(sha_pattern.fullmatch(value) is not None, f"{label} digest is malformed")
expect(
    wandb_run_id == (
        f"rrd-m000000-id{learner_identity_sha256}"
        f"-b{validation_binding_sha256[:32]}"
    ),
    "W&B run ID is not identity-and-binding-bound",
)

evidence = Path(os.path.abspath(evidence_value))
expect(not evidence.is_symlink() and evidence.is_dir(), "evidence root is unsafe")
expect(not evidence.stat().st_mode & 0o222, "evidence root is mutable")
expected_files = {
    "success_attestation.json",
    "heldout100_wandb_readback.json",
    "restart_equivalence.json",
}
try:
    children = {entry.name for entry in evidence.iterdir()}
except OSError as exc:
    fail(f"cannot inspect evidence root: {exc}")
expect(children == expected_files, "evidence file set differs")
attestation_path = evidence / "success_attestation.json"
readback_path = evidence / "heldout100_wandb_readback.json"
restart_path = evidence / "restart_equivalence.json"
attestation = strict_object(attestation_path, "success attestation")
readback = strict_object(readback_path, "W&B readback")
restart = strict_object(restart_path, "restart-equivalence report")

attestation_keys = {
    "schema", "status", "purpose", "campaign_id", "allocation",
    "optimizer_steps", "intentional_stop_step", "heldout_optimizer_steps",
    "heldout_case_count", "heldout_snapshot_path",
    "heldout_snapshot_file_sha256", "heldout_snapshot_sha256",
    "validation_contract_sha256", "validation_binding_sha256",
    "production_target_contract_sha256",
    "learner_code_sha256", "learner_config_sha256",
    "learner_identity_sha256", "runtime_container_sha256",
    "training_replay_snapshot_file_sha256",
    "training_replay_snapshot_sha256", "parent_checkpoint_sha256",
    "parent_runtime_policy_sha256", "wandb_run_id",
    "wandb_quality_key_count", "quality_improvement", "loss_improvement",
    "wandb_readback_sha256", "restart_report_sha256",
    "strict_resume_verified", "uninterrupted_equivalence_verified",
    "stale_wandb_history_policy",
    "quality_improvement_verified", "loss_improvement_verified",
}
expect(set(attestation) == attestation_keys, "success attestation field set differs")
exact_values(
    attestation,
    {
        "schema": "fast_roborender.round_dagger.oci_nrt_h100_heldout100_validation.v1",
        "status": "PASS",
        "purpose": "live_h100_fixed256_step100_wandb_restart_gate",
        "campaign_id": campaign_id,
        "allocation": {
            "nodes": 1,
            "gpus_per_node": 8,
            "gpu_architecture": "H100-SM90",
            "backend": "nccl",
            "batch_size_per_rank": 1,
            "global_batch_size": 8,
        },
        "optimizer_steps": 100,
        "intentional_stop_step": 50,
        "heldout_optimizer_steps": [0, 100],
        "heldout_case_count": 256,
        "heldout_snapshot_path": heldout_snapshot_value,
        "heldout_snapshot_file_sha256": heldout_file_sha256,
        "heldout_snapshot_sha256": heldout_sha256,
        "validation_contract_sha256": validation_contract_sha256,
        "validation_binding_sha256": validation_binding_sha256,
        "production_target_contract_sha256": production_contract_sha256,
        "learner_code_sha256": learner_code_sha256,
        "learner_config_sha256": learner_config_sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "training_replay_snapshot_file_sha256": replay_file_sha256,
        "training_replay_snapshot_sha256": replay_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_runtime_policy_sha256": parent_runtime_policy_sha256,
        "wandb_run_id": wandb_run_id,
        "wandb_quality_key_count": 49,
        "strict_resume_verified": True,
        "uninterrupted_equivalence_verified": True,
        "stale_wandb_history_policy": "fail_closed_no_adoption",
        "quality_improvement_verified": True,
        "loss_improvement_verified": True,
    },
    "success attestation",
)
expect(
    attestation["wandb_readback_sha256"] == digest(readback_path),
    "W&B readback digest differs",
)
expect(
    attestation["restart_report_sha256"] == digest(restart_path),
    "restart report digest differs",
)

exact_values(
    readback,
    {
        "schema": "fast_roborender.round_dagger.h100_heldout100_wandb_readback.v1",
        "status": "PASS",
        "account_email": "ldu@nvidia.com",
        "entity": "nvidia",
        "authorization_scope": "team",
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
        "project": "roborender-round-dagger-validation",
        "wandb_run_id": wandb_run_id,
        "campaign_id": campaign_id,
        "macro_round": 0,
        "learner_code_sha256": learner_code_sha256,
        "learner_config_sha256": learner_config_sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "replay_snapshot_file_sha256": replay_file_sha256,
        "replay_snapshot_sha256": replay_sha256,
        "model_lineage_sha256": parent_checkpoint_sha256,
        "parent_runtime_policy_sha256": parent_runtime_policy_sha256,
        "learner_contract_sha256": validation_contract_sha256,
        "validation_contract_sha256": validation_contract_sha256,
        "validation_binding_sha256": validation_binding_sha256,
        "production_contract_sha256": production_contract_sha256,
        "heldout_snapshot_file_sha256": heldout_file_sha256,
        "heldout_snapshot_sha256": heldout_sha256,
        "heldout_case_count": 256,
        "heldout_optimizer_steps": [0, 100],
        "histories_identical": True,
        "quality_histories_identical": True,
        "quality_improvement_verified": True,
        "loss_improvement_verified": True,
        "strict_resume_verified": True,
        "single_canonical_wandb_run": True,
        "stale_wandb_history_policy": "fail_closed_no_adoption",
    },
    "W&B readback",
)
remote = readback.get("remote")
expect(isinstance(remote, dict), "W&B remote readback is absent")
exact_values(
    remote,
    {
        "state": "finished",
        "training_history_steps": list(range(1, 101)),
        "quality_history_steps": [0, 100],
        "quality_key_count": 49,
        "config_verified": True,
        "quality_improvement_verified": True,
    },
    "W&B remote readback",
)

quality = readback.get("quality_improvement")
expect(
    isinstance(quality, dict)
    and set(quality)
    == {
        "initial_action_mse", "final_action_mse",
        "absolute_action_mse_reduction", "relative_action_mse_reduction",
        "action_mae_delta", "action_rmse_delta", "action_cosine_delta",
    },
    "quality-improvement payload differs",
)
initial_mse = finite(quality["initial_action_mse"], "initial action MSE", nonnegative=True)
final_mse = finite(quality["final_action_mse"], "final action MSE", nonnegative=True)
absolute_mse = finite(
    quality["absolute_action_mse_reduction"],
    "absolute action MSE reduction",
)
relative_mse = finite(
    quality["relative_action_mse_reduction"],
    "relative action MSE reduction",
)
expect(initial_mse > 0.0 and final_mse < initial_mse, "held-out MSE did not improve")
expect(math.isclose(absolute_mse, initial_mse - final_mse, rel_tol=1e-12, abs_tol=1e-12), "absolute MSE reduction is inconsistent")
expect(math.isclose(relative_mse, absolute_mse / initial_mse, rel_tol=1e-12, abs_tol=1e-12) and relative_mse >= 0.001, "relative MSE reduction gate did not pass")
for key in ("action_mae_delta", "action_rmse_delta", "action_cosine_delta"):
    finite(quality[key], key)

loss = readback.get("loss_improvement")
expect(
    isinstance(loss, dict)
    and set(loss)
    == {"initial_loss", "final_loss", "absolute_loss_reduction", "relative_loss_reduction"},
    "loss-improvement payload differs",
)
initial_loss = finite(loss["initial_loss"], "initial loss", nonnegative=True)
final_loss = finite(loss["final_loss"], "final loss", nonnegative=True)
absolute_loss = finite(loss["absolute_loss_reduction"], "absolute loss reduction")
relative_loss = finite(loss["relative_loss_reduction"], "relative loss reduction")
expect(initial_loss > 0.0 and final_loss < initial_loss, "deterministic-probe loss did not improve")
expect(math.isclose(absolute_loss, initial_loss - final_loss, rel_tol=1e-12, abs_tol=1e-12), "absolute loss reduction is inconsistent")
expect(math.isclose(relative_loss, absolute_loss / initial_loss, rel_tol=1e-12, abs_tol=1e-12), "relative loss reduction is inconsistent")
expect(attestation["quality_improvement"] == quality, "attested quality improvement differs")
expect(attestation["loss_improvement"] == loss, "attested loss improvement differs")

restart_keys = {
    "schema", "semantic_equivalent", "expected_completed_steps",
    "uninterrupted_checkpoint", "uninterrupted_checkpoint_sha256",
    "resumed_checkpoint", "resumed_checkpoint_sha256", "results_equivalent",
    "uninterrupted_result", "uninterrupted_result_sha256",
    "resumed_result", "resumed_result_sha256",
}
expect(set(restart) == restart_keys, "restart report field set differs")
exact_values(
    restart,
    {
        "schema": "fast_roborender.round_dagger.restart_equivalence.v1",
        "semantic_equivalent": True,
        "results_equivalent": True,
        "expected_completed_steps": 100,
    },
    "restart report",
)
for key in (
    "uninterrupted_checkpoint_sha256", "resumed_checkpoint_sha256",
    "uninterrupted_result_sha256", "resumed_result_sha256",
):
    expect(sha_pattern.fullmatch(restart.get(key, "")) is not None, f"restart report {key} is malformed")
for key in (
    "uninterrupted_checkpoint", "resumed_checkpoint",
    "uninterrupted_result", "resumed_result",
):
    expect(isinstance(restart.get(key), str) and restart[key].startswith("/"), f"restart report {key} is malformed")

print("HELDOUT100_GATE_EVIDENCE_VERIFIED")
PY
}
# END_HELDOUT100_GATE_EVIDENCE_VERIFIER

heldout100_evidence_arguments=()
if ((HELDOUT100_GATE_ENABLED)); then
    heldout100_evidence_arguments=(
        "$ROUND_DAGGER_CAMPAIGN_ID" "$ROUND_DECIMAL"
        "$ROUND_DAGGER_LEARNER_CODE_SHA256" "$HELDOUT100_CONFIG_SHA256"
        "$HELDOUT100_IDENTITY_SHA256" "$HELDOUT100_CONTRACT_SHA256"
        "$HELDOUT100_BINDING_SHA256"
        "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
        "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
        "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"
        "$ROUND_DAGGER_POLICY_SHA256" "$ROUND_DAGGER_RUNTIME_POLICY_SHA256"
        "$REMOTE_CONTAINER_SHA" "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT"
        "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256"
        "$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256"
        "$HELDOUT100_WANDB_RUN_ID"
    )
fi

heldout100_remote_state() {
    ssh -o BatchMode=yes "$REMOTE" bash -s -- \
        "$HELDOUT100_REMOTE_OUTPUT" "$HELDOUT100_REMOTE_JOB" \
        "$HELDOUT100_REMOTE_LOG" "$HELDOUT100_JOB_NAME" \
        "$REMOTE_SLURM_BIN/squeue" <<'REMOTE_HELDOUT100_STATE'
set -Eeuo pipefail
output=$1; job=$2; log=$3; job_name=$4; squeue_bin=$5
for path in "$output" "$job" "$log"; do
    [ -d "$path" ] && [ ! -L "$path" ] && \
        [ "$(realpath -e -- "$path")" = "$path" ] || exit 3
done
active_output=$("$squeue_bin" -h --name="$job_name" --format='%A')
active=()
while IFS= read -r item; do
    [ -z "$item" ] || active+=("$item")
done <<<"$active_output"
[ "${#active[@]}" -le 1 ] || { printf '%s\n' unsafe; exit 3; }
if [ "${#active[@]}" -eq 1 ]; then
    [[ ${active[0]} =~ ^[0-9]+$ ]] || { printf '%s\n' unsafe; exit 3; }
    printf 'active:%s\n' "${active[0]}"
    exit 0
fi
attestation="$output/success_attestation.json"
readback="$output/heldout100_wandb_readback.json"
restart="$output/restart_equivalence.json"
if [ -e "$attestation" ] || [ -L "$attestation" ] || \
   [ -e "$readback" ] || [ -L "$readback" ] || \
   [ -e "$restart" ] || [ -L "$restart" ]; then
    for path in "$attestation" "$readback" "$restart"; do
        [ -f "$path" ] && [ ! -L "$path" ] && \
            [ -z "$(find "$path" -maxdepth 0 -perm /222 -print -quit)" ] || \
            { printf '%s\n' partial; exit 0; }
    done
    printf '%s\n' complete
elif [ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ] && \
     [ -z "$(find "$job" -mindepth 1 -maxdepth 1 -print -quit)" ] && \
     [ -z "$(find "$log" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    printf '%s\n' empty
else
    printf '%s\n' partial
fi
REMOTE_HELDOUT100_STATE
}

wait_for_heldout100_remote_completion() {
    local smoke_state=$1 job_id polls accounting_output accounting state exit_code
    [[ $smoke_state == active:* ]] || {
        printf '%s\n' "$smoke_state"
        return 0
    }
    job_id=${smoke_state#active:}
    [[ $job_id =~ ^[0-9]+$ ]] || fatal "heldout100 active job ID is malformed"
    polls=$(((18000 + POLL_SECONDS - 1) / POLL_SECONDS))
    for ((poll=0; poll < polls; poll++)); do
        sleep "$POLL_SECONDS"
        smoke_state=$(heldout100_remote_state) || \
            fatal "cannot poll recovered heldout100 gate"
        [[ $smoke_state == active:* ]] || break
        [[ ${smoke_state#active:} == "$job_id" ]] || \
            fatal "heldout100 active job identity changed while polling"
    done
    if [[ $smoke_state != complete ]]; then
        accounting=
        for _ in {1..60}; do
            accounting_output=$(ssh -o BatchMode=yes "$REMOTE" \
                "$REMOTE_SLURM_BIN/sacct" -nP -X -j "$job_id" \
                --format=State,ExitCode) || { sleep 2; continue; }
            accounting=$(awk -F'|' 'NF>=2 {print $1 "|" $2; exit}' \
                <<<"$accounting_output")
            [[ -n $accounting ]] && break
            sleep 2
        done
        if [[ -n $accounting ]]; then
            state=${accounting%%|*}; state=${state%%+*}; exit_code=${accounting#*|}
            echo "OCI-NRT heldout100 recovered allocation ended state=$state exit_code=$exit_code evidence_state=$smoke_state" >&2
        else
            echo "OCI-NRT heldout100 recovered allocation accounting unresolved evidence_state=$smoke_state" >&2
        fi
    fi
    printf '%s\n' "$smoke_state"
}

if ((HELDOUT100_GATE_ENABLED)); then
    if [[ -e $HELDOUT100_LOCAL_EVIDENCE || -L $HELDOUT100_LOCAL_EVIDENCE ]]; then
        [[ -d $HELDOUT100_LOCAL_EVIDENCE && ! -L $HELDOUT100_LOCAL_EVIDENCE ]] || \
            fatal "existing heldout100 local evidence is unsafe"
        verify_heldout100_gate_evidence "$HELDOUT100_LOCAL_EVIDENCE" \
            "${heldout100_evidence_arguments[@]}" >/dev/null || \
            fatal "existing heldout100 evidence does not bind this exact run"
        echo "OCI-NRT heldout100 gate evidence adopted identity=$HELDOUT100_IDENTITY_SHA256"
    else
        ssh -o BatchMode=yes "$REMOTE" bash -s -- \
            "$HELDOUT100_REMOTE_OUTPUT" "$HELDOUT100_REMOTE_JOB" \
            "$HELDOUT100_REMOTE_LOG" <<'REMOTE_HELDOUT100_PREPARE'
set -Eeuo pipefail
output=$1; job=$2; log=$3
[ -d "$output" ] && [ ! -L "$output" ] && \
    [ "$(realpath -e -- "$output")" = "$output" ] || exit 2
for path in "$job" "$log"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -d "$path" ] && [ ! -L "$path" ] || exit 2
    else
        mkdir -p "$path"
    fi
    [ "$(realpath -e -- "$path")" = "$path" ] || exit 2
done
chmod u+rwx "$output" "$job" "$log"
REMOTE_HELDOUT100_PREPARE

        smoke_state=$(heldout100_remote_state) || \
            fatal "cannot determine remote heldout100 gate state"
        case "$smoke_state" in
            empty)
                echo "OCI-NRT heldout100 gate submitting identity=$HELDOUT100_IDENTITY_SHA256"
                ssh -o BatchMode=yes "$REMOTE" env \
                    ROUND_DAGGER_HELDOUT100_SBATCH_BIN="$REMOTE_SLURM_BIN/sbatch" \
                    ROUND_DAGGER_HELDOUT100_SOURCE_ROOT="$REMOTE_SOURCE" \
                    ROUND_DAGGER_HELDOUT100_SOURCE_INVENTORY_SHA256="$REMOTE_SOURCE_INVENTORY_SHA" \
                    ROUND_DAGGER_HELDOUT100_CONTAINER="$REMOTE_CONTAINER" \
                    ROUND_DAGGER_HELDOUT100_CONTAINER_SHA256="$REMOTE_CONTAINER_SHA" \
                    ROUND_DAGGER_HELDOUT100_REMOTE_CAMPAIGN_ROOT="$HELDOUT100_REMOTE_CAMPAIGN" \
                    ROUND_DAGGER_HELDOUT100_LOCAL_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
                    ROUND_DAGGER_HELDOUT100_REMOTE_OUTPUT_ROOT="$HELDOUT100_REMOTE_OUTPUT" \
                    ROUND_DAGGER_HELDOUT100_LOCAL_OUTPUT_ROOT="$HELDOUT100_LOCAL_OUTPUT" \
                    ROUND_DAGGER_HELDOUT100_REMOTE_JOB_ROOT="$HELDOUT100_REMOTE_JOB" \
                    ROUND_DAGGER_HELDOUT100_REMOTE_LOG_ROOT="$HELDOUT100_REMOTE_LOG" \
                    ROUND_DAGGER_HELDOUT100_WANDB_NETRC="$REMOTE_NETRC" \
                    ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT="$ROUND_DAGGER_REPLAY_SNAPSHOT" \
                    ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_FILE_SHA256="$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
                    ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_SHA256="$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
                    ROUND_DAGGER_HELDOUT100_POLICY_CHECKPOINT="$ROUND_DAGGER_POLICY_CHECKPOINT" \
                    ROUND_DAGGER_HELDOUT100_POLICY_SHA256="$ROUND_DAGGER_POLICY_SHA256" \
                    ROUND_DAGGER_HELDOUT100_POLICY_SPEC_SHA256="$ROUND_DAGGER_POLICY_SPEC_SHA256" \
                    ROUND_DAGGER_HELDOUT100_POLICY_MANIFEST_SHA256="$ROUND_DAGGER_POLICY_MANIFEST_SHA256" \
                    ROUND_DAGGER_HELDOUT100_RUNTIME_POLICY_SHA256="$ROUND_DAGGER_RUNTIME_POLICY_SHA256" \
                    ROUND_DAGGER_HELDOUT100_LEARNER_CONFIG_SHA256="$HELDOUT100_CONFIG_SHA256" \
                    ROUND_DAGGER_HELDOUT100_CONTRACT="$REMOTE_SOURCE/configs/round_dagger/validation_learner_1x8h100_heldout100.json" \
                    ROUND_DAGGER_HELDOUT100_CONTRACT_SHA256="$HELDOUT100_CONTRACT_SHA256" \
                    ROUND_DAGGER_HELDOUT100_BINDING_SHA256="$HELDOUT100_BINDING_SHA256" \
                    ROUND_DAGGER_HELDOUT100_PRODUCTION_CONTRACT_SHA256="$HELDOUT100_PRODUCTION_CONTRACT_SHA256" \
                    ROUND_DAGGER_HELDOUT100_HELDOUT_SNAPSHOT_FILE_SHA256="$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256" \
                    ROUND_DAGGER_HELDOUT100_HELDOUT_SNAPSHOT_SHA256="$ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256" \
                    ROUND_DAGGER_HELDOUT100_WANDB_RUN_ID="$HELDOUT100_WANDB_RUN_ID" \
                    ROUND_DAGGER_HELDOUT100_CAMPAIGN_ID="$ROUND_DAGGER_CAMPAIGN_ID" \
                    ROUND_DAGGER_HELDOUT100_MACRO_ROUND="$ROUND_DECIMAL" \
                    bash "$REMOTE_SOURCE/scripts/round_dagger/submit_oci_nrt_h100_heldout100.sh" >/dev/null || \
                    fatal "OCI-NRT heldout100 gate allocation failed"
                smoke_state=$(heldout100_remote_state) || \
                    fatal "cannot determine post-allocation heldout100 gate state"
                if [[ $smoke_state == active:* ]]; then
                    smoke_state=$(wait_for_heldout100_remote_completion \
                        "$smoke_state")
                fi
                ;;
            active:*)
                echo "OCI-NRT heldout100 gate waiting for recovered job ${smoke_state#active:}"
                smoke_state=$(wait_for_heldout100_remote_completion \
                    "$smoke_state")
                ;;
            complete) ;;
            partial) fatal "remote heldout100 gate contains non-adoptable partial state" ;;
            *) fatal "remote heldout100 gate state is malformed" ;;
        esac
        [[ $smoke_state == complete ]] || \
            fatal "heldout100 gate did not reach an immutable completed state"

        HELDOUT100_INCOMING=$(mktemp -d \
            "$LOCAL_HANDOFF/.heldout100-evidence.incoming.XXXXXX") || \
            fatal "cannot create heldout100 evidence staging directory"
        for name in success_attestation.json heldout100_wandb_readback.json \
                    restart_equivalence.json; do
            rsync --archive --no-links --protect-args -- \
                "$REMOTE:$HELDOUT100_REMOTE_OUTPUT/$name" \
                "$HELDOUT100_INCOMING/$name"
        done
        chmod a-w "$HELDOUT100_INCOMING"
        verify_heldout100_gate_evidence "$HELDOUT100_INCOMING" \
            "${heldout100_evidence_arguments[@]}" >/dev/null || \
            fatal "remote heldout100 evidence does not bind this exact run"
        [[ ! -e $HELDOUT100_LOCAL_EVIDENCE && ! -L $HELDOUT100_LOCAL_EVIDENCE ]] || \
            fatal "heldout100 evidence destination appeared during adoption"
        mv "$HELDOUT100_INCOMING" "$HELDOUT100_LOCAL_EVIDENCE"
        verify_heldout100_gate_evidence "$HELDOUT100_LOCAL_EVIDENCE" \
            "${heldout100_evidence_arguments[@]}" >/dev/null || \
            fatal "published heldout100 evidence failed re-verification"
        echo "OCI-NRT heldout100 gate PASS identity=$HELDOUT100_IDENTITY_SHA256"
    fi
fi

export_names=(
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND
    ROUND_DAGGER_POLICY_SPEC ROUND_DAGGER_POLICY_MANIFEST
    ROUND_DAGGER_POLICY_SPEC_SHA256 ROUND_DAGGER_POLICY_MANIFEST_SHA256
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_REPLAY_SNAPSHOT
    ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256
    ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_OUTPUT_DIR
    ROUND_DAGGER_LEARNER_RESULT ROUND_DAGGER_LEARNER_CODE_SHA256
    ROUND_DAGGER_LEARNER_CONFIG_SHA256 ROUND_DAGGER_LEARNER_CONTRACT_SHA256
    ROUND_DAGGER_LEARNER_CONTRACT
    ROUND_DAGGER_LEARNER_IDENTITY_SHA256
    ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256 ROUND_DAGGER_BASE_POLICY_SPEC
    ROUND_DAGGER_LEARNER_BATCH_SIZE ROUND_DAGGER_LEARNER_MAX_STEPS
    ROUND_DAGGER_LEARNER_SAVE_EVERY ROUND_DAGGER_LEARNER_LEARNING_RATE
    ROUND_DAGGER_LEARNER_WEIGHT_DECAY ROUND_DAGGER_LEARNER_SEED
    ROUND_DAGGER_LEARNER_PARTITION_SEED ROUND_DAGGER_TARGET_ENVIRONMENTS
    ROUND_DAGGER_COHORT_COUNT ROUND_DAGGER_ENVIRONMENTS_PER_COHORT
    ROUND_DAGGER_STEPS_PER_ENVIRONMENT
    ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND ROUND_DAGGER_WANDB_PROJECT
    ROUND_DAGGER_WANDB_ENTITY ROUND_DAGGER_WANDB_RUN_ID
    ROUND_DAGGER_WANDB_EXPECTED_EMAIL ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE
)
if ((HELDOUT_ENABLED)); then
    export_names+=("${heldout_names[@]}")
fi
exports="PATH=$REMOTE_SLURM_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
for name in "${export_names[@]}"; do exports+=",$name=${!name}"; done
exports+=",ROUND_DAGGER_REMOTE_SOURCE_ROOT=$REMOTE_SOURCE"
exports+=",ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256=$REMOTE_SOURCE_INVENTORY_SHA"
exports+=",ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT=$REMOTE_CAMPAIGN"
exports+=",ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY=$REMOTE_INVENTORY"
exports+=",ROUND_DAGGER_REMOTE_LEARNER_OUTPUT=$REMOTE_OUTPUT"
exports+=",ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
exports+=",ROUND_DAGGER_REMOTE_CONTAINER=$REMOTE_CONTAINER"
exports+=",ROUND_DAGGER_REMOTE_CONTAINER_SHA256=$REMOTE_CONTAINER_SHA"
exports+=",ROUND_DAGGER_REMOTE_WANDB_NETRC=$REMOTE_NETRC"
exports+=",ROUND_DAGGER_REMOTE_JOB_ROOT=$REMOTE_JOB_ROOT"
exports+=",ROUND_DAGGER_REMOTE_SUCCESS_MARKER=$REMOTE_SUCCESS_MARKER"

retryable='^(PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT|REVOKED)$'
attempt=1
REMOTE_RESULT_REL=${ROUND_DAGGER_LEARNER_RESULT#"$ROUND_DAGGER_LEARNER_OUTPUT_DIR"/}
[[ $REMOTE_RESULT_REL != "$ROUND_DAGGER_LEARNER_RESULT" && \
   $REMOTE_RESULT_REL != */* && -n $REMOTE_RESULT_REL ]] || \
    fatal "learner result is not a canonical output basename"
REMOTE_COMPLETION_ATTESTATION="$REMOTE_OUTPUT/learner_completion_round_${ROUND_PADDED}.json"
LOCAL_COMPLETION_ATTESTATION="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/learner_completion_round_${ROUND_PADDED}.json"

verify_remote_success() {
    ssh -o BatchMode=yes "$REMOTE" env PYTHONDONTWRITEBYTECODE=1 \
        python3 "$REMOTE_SOURCE/scripts/round_dagger/remote_learner_success.py" verify \
        --marker "$REMOTE_SUCCESS_MARKER" \
        --result "$REMOTE_OUTPUT/$REMOTE_RESULT_REL" \
        --completion-attestation "$REMOTE_COMPLETION_ATTESTATION" \
        --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
        --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
        --learner-code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
        --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
        --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
}
remote_completion_state() {
    ssh -o BatchMode=yes "$REMOTE" bash -s -- \
        "$REMOTE_SUCCESS_MARKER" "$REMOTE_OUTPUT/$REMOTE_RESULT_REL" \
        "$REMOTE_COMPLETION_ATTESTATION" <<'REMOTE_COMPLETION_STATE'
set -Eeuo pipefail
marker=$1; result=$2; attestation=$3
if [ -e "$marker" ] || [ -L "$marker" ]; then
    [ -f "$marker" ] && [ ! -L "$marker" ] || { printf '%s\n' unsafe; exit 3; }
    printf '%s\n' marker
elif [ -e "$result" ] || [ -L "$result" ]; then
    [ -f "$result" ] && [ ! -L "$result" ] || { printf '%s\n' unsafe; exit 3; }
    if [ -e "$attestation" ] || [ -L "$attestation" ]; then
        [ -f "$attestation" ] && [ ! -L "$attestation" ] || \
            { printf '%s\n' unsafe; exit 3; }
        printf '%s\n' attested-result
    else
        printf '%s\n' result-only
    fi
elif [ -e "$attestation" ] || [ -L "$attestation" ]; then
    # An attestation without its bound result is never recoverable.
    printf '%s\n' unsafe
    exit 3
else
    printf '%s\n' empty
fi
REMOTE_COMPLETION_STATE
}
remote_restart_state() {
    ssh -o BatchMode=yes "$REMOTE" bash -s -- \
        "$REMOTE_OUTPUT/restart.pt" <<'REMOTE_RESTART_STATE'
set -Eeuo pipefail
restart=$1
if [ -e "$restart" ] || [ -L "$restart" ]; then
    [ -f "$restart" ] && [ ! -L "$restart" ] || \
        { printf '%s\n' unsafe; exit 3; }
    printf '%s\n' durable
else
    printf '%s\n' missing
fi
REMOTE_RESTART_STATE
}

remote_result_complete=0

while ((remote_result_complete == 0 && attempt <= MAX_ATTEMPTS)); do
    active_output=$(ssh -o BatchMode=yes "$REMOTE" \
        squeue -h --name="$JOB_NAME" --format='%A') || \
        fatal "cannot query active remote learner jobs"
    active_jobs=()
    while IFS= read -r active_job; do
        [[ -z $active_job ]] || active_jobs+=("$active_job")
    done <<<"$active_output"
    ((${#active_jobs[@]} <= 1)) || fatal "multiple active learner jobs match $JOB_NAME"
    if ((${#active_jobs[@]} == 1)); then
        job_id=${active_jobs[0]}
        [[ $job_id =~ ^[0-9]+$ ]] || fatal "active remote job ID is malformed"
        echo "OCI-NRT learner recovered job_id=$job_id attempt=$attempt/$MAX_ATTEMPTS"
    else
        completion_state=$(remote_completion_state) || \
            fatal "cannot determine remote learner completion state"
        case "$completion_state" in
            marker)
                verify_remote_success || fatal "remote success marker is invalid"
                remote_result_complete=1
                job_id=remote-complete
                attempt=0
                echo "OCI-NRT learner success marker verified; resuming artifact return"
                break
                ;;
            attested-result)
                # Completion adoption imports the learner stack and therefore
                # belongs inside the pinned trainer container, never in bare
                # host Python. Resubmit the same sealed job as a finalizer.
                echo "OCI-NRT learner attestation found without marker; submitting container finalization" >&2
                ;;&
            result-only)
                # Training has already committed its immutable result, which
                # is published only after W&B finish.  Resubmit the same sealed
                # job as a finalizer: the container verifies the final restart
                # and result, performs authenticated W&B readback, and publishes
                # the missing completion attestation without another update.
                echo "OCI-NRT learner result found without completion attestation; submitting finalization recovery" >&2
                ;;&
            empty|result-only|attested-result)
                submission=$(ssh -o BatchMode=yes "$REMOTE" sbatch --parsable \
                --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
                --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=64 --mem=0 \
                --time="$LEARNER_WALLTIME" \
                --job-name="$JOB_NAME" \
                --output="$REMOTE_JOB_ROOT/slurm_%j.out" \
                --error="$REMOTE_JOB_ROOT/slurm_%j.err" --export="$exports" \
                "$REMOTE_SOURCE/scripts/round_dagger/oci_nrt_round_learner.sbatch")
                job_id=${submission%%;*}
                [[ $job_id =~ ^[0-9]+$ ]] || \
                    fatal "remote sbatch returned an invalid job ID"
                echo "OCI-NRT learner submitted job_id=$job_id attempt=$attempt/$MAX_ATTEMPTS"
                ;;
            *) fatal "remote learner completion state is malformed" ;;
        esac
    fi
    queue_failures=0
    while :; do
        if queue_output=$(ssh -o BatchMode=yes "$REMOTE" squeue -h -j "$job_id"); then
            queue_failures=0
            [[ -z $queue_output ]] && break
        else
            queue_failures=$((queue_failures + 1))
            ((queue_failures < 5)) || fatal "remote queue query repeatedly failed"
        fi
        sleep "$POLL_SECONDS"
    done
    accounting=
    terminal='^(COMPLETED|PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT|FAILED|CANCELLED|OUT_OF_MEMORY|DEADLINE|REVOKED)$'
    for _ in {1..60}; do
        accounting_output=$(ssh -o BatchMode=yes "$REMOTE" sacct -nP -X -j "$job_id" \
            --format=State,ExitCode) || { sleep 2; continue; }
        accounting=$(awk -F'|' 'NF>=2 {print $1 "|" $2; exit}' <<<"$accounting_output")
        if [[ -n $accounting ]]; then
            observed_state=${accounting%%|*}; observed_state=${observed_state%%+*}
            [[ $observed_state =~ $terminal ]] && break
        fi
        accounting=
        sleep 2
    done
    [[ -n $accounting ]] || fatal "remote accounting did not resolve job $job_id"
    state=${accounting%%|*}; state=${state%%+*}; exit_code=${accounting#*|}
    completion_state=$(remote_completion_state) || \
        fatal "cannot determine post-allocation learner completion state"
    if [[ $completion_state == marker ]]; then
        verify_remote_success || fatal "post-allocation success marker is invalid"
        remote_result_complete=1
        break
    fi
    if [[ $state == COMPLETED && $exit_code == 0:0 ]]; then
        fatal "completed allocation lacks a valid learner completion marker"
    fi
    if [[ $completion_state =~ ^(result-only|attested-result)$ && \
          $attempt -lt $MAX_ATTEMPTS ]] &&
       [[ $state == FAILED || $state =~ $retryable ]]; then
        echo "OCI-NRT learner container finalization incomplete state=$state; retrying" >&2
        attempt=$((attempt + 1))
        continue
    fi
    if [[ $state == FAILED && $completion_state == empty && \
          $attempt -lt $MAX_ATTEMPTS ]]; then
        restart_state=$(remote_restart_state) || \
            fatal "cannot determine remote durable restart state"
        if [[ $restart_state == durable ]]; then
            # A W&B SDK/network failure exits the batch step as FAILED rather
            # than TIMEOUT even though the last save-every frontier is valid.
            # Bound this recovery by MAX_ATTEMPTS; the next sealed allocation
            # validates full restart lineage before running another update.
            echo "OCI-NRT learner FAILED after durable restart; retrying bounded resume" >&2
            attempt=$((attempt + 1))
            continue
        fi
    fi
    if [[ $state =~ $retryable && $attempt -lt $MAX_ATTEMPTS ]]; then
        # PREEMPTED/TIMEOUT/REVOKED all resubmit the exact same semantic
        # identity and remote output namespace.  run_h100_learner.sh validates
        # restart.pt lineage before advancing beyond the save-every=5 frontier.
        echo "OCI-NRT learner state=$state; resubmitting from durable restart" >&2
        attempt=$((attempt + 1))
        continue
    fi
    fatal "OCI-NRT learner failed state=$state exit_code=$exit_code job_id=$job_id"
done

((remote_result_complete == 1)) || fatal "remote learner exhausted retry budget"

# Persist the allocation's read-only attestation before inventory/download.
# The final local verification below binds the exact downloaded result bytes to
# this marker, closing mutation races in the still-writable remote output tree.
INCOMING_SUCCESS_MARKER="$LOCAL_HANDOFF/.remote_learner_success.incoming.json"
rsync --archive --no-links --protect-args -- \
    "$REMOTE:$REMOTE_SUCCESS_MARKER" "$INCOMING_SUCCESS_MARKER"
[[ -f $INCOMING_SUCCESS_MARKER && ! -L $INCOMING_SUCCESS_MARKER && \
   ! -w $INCOMING_SUCCESS_MARKER ]] || fatal "returned success marker is unsafe"
if [[ -e $LOCAL_SUCCESS_MARKER || -L $LOCAL_SUCCESS_MARKER ]]; then
    [[ -f $LOCAL_SUCCESS_MARKER && ! -L $LOCAL_SUCCESS_MARKER ]] || \
        fatal "existing local success marker is unsafe"
    cmp --silent "$INCOMING_SUCCESS_MARKER" "$LOCAL_SUCCESS_MARKER" || \
        fatal "remote success marker changed across bridge restart"
    unlink "$INCOMING_SUCCESS_MARKER"
else
    mv "$INCOMING_SUCCESS_MARKER" "$LOCAL_SUCCESS_MARKER"
fi
chmod a-w "$LOCAL_SUCCESS_MARKER"

ssh -o BatchMode=yes "$REMOTE" env PYTHONDONTWRITEBYTECODE=1 \
    python3 "$REMOTE_SOURCE/scripts/round_dagger/remote_campaign_transfer.py" \
    inventory --root "$REMOTE_OUTPUT" --output "$REMOTE_OUTPUT_INVENTORY" >/dev/null
INCOMING_OUTPUT="${ROUND_DAGGER_LEARNER_OUTPUT_DIR}.incoming-oci-nrt"
if [[ -e $INCOMING_OUTPUT || -L $INCOMING_OUTPUT ]]; then
    [[ -d $INCOMING_OUTPUT && ! -L $INCOMING_OUTPUT ]] || \
        fatal "unsafe incoming learner output"
    "$PYTHON" "$TRANSFER_TOOL" validate-structure --root "$INCOMING_OUTPUT" >/dev/null
else
    mkdir -p "$INCOMING_OUTPUT"
fi
rsync --archive --no-links --protect-args --partial --append-verify \
    --delete-delay -- "$REMOTE:$REMOTE_OUTPUT/" "$INCOMING_OUTPUT/"
INCOMING_OUTPUT_INVENTORY="$LOCAL_HANDOFF/.learner_output.inventory.incoming.json"
rsync --archive --no-links --protect-args -- "$REMOTE:$REMOTE_OUTPUT_INVENTORY" \
    "$INCOMING_OUTPUT_INVENTORY"
[[ -f $INCOMING_OUTPUT_INVENTORY && ! -L $INCOMING_OUTPUT_INVENTORY ]] || \
    fatal "returned learner inventory is unsafe"
if [[ -e $LOCAL_OUTPUT_INVENTORY || -L $LOCAL_OUTPUT_INVENTORY ]]; then
    [[ -f $LOCAL_OUTPUT_INVENTORY && ! -L $LOCAL_OUTPUT_INVENTORY ]] || \
        fatal "existing learner inventory is unsafe"
    cmp --silent "$INCOMING_OUTPUT_INVENTORY" "$LOCAL_OUTPUT_INVENTORY" || \
        fatal "remote learner inventory changed across bridge restart"
    unlink "$INCOMING_OUTPUT_INVENTORY"
else
    mv "$INCOMING_OUTPUT_INVENTORY" "$LOCAL_OUTPUT_INVENTORY"
fi
chmod a-w "$LOCAL_OUTPUT_INVENTORY"
"$PYTHON" "$TRANSFER_TOOL" verify --root "$INCOMING_OUTPUT" \
    --inventory "$LOCAL_OUTPUT_INVENTORY" >/dev/null
INCOMING_RESULT="$INCOMING_OUTPUT/$REMOTE_RESULT_REL"
INCOMING_COMPLETION_ATTESTATION="$INCOMING_OUTPUT/$(basename "$LOCAL_COMPLETION_ATTESTATION")"
[[ -f $INCOMING_RESULT && ! -L $INCOMING_RESULT ]] || \
    fatal "verified incoming tree lacks learner result"
PYTHONPATH="$ROOT" "$PYTHON" \
    "$ROOT/scripts/round_dagger/learner_completion_attestation.py" verify \
    --attestation "$INCOMING_COMPLETION_ATTESTATION" \
    --result "$INCOMING_RESULT" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --parent-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
    --snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
    --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
    --runtime-container-sha256 "$REMOTE_CONTAINER_SHA" \
    --learner-contract-sha256 "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" \
    --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/remote_learner_success.py" verify \
    --marker "$LOCAL_SUCCESS_MARKER" --result "$INCOMING_RESULT" \
    --completion-attestation "$INCOMING_COMPLETION_ATTESTATION" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --learner-code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
"$PYTHON" "$TRANSFER_TOOL" publish --stage "$INCOMING_OUTPUT" \
    --destination "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" \
    --inventory "$LOCAL_OUTPUT_INVENTORY" >/dev/null
[[ -f $ROUND_DAGGER_LEARNER_RESULT && ! -L $ROUND_DAGGER_LEARNER_RESULT ]] || \
    fatal "verified returned tree lacks learner result"
PYTHONPATH="$ROOT" "$PYTHON" \
    "$ROOT/scripts/round_dagger/learner_completion_attestation.py" verify \
    --attestation "$LOCAL_COMPLETION_ATTESTATION" \
    --result "$ROUND_DAGGER_LEARNER_RESULT" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --parent-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
    --snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
    --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" \
    --runtime-container-sha256 "$REMOTE_CONTAINER_SHA" \
    --learner-contract-sha256 "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" \
    --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/remote_learner_success.py" verify \
    --marker "$LOCAL_SUCCESS_MARKER" --result "$ROUND_DAGGER_LEARNER_RESULT" \
    --completion-attestation "$LOCAL_COMPLETION_ATTESTATION" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --learner-code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
    --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256" >/dev/null
echo "OCI_NRT_ROUND_LEARNER_BRIDGE_COMPLETE job_id=$job_id attempts=$attempt"
