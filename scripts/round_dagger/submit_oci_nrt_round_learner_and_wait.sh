#!/bin/bash
# Synchronous RTX coordinator -> OCI-NRT 8xH100 learner bridge.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }
safe_absolute_path() {
    [[ $1 =~ ^/[A-Za-z0-9_./-]+$ && $1 != *//* && \
       $1 != */../* && $1 != */./* && $1 != */.. && $1 != */. ]]
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
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
    ROUND_DAGGER_LEARNER_CONFIG_SHA256
    ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256
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
for name in ROUND_DAGGER_POLICY_SPEC_SHA256 ROUND_DAGGER_POLICY_MANIFEST_SHA256 \
            ROUND_DAGGER_POLICY_SHA256 ROUND_DAGGER_RUNTIME_POLICY_SHA256 \
            ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256 \
            ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_CODE_SHA256 \
            ROUND_DAGGER_LEARNER_CONFIG_SHA256 \
            ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ $REMOTE_SOURCE_INVENTORY_SHA == "$ROUND_DAGGER_LEARNER_CODE_SHA256" ]] || \
    fatal "learner code identity is not the sealed source inventory"
[[ $ROUND_DAGGER_MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "macro round is invalid"
[[ "$ROUND_DAGGER_TARGET_ENVIRONMENTS:$ROUND_DAGGER_COHORT_COUNT:$ROUND_DAGGER_ENVIRONMENTS_PER_COHORT:$ROUND_DAGGER_STEPS_PER_ENVIRONMENT:$ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND" == \
   "256:4:64:300:76800" ]] || fatal "production topology is not exactly 4x64x300"

CAMPAIGN_ROOT=$ROUND_DAGGER_CAMPAIGN_ROOT
[[ $CAMPAIGN_ROOT == /* && -d $CAMPAIGN_ROOT && ! -L $CAMPAIGN_ROOT ]] || \
    fatal "campaign root is missing or unsafe"
for path in "$CAMPAIGN_ROOT" "$ROUND_DAGGER_POLICY_CHECKPOINT" \
            "$ROUND_DAGGER_POLICY_SPEC" "$ROUND_DAGGER_POLICY_MANIFEST" \
            "$ROUND_DAGGER_REPLAY_SNAPSHOT" "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" \
            "$ROUND_DAGGER_LEARNER_RESULT"; do
    safe_absolute_path "$path" || fatal "unsafe learner mount/input path"
done
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
ROUND_DAGGER_LEARNER_MAX_STEPS=${ROUND_DAGGER_LEARNER_MAX_STEPS:-500}
ROUND_DAGGER_LEARNER_SAVE_EVERY=${ROUND_DAGGER_LEARNER_SAVE_EVERY:-5}
ROUND_DAGGER_LEARNER_LEARNING_RATE=${ROUND_DAGGER_LEARNER_LEARNING_RATE:-2.5e-5}
ROUND_DAGGER_LEARNER_WEIGHT_DECAY=${ROUND_DAGGER_LEARNER_WEIGHT_DECAY:-0.01}
ROUND_DAGGER_LEARNER_SEED=${ROUND_DAGGER_LEARNER_SEED:-17}
ROUND_DAGGER_LEARNER_PARTITION_SEED=${ROUND_DAGGER_LEARNER_PARTITION_SEED:-23}
[[ "$ROUND_DAGGER_LEARNER_BATCH_SIZE:$ROUND_DAGGER_LEARNER_MAX_STEPS:$ROUND_DAGGER_LEARNER_SAVE_EVERY:$ROUND_DAGGER_LEARNER_LEARNING_RATE:$ROUND_DAGGER_LEARNER_WEIGHT_DECAY:$ROUND_DAGGER_LEARNER_SEED:$ROUND_DAGGER_LEARNER_PARTITION_SEED" == \
   "8:500:5:2.5e-5:0.01:17:23" ]] || fatal "production learner hyperparameters are immutable"
ROUND_DAGGER_WANDB_PROJECT=${ROUND_DAGGER_WANDB_PROJECT:-roborender-round-dagger}
ROUND_DAGGER_WANDB_ENTITY=${ROUND_DAGGER_WANDB_ENTITY:-nvidia}
ROUND_DAGGER_WANDB_EXPECTED_EMAIL=${ROUND_DAGGER_WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE=${ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE:-team}
printf -v ROUND_PADDED '%06d' "$((10#$ROUND_DAGGER_MACRO_ROUND))"
EXPECTED_WANDB_RUN_ID=${ROUND_DAGGER_CAMPAIGN_ID}-learner-m${ROUND_PADDED}-s${ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256:0:12}
ROUND_DAGGER_WANDB_RUN_ID=${ROUND_DAGGER_WANDB_RUN_ID:-$EXPECTED_WANDB_RUN_ID}
[[ $ROUND_DAGGER_WANDB_ENTITY == nvidia ]] || fatal "W&B entity must be nvidia"
[[ $ROUND_DAGGER_WANDB_EXPECTED_EMAIL == ldu@nvidia.com && \
   $ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE == team ]] || \
    fatal "W&B authorization identity changed"
[[ $ROUND_DAGGER_WANDB_RUN_ID == "$EXPECTED_WANDB_RUN_ID" ]] || \
    fatal "W&B run ID is not bound to the replay snapshot"
for token in "$ROUND_DAGGER_CAMPAIGN_ID" "$ROUND_DAGGER_WANDB_PROJECT" \
             "$ROUND_DAGGER_WANDB_RUN_ID"; do
    [[ $token =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "unsafe campaign/W&B token"
done

LOCAL_HANDOFF="$HANDOFF_ROOT/$ROUND_DAGGER_CAMPAIGN_ID/macro_round_$ROUND_PADDED"
LOCAL_PROJECTION="$LOCAL_HANDOFF/campaign_input"
LOCAL_INVENTORY="$LOCAL_HANDOFF/campaign_input.inventory.json"
REMOTE_RUN="$STAGE_ROOT/round_runs/$ROUND_DAGGER_CAMPAIGN_ID/macro_round_$ROUND_PADDED"
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

if ((DRY_RUN)); then
    printf 'DRY_RUN local_campaign=%q projection=learner-inputs-only\n' \
        "$CAMPAIGN_ROOT"
    printf 'DRY_RUN remote_host=%s account=%s partition=%s gpus=8 updates=500 save_every=5\n' \
        "$REMOTE" "$ACCOUNT" "$PARTITION"
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
              "$ROUND_DAGGER_POLICY_MANIFEST" "$ROUND_DAGGER_REPLAY_SNAPSHOT"; do
    [[ $source == "$CAMPAIGN_ROOT"/* ]] || fatal "learner input is outside campaign"
    relative=${source#"$CAMPAIGN_ROOT"/}
    projected="$LOCAL_PROJECTION/$relative"
    [[ -f $projected && ! -L $projected ]] || fatal "projection omitted learner input"
    [[ $(sha256sum "$source" | awk '{print $1}') == \
       $(sha256sum "$projected" | awk '{print $1}') ]] || \
        fatal "projected learner input differs from canonical campaign"
done

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
LEARNER_OUTPUT_REL=${ROUND_DAGGER_LEARNER_OUTPUT_DIR#"$CAMPAIGN_ROOT"/}
[[ $LEARNER_OUTPUT_REL != "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" && \
   -n $LEARNER_OUTPUT_REL && $LEARNER_OUTPUT_REL != /* && \
   $LEARNER_OUTPUT_REL != ../* && $LEARNER_OUTPUT_REL != */../* ]] || \
    fatal "learner output is not a safe campaign-relative directory"
ssh -o BatchMode=yes "$REMOTE" bash -s -- \
    "$REMOTE_CAMPAIGN" "$LEARNER_OUTPUT_REL" <<'REMOTE_SEAL'
set -Eeuo pipefail
root=$1; learner_relative=$2
[ -d "$root" ] && [ ! -L "$root" ] || exit 2
# Pyxis requires a nested writable bind target to exist below the outer
# read-only campaign mount.  Create it only in the remote projection; the
# canonical RTX destination remains absent for atomic publication.
mkdir -p "$root/$learner_relative"
[ -d "$root/$learner_relative" ] && [ ! -L "$root/$learner_relative" ] || exit 2
find "$root" -type f -exec chmod a-w {} +
find "$root" -type d -exec chmod a-w {} +
[ -z "$(find "$root" -perm /222 -print -quit)" ]
REMOTE_SEAL

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
    ROUND_DAGGER_LEARNER_CONFIG_SHA256
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

retryable='^(PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT)$'
attempt=1
campaign_tag=$(printf '%s' "$ROUND_DAGGER_CAMPAIGN_ID" | sha256sum | awk '{print substr($1,1,8)}')
JOB_NAME="rr-rd-${ROUND_DAGGER_CAMPAIGN_ID:0:12}-${campaign_tag}-${ROUND_DAGGER_LEARNER_CODE_SHA256:0:6}-${ROUND_DAGGER_LEARNER_CONFIG_SHA256:0:6}-m${ROUND_PADDED}"
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
        --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" >/dev/null
}
adopt_remote_success() {
    # This path performs no training.  It is the fail-closed recovery for an
    # allocation that finished every learner/W&B validation and published its
    # immutable completion attestation, but died before creating the tiny job
    # success marker.
    ssh -o BatchMode=yes "$REMOTE" env PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$REMOTE_SOURCE" \
        python3 "$REMOTE_SOURCE/scripts/round_dagger/learner_completion_attestation.py" \
        adopt --attestation "$REMOTE_COMPLETION_ATTESTATION" \
        --result "$REMOTE_OUTPUT/$REMOTE_RESULT_REL" \
        --success-marker "$REMOTE_SUCCESS_MARKER" \
        --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
        --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
        --code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
        --config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" \
        --parent-sha256 "$ROUND_DAGGER_POLICY_SHA256" \
        --snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" \
        --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" >/dev/null
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
                adopt_remote_success || \
                    fatal "remote learner result lacks a valid completion attestation; refusing rerun or adoption"
                verify_remote_success || \
                    fatal "adopted remote learner success marker is invalid"
                remote_result_complete=1
                job_id=remote-adopted
                attempt=0
                echo "OCI-NRT learner completion attestation adopted; resuming artifact return"
                break
                ;;
            result-only)
                # Training has already committed its immutable result, which
                # is published only after W&B finish.  Resubmit the same sealed
                # job as a finalizer: the container verifies the final restart
                # and result, performs authenticated W&B readback, and publishes
                # the missing completion attestation without another update.
                echo "OCI-NRT learner result found without completion attestation; submitting finalization recovery" >&2
                ;;&
            empty|result-only)
                submission=$(ssh -o BatchMode=yes "$REMOTE" sbatch --parsable \
                --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
                --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=64 --mem=0 \
                --time=04:00:00 \
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
    if [[ $completion_state == attested-result ]]; then
        adopt_remote_success || \
            fatal "post-allocation completion attestation is invalid"
        verify_remote_success || \
            fatal "post-allocation adopted success marker is invalid"
        remote_result_complete=1
        break
    fi
    if [[ $state == COMPLETED && $exit_code == 0:0 ]]; then
        fatal "completed allocation lacks a valid learner completion marker"
    fi
    if [[ $completion_state == result-only && $attempt -lt $MAX_ATTEMPTS ]] &&
       [[ $state == FAILED || $state =~ $retryable ]]; then
        echo "OCI-NRT learner finalization incomplete state=$state; retrying finalizer" >&2
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
    --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/remote_learner_success.py" verify \
    --marker "$LOCAL_SUCCESS_MARKER" --result "$INCOMING_RESULT" \
    --completion-attestation "$INCOMING_COMPLETION_ATTESTATION" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --learner-code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" >/dev/null
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
    --snapshot-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256" >/dev/null
"$PYTHON" "$ROOT/scripts/round_dagger/remote_learner_success.py" verify \
    --marker "$LOCAL_SUCCESS_MARKER" --result "$ROUND_DAGGER_LEARNER_RESULT" \
    --completion-attestation "$LOCAL_COMPLETION_ATTESTATION" \
    --campaign-id "$ROUND_DAGGER_CAMPAIGN_ID" \
    --macro-round "$ROUND_DAGGER_MACRO_ROUND" \
    --learner-code-sha256 "$ROUND_DAGGER_LEARNER_CODE_SHA256" \
    --learner-config-sha256 "$ROUND_DAGGER_LEARNER_CONFIG_SHA256" >/dev/null
echo "OCI_NRT_ROUND_LEARNER_BRIDGE_COMPLETE job_id=$job_id attempts=$attempt"
