#!/usr/bin/env bash
# Submit-and-wait launcher for the validation-only OCI-NRT H100 live gate.
# Run this on the oci-nrt login host after staging an immutable campaign input.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }
safe_path() {
    [[ $1 =~ ^/[A-Za-z0-9_./-]+$ && $1 != *//* && \
       $1 != */../* && $1 != */./* && $1 != */.. && $1 != */. ]]
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
SBATCH_BIN=${ROUND_DAGGER_VALIDATION_SBATCH_BIN:-/cm/shared/apps/slurm/current/bin/sbatch}
ACCOUNT=${ROUND_DAGGER_VALIDATION_ACCOUNT:-general_cs_infra__ngcsre}
PARTITION=${ROUND_DAGGER_VALIDATION_PARTITION:-batch_singlenode}
QOS=${ROUND_DAGGER_VALIDATION_QOS:-normal}
SOURCE=${ROUND_DAGGER_VALIDATION_SOURCE_ROOT:?set sealed OCI source root}
SOURCE_SHA=${ROUND_DAGGER_VALIDATION_SOURCE_INVENTORY_SHA256:?set sealed source inventory SHA-256}
CONTAINER=${ROUND_DAGGER_VALIDATION_CONTAINER:?set OCI H100 container path}
CONTAINER_SHA=${ROUND_DAGGER_VALIDATION_CONTAINER_SHA256:?set OCI H100 container SHA-256}
REMOTE_CAMPAIGN=${ROUND_DAGGER_VALIDATION_REMOTE_CAMPAIGN_ROOT:?set immutable remote campaign projection}
LOCAL_CAMPAIGN=${ROUND_DAGGER_VALIDATION_LOCAL_CAMPAIGN_ROOT:?set campaign mount path used inside container}
REMOTE_OUTPUT=${ROUND_DAGGER_VALIDATION_REMOTE_OUTPUT_ROOT:?set empty remote learner-output directory}
LOCAL_OUTPUT=${ROUND_DAGGER_VALIDATION_LOCAL_OUTPUT_ROOT:?set nested container learner-output path}
REMOTE_JOB=${ROUND_DAGGER_VALIDATION_REMOTE_JOB_ROOT:?set empty remote job scratch directory}
REMOTE_LOG=${ROUND_DAGGER_VALIDATION_REMOTE_LOG_ROOT:?set empty sibling scheduler/log directory}
NETRC=${ROUND_DAGGER_VALIDATION_WANDB_NETRC:-/home/ldu/.netrc}
SNAPSHOT=${ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT:?set container replay snapshot path}
SNAPSHOT_FILE_SHA=${ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_FILE_SHA256:?set replay snapshot file SHA-256}
SNAPSHOT_SHA=${ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_SHA256:?set replay snapshot semantic SHA-256}
CHECKPOINT=${ROUND_DAGGER_VALIDATION_POLICY_CHECKPOINT:?set container policy checkpoint path}
CHECKPOINT_SHA=${ROUND_DAGGER_VALIDATION_POLICY_SHA256:?set policy checkpoint SHA-256}
SPEC_SHA=${ROUND_DAGGER_VALIDATION_POLICY_SPEC_SHA256:?set policy spec SHA-256}
MANIFEST_SHA=${ROUND_DAGGER_VALIDATION_POLICY_MANIFEST_SHA256:?set policy manifest SHA-256}
RUNTIME_SHA=${ROUND_DAGGER_VALIDATION_RUNTIME_POLICY_SHA256:?set runtime policy SHA-256}
CONFIG_SHA=${ROUND_DAGGER_VALIDATION_LEARNER_CONFIG_SHA256:?set exact batch1/step2/save1 bundle SHA-256}
CONTRACT=${ROUND_DAGGER_VALIDATION_CONTRACT:-$SOURCE/configs/round_dagger/validation_learner_1x8h100_2step.json}
CONTRACT_SHA=${ROUND_DAGGER_VALIDATION_CONTRACT_SHA256:?set validation execution contract SHA-256}
CAMPAIGN_ID=${ROUND_DAGGER_VALIDATION_CAMPAIGN_ID:?set the replay campaign ID}
MACRO_ROUND=${ROUND_DAGGER_VALIDATION_MACRO_ROUND:?set the replay frontier macro round}
PROJECT=${ROUND_DAGGER_VALIDATION_WANDB_PROJECT:-roborender-round-dagger-validation}
DRY_RUN=${DRY_RUN:-0}

[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || fatal "DRY_RUN must be exactly 0 or 1"
[[ $ACCOUNT == general_cs_infra__ngcsre ]] || fatal "OCI-NRT account is not approved"
[[ $PARTITION == batch_singlenode ]] || \
    fatal "OCI-NRT partition must be batch_singlenode"
[[ $QOS == normal ]] || fatal "OCI-NRT QOS must be normal"
[[ $CAMPAIGN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "validation campaign ID is malformed"
[[ $MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "macro round is invalid"
[[ $PROJECT =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "W&B project token is malformed"
for path in "$SBATCH_BIN" "$SOURCE" "$CONTAINER" "$REMOTE_CAMPAIGN" \
            "$LOCAL_CAMPAIGN" "$REMOTE_OUTPUT" "$LOCAL_OUTPUT" "$REMOTE_JOB" \
            "$REMOTE_LOG" "$NETRC" "$SNAPSHOT" "$CHECKPOINT" "$CONTRACT"; do
    safe_path "$path" || fatal "unsafe validation path: $path"
done
for name in SOURCE_SHA CONTAINER_SHA SNAPSHOT_FILE_SHA SNAPSHOT_SHA CHECKPOINT_SHA \
            SPEC_SHA MANIFEST_SHA RUNTIME_SHA CONFIG_SHA CONTRACT_SHA; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ -x $SBATCH_BIN ]] || fatal "sbatch is unavailable"
[[ -d $SOURCE && ! -L $SOURCE ]] || fatal "sealed source root is unsafe"
[[ $CONTRACT == "$SOURCE/configs/round_dagger/validation_learner_1x8h100_2step.json" && \
   -f $CONTRACT && ! -L $CONTRACT ]] || fatal "validation contract is non-canonical"
[[ $(sha256sum "$CONTRACT" | awk '{print $1}') == "$CONTRACT_SHA" ]] || \
    fatal "validation contract digest mismatch"
[[ -f $CONTAINER && ! -L $CONTAINER ]] || fatal "container is unsafe"
[[ -d $REMOTE_CAMPAIGN && ! -L $REMOTE_CAMPAIGN ]] || fatal "campaign is unsafe"
[[ -d $REMOTE_OUTPUT && ! -L $REMOTE_OUTPUT ]] || fatal "learner output is unsafe"
[[ -d $REMOTE_JOB && ! -L $REMOTE_JOB ]] || fatal "job scratch is unsafe"
[[ -d $REMOTE_LOG && ! -L $REMOTE_LOG ]] || fatal "scheduler log root is unsafe"
[[ $REMOTE_LOG != "$REMOTE_JOB" && \
   $(dirname "$REMOTE_LOG") == "$(dirname "$REMOTE_JOB")" ]] || \
    fatal "scheduler log root must be a distinct sibling of job scratch"
[[ -f $NETRC && ! -L $NETRC ]] || fatal "W&B netrc is unsafe"
case $(stat -c %a "$NETRC") in 400|600) ;; \
    *) fatal "W&B netrc permissions must be 0400 or 0600" ;; esac
[[ $LOCAL_OUTPUT == "$LOCAL_CAMPAIGN"/* ]] || \
    fatal "container learner output must be nested under campaign"

translate_campaign_path() {
    local value=$1 relative
    [[ $value == "$LOCAL_CAMPAIGN"/* ]] || \
        fatal "container input is outside the campaign mount: $value"
    relative=${value#"$LOCAL_CAMPAIGN"/}
    [[ -n $relative && $relative != ../* && $relative != */../* ]] || \
        fatal "unsafe campaign-relative input: $value"
    printf '%s/%s\n' "$REMOTE_CAMPAIGN" "$relative"
}

REMOTE_SNAPSHOT=$(translate_campaign_path "$SNAPSHOT")
REMOTE_CHECKPOINT=$(translate_campaign_path "$CHECKPOINT")
REMOTE_SPEC="${REMOTE_CHECKPOINT}.policy_spec.json"
REMOTE_MANIFEST="${REMOTE_CHECKPOINT}.manifest.json"
OUTPUT_RELATIVE=${LOCAL_OUTPUT#"$LOCAL_CAMPAIGN"/}
[[ -d $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE && ! -L $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE ]] || \
    fatal "campaign lacks the nested writable-mount target"
for path in "$REMOTE_SNAPSHOT" "$REMOTE_CHECKPOINT" "$REMOTE_SPEC" "$REMOTE_MANIFEST"; do
    [[ -f $path && ! -L $path ]] || fatal "required campaign input is unsafe: $path"
done
[[ $(sha256sum "$REMOTE_SNAPSHOT" | awk '{print $1}') == "$SNAPSHOT_FILE_SHA" ]] || \
    fatal "snapshot file digest mismatch"
[[ $(sha256sum "$REMOTE_CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA" ]] || \
    fatal "policy checkpoint digest mismatch"
[[ $(sha256sum "$REMOTE_SPEC" | awk '{print $1}') == "$SPEC_SHA" ]] || \
    fatal "policy spec digest mismatch"
[[ $(sha256sum "$REMOTE_MANIFEST" | awk '{print $1}') == "$MANIFEST_SHA" ]] || \
    fatal "policy manifest digest mismatch"

python3 - "$CONTRACT" "$CONTAINER" "$CONTAINER_SHA" "$PROJECT" <<'PY'
import json
import sys

path, container, container_sha, project = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
expected = {
    "schema": "fast_roborender.round_dagger.validation_learner_contract.v1",
    "purpose": "live_h100_restart_and_telemetry_gate",
}
if any(value.get(key) != item for key, item in expected.items()):
    raise SystemExit("validation contract identity mismatch")
if value.get("allocation") != {
    "nodes": 1, "gpus_per_node": 8, "gpu_architecture": "H100-SM90",
    "backend": "nccl", "batch_size_per_rank": 1, "global_batch_size": 8,
}:
    raise SystemExit("validation allocation contract mismatch")
if value.get("optimization") != {
    "optimizer_steps": 2, "save_every_steps": 1, "intentional_stop_step": 1,
    "learning_rate": 2.5e-5, "weight_decay": 0.01, "seed": 17,
    "partition_seed": 23,
}:
    raise SystemExit("validation optimizer contract mismatch")
if value.get("restart_validation") != {
    "uninterrupted_comparator": "local_only",
    "wandb_lineage": "single_canonical_run",
    "first_invocation_resume": "allow", "second_invocation_resume": "must",
}:
    raise SystemExit("validation restart contract mismatch")
container_doc = value.get("container", {})
if container_doc.get("path") != container or container_doc.get("sha256") != container_sha:
    raise SystemExit("validation container differs from checked-in contract")
if value.get("wandb", {}).get("project") != project:
    raise SystemExit("validation W&B project differs from checked-in contract")
PY

mapfile -t IDENTITY_VALUES < <(
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "$SOURCE/round_dagger/learner_identity.py" \
        "$CAMPAIGN_ID" "$MACRO_ROUND" "$SOURCE_SHA" \
        "$CONFIG_SHA" "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" \
        "$CHECKPOINT_SHA" "$RUNTIME_SHA" "$CONTAINER_SHA" "$CONTRACT_SHA"
)
[[ ${#IDENTITY_VALUES[@]} == 2 ]] || fatal "learner identity derivation failed"
LEARNER_IDENTITY_SHA=${IDENTITY_VALUES[0]}
WANDB_RUN_ID=${IDENTITY_VALUES[1]}
[[ $LEARNER_IDENTITY_SHA =~ ^[0-9a-f]{64}$ ]] || fatal "learner identity is malformed"
[[ $WANDB_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "canonical W&B run ID is malformed"

# The host-side raw JSON gate is intentionally independent of replay payload
# resolution; the container repeats the full load after the campaign is mounted
# at its original absolute path.
python3 - "$REMOTE_SNAPSHOT" "$SNAPSHOT_SHA" "$CAMPAIGN_ID" "$MACRO_ROUND" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict) or value.get("snapshot_sha256") != sys.argv[2]:
    raise SystemExit("replay snapshot semantic identity mismatch")
if value.get("campaign_id") != sys.argv[3]:
    raise SystemExit("replay snapshot campaign identity mismatch")
if value.get("through_macro_round") != int(sys.argv[4]):
    raise SystemExit("replay snapshot macro-round frontier mismatch")
if value.get("expected_train_cohorts") != 4:
    raise SystemExit("replay snapshot must preserve four train cohorts per round")
cohorts = value.get("cohorts")
if not isinstance(cohorts, list):
    raise SystemExit("replay snapshot cohorts are malformed")
count = sum(len(row.get("shards", ())) for row in cohorts if isinstance(row, dict))
if count < 8:
    raise SystemExit("validation replay snapshot has fewer than eight shards")
print(f"validation_replay_shards={count}")
PY

VERIFY_SOURCE="$SOURCE/scripts/round_dagger/verify_sealed_source_snapshot.py"
[[ -f $VERIFY_SOURCE && ! -L $VERIFY_SOURCE ]] || fatal "source verifier is missing"
PYTHONDONTWRITEBYTECODE=1 python3 "$VERIFY_SOURCE" --root "$SOURCE" \
    --expected-inventory-sha256 "$SOURCE_SHA" >/dev/null
[[ $(sha256sum "$CONTAINER" | awk '{print $1}') == "$CONTAINER_SHA" ]] || \
    fatal "container digest mismatch"
[[ -z $(find "$REMOTE_OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fatal "validation learner output must be empty"
[[ -z $(find "$REMOTE_JOB" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fatal "validation job scratch must be empty"
[[ -z $(find "$REMOTE_LOG" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fatal "validation scheduler log root must be empty"

BATCH="$SOURCE/scripts/round_dagger/oci_nrt_h100_validation.sbatch"
[[ -x $BATCH && ! -L $BATCH ]] || fatal "sealed source lacks executable validation batch"
COMMAND=(
    "$SBATCH_BIN" --parsable --wait --export=NONE
    --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS"
    --nodes=1 --gpus-per-node=8 --cpus-per-task=64 --mem=0
    --time=01:00:00 --job-name="rr-h100-live-${LEARNER_IDENTITY_SHA:0:12}"
    --output="$REMOTE_LOG/slurm_%j.out" --error="$REMOTE_LOG/slurm_%j.err"
    "$BATCH" "$SOURCE" "$SOURCE_SHA" "$CONTAINER" "$CONTAINER_SHA"
    "$REMOTE_CAMPAIGN" "$LOCAL_CAMPAIGN" "$REMOTE_OUTPUT" "$LOCAL_OUTPUT"
    "$REMOTE_JOB" "$REMOTE_LOG" "$NETRC" "$SNAPSHOT" "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA"
    "$CHECKPOINT" "$CHECKPOINT_SHA" "$SPEC_SHA" "$MANIFEST_SHA" "$RUNTIME_SHA"
    "$CONFIG_SHA" "$CAMPAIGN_ID" "$MACRO_ROUND" "$PROJECT"
    "$CONTRACT" "$CONTRACT_SHA" "$LEARNER_IDENTITY_SHA" "$WANDB_RUN_ID"
)

if ((DRY_RUN)); then
    printf 'DRY_RUN'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    echo "OCI_NRT_H100_VALIDATION_DRY_RUN_OK identity=$LEARNER_IDENTITY_SHA wandb_run_id=$WANDB_RUN_ID contract_sha256=$CONTRACT_SHA gpus=8 batch_per_rank=1 steps=2 save_every=1 stop_step=1"
    exit 0
fi

job_id=$("${COMMAND[@]}") || fatal "OCI-NRT H100 validation job failed"
[[ $job_id =~ ^[0-9]+([.;][A-Za-z0-9_.-]+)?$ ]] || \
    fatal "sbatch returned a malformed job ID"
ATTESTATION="$REMOTE_OUTPUT/success_attestation.json"
[[ -f $ATTESTATION && ! -L $ATTESTATION ]] || fatal "success attestation is absent"
[[ $(stat -c %a "$ATTESTATION") == 444 ]] || fatal "success attestation is not read-only"
job_numeric=${job_id%%[.;]*}
expected_logs=(
    "slurm_${job_numeric}.out" "slurm_${job_numeric}.err"
    "preflight_${job_numeric}.out" "preflight_${job_numeric}.err"
    "learner_${job_numeric}.out" "learner_${job_numeric}.err"
)
for name in "${expected_logs[@]}"; do
    path="$REMOTE_LOG/$name"
    [[ -f $path && ! -L $path ]] || fatal "expected validation log is absent: $name"
    chmod 0444 "$path"
done
[[ $(find "$REMOTE_LOG" -mindepth 1 -maxdepth 1 -type f | wc -l) == 6 ]] || \
    fatal "scheduler log root contains unexpected files"
echo "OCI_NRT_H100_VALIDATION_COMPLETE job_id=$job_id attestation=$ATTESTATION sha256=$(sha256sum "$ATTESTATION" | awk '{print $1}')"
