#!/usr/bin/env bash
# Submit the dedicated fixed-256/step-100 H100 smoke and wait for attestation.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }
safe_path() {
    [[ $1 =~ ^/[A-Za-z0-9_./-]+$ && $1 != *//* && \
       $1 != */../* && $1 != */./* && $1 != */.. && $1 != */. ]]
}
canonical_remote_dir() {
    local resolved
    [[ -d $1 && ! -L $1 ]] || fatal "remote directory is unsafe: $1"
    resolved=$(realpath -e -- "$1") || fatal "cannot resolve remote directory: $1"
    [[ $resolved == "$1" ]] || fatal "remote directory is not canonical: $1"
}
tree_disjoint() {
    [[ $1 != "$2" && $1 != "$2"/* && $2 != "$1"/* ]]
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
SBATCH_BIN=${ROUND_DAGGER_HELDOUT100_SBATCH_BIN:-/cm/shared/apps/slurm/current/bin/sbatch}
SOURCE=${ROUND_DAGGER_HELDOUT100_SOURCE_ROOT:?set sealed OCI source root}
SOURCE_SHA=${ROUND_DAGGER_HELDOUT100_SOURCE_INVENTORY_SHA256:?set sealed source inventory SHA-256}
CONTAINER=${ROUND_DAGGER_HELDOUT100_CONTAINER:?set OCI H100 container path}
CONTAINER_SHA=${ROUND_DAGGER_HELDOUT100_CONTAINER_SHA256:?set H100 container SHA-256}
REMOTE_CAMPAIGN=${ROUND_DAGGER_HELDOUT100_REMOTE_CAMPAIGN_ROOT:?set immutable remote campaign projection}
LOCAL_CAMPAIGN=${ROUND_DAGGER_HELDOUT100_LOCAL_CAMPAIGN_ROOT:?set exact campaign mount path inside container}
REMOTE_OUTPUT=${ROUND_DAGGER_HELDOUT100_REMOTE_OUTPUT_ROOT:?set empty remote learner-output directory}
LOCAL_OUTPUT=${ROUND_DAGGER_HELDOUT100_LOCAL_OUTPUT_ROOT:?set nested container learner-output path}
REMOTE_JOB=${ROUND_DAGGER_HELDOUT100_REMOTE_JOB_ROOT:?set empty remote job scratch directory}
REMOTE_LOG=${ROUND_DAGGER_HELDOUT100_REMOTE_LOG_ROOT:?set empty sibling scheduler/log directory}
NETRC=${ROUND_DAGGER_HELDOUT100_WANDB_NETRC:-/home/ldu/.netrc}
SNAPSHOT=${ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT:?set container training replay snapshot path}
SNAPSHOT_FILE_SHA=${ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_FILE_SHA256:?set training replay file SHA-256}
SNAPSHOT_SHA=${ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_SHA256:?set training replay semantic SHA-256}
CHECKPOINT=${ROUND_DAGGER_HELDOUT100_POLICY_CHECKPOINT:?set container parent policy checkpoint path}
CHECKPOINT_SHA=${ROUND_DAGGER_HELDOUT100_POLICY_SHA256:?set parent policy checkpoint SHA-256}
SPEC_SHA=${ROUND_DAGGER_HELDOUT100_POLICY_SPEC_SHA256:?set policy spec SHA-256}
MANIFEST_SHA=${ROUND_DAGGER_HELDOUT100_POLICY_MANIFEST_SHA256:?set policy manifest SHA-256}
RUNTIME_SHA=${ROUND_DAGGER_HELDOUT100_RUNTIME_POLICY_SHA256:?set runtime policy SHA-256}
CONFIG_SHA=${ROUND_DAGGER_HELDOUT100_LEARNER_CONFIG_SHA256:?set exact heldout-enabled 100-step bundle SHA-256}
CONTRACT=${ROUND_DAGGER_HELDOUT100_CONTRACT:-$SOURCE/configs/round_dagger/validation_learner_1x8h100_heldout100.json}
CONTRACT_SHA=${ROUND_DAGGER_HELDOUT100_CONTRACT_SHA256:?set heldout100 validation contract SHA-256}
CAMPAIGN_ID=${ROUND_DAGGER_HELDOUT100_CAMPAIGN_ID:?set replay campaign ID}
MACRO_ROUND=${ROUND_DAGGER_HELDOUT100_MACRO_ROUND:?set replay frontier macro round}
PROJECT=${ROUND_DAGGER_HELDOUT100_WANDB_PROJECT:-roborender-round-dagger-validation}
DRY_RUN=${DRY_RUN:-0}

[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || fatal "DRY_RUN must be 0 or 1"
[[ $LOCAL_CAMPAIGN == /mnt/data2/ldu/roborender_round_dagger_production/dagger_prod_run1 ]] || \
    fatal "campaign mount must preserve the exact fixed-256 artifact paths"
[[ $CAMPAIGN_ID == dagger_prod_run1 ]] || fatal "heldout100 campaign must be dagger_prod_run1"
[[ $MACRO_ROUND == 0 ]] || fatal "heldout100 macro round must be exactly zero"
[[ $PROJECT == roborender-round-dagger-validation ]] || fatal "W&B validation project changed"
for path in "$SBATCH_BIN" "$SOURCE" "$CONTAINER" "$REMOTE_CAMPAIGN" \
            "$LOCAL_CAMPAIGN" \
            "$REMOTE_OUTPUT" "$LOCAL_OUTPUT" "$REMOTE_JOB" "$REMOTE_LOG" \
            "$NETRC" "$SNAPSHOT" "$CHECKPOINT" "$CONTRACT"; do
    safe_path "$path" || fatal "unsafe heldout100 path: $path"
done
for name in SOURCE_SHA CONTAINER_SHA SNAPSHOT_FILE_SHA SNAPSHOT_SHA CHECKPOINT_SHA \
            SPEC_SHA MANIFEST_SHA RUNTIME_SHA CONFIG_SHA CONTRACT_SHA; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ -x $SBATCH_BIN ]] || fatal "sbatch is unavailable"
[[ -d $SOURCE && ! -L $SOURCE ]] || fatal "sealed source root is unsafe"
[[ -f $CONTAINER && ! -L $CONTAINER ]] || fatal "container is unsafe"
[[ -d $REMOTE_CAMPAIGN && ! -L $REMOTE_CAMPAIGN ]] || fatal "campaign is unsafe"
[[ -d $REMOTE_OUTPUT && ! -L $REMOTE_OUTPUT ]] || fatal "learner output is unsafe"
[[ -d $REMOTE_JOB && ! -L $REMOTE_JOB ]] || fatal "job scratch is unsafe"
[[ -d $REMOTE_LOG && ! -L $REMOTE_LOG ]] || fatal "scheduler log root is unsafe"
for path in "$REMOTE_CAMPAIGN" "$REMOTE_OUTPUT" "$REMOTE_JOB" "$REMOTE_LOG"; do
    canonical_remote_dir "$path"
done
[[ $REMOTE_OUTPUT != "$REMOTE_CAMPAIGN" ]] || fatal "remote output aliases campaign"
tree_disjoint "$REMOTE_OUTPUT" "$REMOTE_JOB" || fatal "remote output/job roots overlap"
tree_disjoint "$REMOTE_OUTPUT" "$REMOTE_LOG" || fatal "remote output/log roots overlap"
tree_disjoint "$REMOTE_JOB" "$REMOTE_LOG" || fatal "remote job/log roots overlap"
tree_disjoint "$REMOTE_CAMPAIGN" "$REMOTE_JOB" || fatal "remote job root overlaps campaign"
tree_disjoint "$REMOTE_CAMPAIGN" "$REMOTE_LOG" || fatal "remote log root overlaps campaign"
[[ $REMOTE_LOG != "$REMOTE_JOB" && $(dirname "$REMOTE_LOG") == $(dirname "$REMOTE_JOB") ]] || \
    fatal "scheduler log root must be a distinct sibling of job scratch"
[[ -f $NETRC && ! -L $NETRC ]] || fatal "W&B netrc is unsafe"
case $(stat -c %a "$NETRC") in 400|600) ;; *) fatal "W&B netrc mode must be 0400 or 0600" ;; esac
[[ $CONTRACT == "$SOURCE/configs/round_dagger/validation_learner_1x8h100_heldout100.json" && \
   -f $CONTRACT && ! -L $CONTRACT ]] || fatal "heldout100 contract is non-canonical"
[[ $(sha256sum "$CONTRACT" | awk '{print $1}') == "$CONTRACT_SHA" ]] || \
    fatal "heldout100 contract digest mismatch"
PYTHONDONTWRITEBYTECODE=1 python3 \
    "$SOURCE/scripts/round_dagger/h100_heldout100_validation.py" contract \
    --path "$CONTRACT" --sha256 "$CONTRACT_SHA" --source-root "$SOURCE" >/dev/null
[[ $(sha256sum "$CONTAINER" | awk '{print $1}') == "$CONTAINER_SHA" ]] || fatal "container digest mismatch"
[[ $LOCAL_OUTPUT == "$LOCAL_CAMPAIGN"/* ]] || fatal "learner output must be nested in campaign mount"
OUTPUT_RELATIVE=${LOCAL_OUTPUT#"$LOCAL_CAMPAIGN"/}
[[ -d $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE && ! -L $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE ]] || \
    fatal "campaign lacks nested writable output target"
[[ $(realpath -e -- "$REMOTE_CAMPAIGN/$OUTPUT_RELATIVE") == "$REMOTE_OUTPUT" ]] || \
    fatal "remote output does not equal the canonical nested campaign target"
PYTHONDONTWRITEBYTECODE=1 python3 \
    "$SOURCE/scripts/round_dagger/h100_heldout100_validation.py" remote-layout \
    --campaign "$REMOTE_CAMPAIGN" --output "$REMOTE_OUTPUT" \
    --job "$REMOTE_JOB" --log "$REMOTE_LOG" \
    --output-relative-path "$OUTPUT_RELATIVE" >/dev/null
[[ -z $(find "$REMOTE_OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || fatal "learner output must start empty"
[[ -z $(find "$REMOTE_JOB" -mindepth 1 -maxdepth 1 -print -quit) ]] || fatal "job scratch must start empty"
[[ -z $(find "$REMOTE_LOG" -mindepth 1 -maxdepth 1 -print -quit) ]] || fatal "scheduler log must start empty"

translate_campaign_path() {
    local value=$1 relative
    [[ $value == "$LOCAL_CAMPAIGN"/* ]] || fatal "container input is outside campaign: $value"
    relative=${value#"$LOCAL_CAMPAIGN"/}
    [[ -n $relative && $relative != ../* && $relative != */../* ]] || fatal "unsafe campaign-relative input"
    printf '%s/%s\n' "$REMOTE_CAMPAIGN" "$relative"
}
REMOTE_SNAPSHOT=$(translate_campaign_path "$SNAPSHOT")
REMOTE_CHECKPOINT=$(translate_campaign_path "$CHECKPOINT")
REMOTE_SPEC="${REMOTE_CHECKPOINT}.policy_spec.json"
REMOTE_MANIFEST="${REMOTE_CHECKPOINT}.manifest.json"
for path in "$REMOTE_SNAPSHOT" "$REMOTE_CHECKPOINT" "$REMOTE_SPEC" "$REMOTE_MANIFEST"; do
    [[ -f $path && ! -L $path ]] || fatal "required campaign input is unsafe: $path"
done
[[ $(sha256sum "$REMOTE_SNAPSHOT" | awk '{print $1}') == "$SNAPSHOT_FILE_SHA" ]] || fatal "training snapshot digest mismatch"
[[ $(sha256sum "$REMOTE_CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA" ]] || fatal "checkpoint digest mismatch"
[[ $(sha256sum "$REMOTE_SPEC" | awk '{print $1}') == "$SPEC_SHA" ]] || fatal "policy spec digest mismatch"
[[ $(sha256sum "$REMOTE_MANIFEST" | awk '{print $1}') == "$MANIFEST_SHA" ]] || fatal "policy manifest digest mismatch"
[[ -f $REMOTE_CAMPAIGN/heldout_snapshot.json && ! -L $REMOTE_CAMPAIGN/heldout_snapshot.json ]] || \
    fatal "staged fixed-256 held-out snapshot is absent"
[[ $(sha256sum "$REMOTE_CAMPAIGN/heldout_snapshot.json" | awk '{print $1}') == \
   cea0ee36b228c611635af8dd23b83db3bdaa0fcea493acd57379f080273fdbc8 ]] || \
    fatal "staged held-out snapshot file identity mismatch"

mapfile -t IDENTITY < <(
    PYTHONDONTWRITEBYTECODE=1 python3 "$SOURCE/round_dagger/learner_identity.py" \
        "$CAMPAIGN_ID" "$MACRO_ROUND" "$SOURCE_SHA" "$CONFIG_SHA" \
        "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" "$CHECKPOINT_SHA" "$RUNTIME_SHA" \
        "$CONTAINER_SHA" "$CONTRACT_SHA"
)
[[ ${#IDENTITY[@]} == 2 ]] || fatal "learner identity derivation failed"
LEARNER_IDENTITY_SHA=${IDENTITY[0]}
WANDB_RUN_ID=${IDENTITY[1]}

PYTHONDONTWRITEBYTECODE=1 python3 "$SOURCE/scripts/round_dagger/verify_sealed_source_snapshot.py" \
    --root "$SOURCE" --expected-inventory-sha256 "$SOURCE_SHA" >/dev/null
BATCH="$SOURCE/scripts/round_dagger/oci_nrt_h100_heldout100.sbatch"
[[ -x $BATCH && ! -L $BATCH ]] || fatal "sealed source lacks executable heldout100 batch"
COMMAND=(
    "$SBATCH_BIN" --parsable --wait --export=NONE
    --account=general_cs_infra__ngcsre --partition=batch_singlenode --qos=normal
    --nodes=1 --gpus-per-node=8 --cpus-per-task=64 --mem=0 --time=04:00:00
    --job-name="rr-heldout100-${LEARNER_IDENTITY_SHA:0:12}"
    --output="$REMOTE_LOG/slurm_%j.out" --error="$REMOTE_LOG/slurm_%j.err"
    "$BATCH" "$SOURCE" "$SOURCE_SHA" "$CONTAINER" "$CONTAINER_SHA"
    "$REMOTE_CAMPAIGN" "$LOCAL_CAMPAIGN" "$REMOTE_OUTPUT" "$LOCAL_OUTPUT"
    "$REMOTE_JOB" "$REMOTE_LOG" "$NETRC"
    "$SNAPSHOT" "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" "$CHECKPOINT"
    "$CHECKPOINT_SHA" "$SPEC_SHA" "$MANIFEST_SHA" "$RUNTIME_SHA" "$CONFIG_SHA"
    "$CAMPAIGN_ID" "$MACRO_ROUND" "$PROJECT" "$CONTRACT" "$CONTRACT_SHA"
    "$LEARNER_IDENTITY_SHA" "$WANDB_RUN_ID"
)
if ((DRY_RUN)); then
    printf 'DRY_RUN'; printf ' %q' "${COMMAND[@]}"; printf '\n'
    echo "OCI_NRT_H100_HELDOUT100_DRY_RUN_OK identity=$LEARNER_IDENTITY_SHA wandb_run_id=$WANDB_RUN_ID gpus=8 steps=100 stop=50 heldout_steps=0,100 cases=256"
    exit 0
fi
job_id=$("${COMMAND[@]}") || fatal "OCI-NRT heldout100 job failed"
[[ $job_id =~ ^[0-9]+([.;][A-Za-z0-9_.-]+)?$ ]] || fatal "sbatch returned malformed job ID"
ATTESTATION="$REMOTE_OUTPUT/success_attestation.json"
[[ -f $ATTESTATION && ! -L $ATTESTATION && $(stat -c %a "$ATTESTATION") == 444 ]] || \
    fatal "heldout100 success attestation is absent or mutable"
echo "OCI_NRT_H100_HELDOUT100_COMPLETE job_id=$job_id attestation=$ATTESTATION sha256=$(sha256sum "$ATTESTATION" | awk '{print $1}')"
