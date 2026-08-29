#!/bin/bash
# Dedicated fixed-256, held-out-at-step-100 validation on one 8xH100 node.
#SBATCH --account=general_cs_infra__ngcsre
#SBATCH --partition=batch_singlenode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=rr_h100_heldout100
set -Eeuo pipefail

PATH=/cm/shared/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
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

[[ $# == 27 ]] || fatal "expected exactly 27 heldout100 arguments"
REMOTE_SOURCE=$1
SOURCE_SHA=$2
CONTAINER=$3
CONTAINER_SHA=$4
REMOTE_CAMPAIGN=$5
LOCAL_CAMPAIGN=$6
REMOTE_OUTPUT=$7
LOCAL_OUTPUT=$8
REMOTE_JOB=$9
REMOTE_LOG=${10}
NETRC=${11}
SNAPSHOT=${12}
SNAPSHOT_FILE_SHA=${13}
SNAPSHOT_SHA=${14}
CHECKPOINT=${15}
CHECKPOINT_SHA=${16}
SPEC_SHA=${17}
MANIFEST_SHA=${18}
RUNTIME_SHA=${19}
CONFIG_SHA=${20}
CAMPAIGN_ID=${21}
MACRO_ROUND=${22}
PROJECT=${23}
REMOTE_CONTRACT=${24}
CONTRACT_SHA=${25}
LEARNER_IDENTITY_SHA=${26}
WANDB_RUN_ID=${27}
SPEC="${CHECKPOINT}.policy_spec.json"
MANIFEST="${CHECKPOINT}.manifest.json"
HELDOUT_SNAPSHOT="$LOCAL_CAMPAIGN/heldout_snapshot.json"

[[ ${SLURM_JOB_NUM_NODES:-} == 1 ]] || fatal "heldout100 requires one node"
[[ ${SLURM_GPUS_ON_NODE:-8} =~ (^|[^0-9])8([^0-9]|$)|^8$ ]] || fatal "heldout100 requires eight H100 GPUs"
[[ $LOCAL_CAMPAIGN == /mnt/data2/ldu/roborender_round_dagger_production/dagger_prod_run1_resilient_20260829 ]] || \
    fatal "campaign mount does not preserve fixed-256 artifact paths"
for path in "$REMOTE_SOURCE" "$CONTAINER" "$REMOTE_CAMPAIGN" "$LOCAL_CAMPAIGN" \
            "$REMOTE_OUTPUT" "$LOCAL_OUTPUT" \
            "$REMOTE_JOB" "$REMOTE_LOG" "$NETRC" "$SNAPSHOT" "$CHECKPOINT" \
            "$SPEC" "$MANIFEST" "$REMOTE_CONTRACT"; do
    safe_path "$path" || fatal "unsafe heldout100 path: $path"
done
for name in SOURCE_SHA CONTAINER_SHA SNAPSHOT_FILE_SHA SNAPSHOT_SHA CHECKPOINT_SHA \
            SPEC_SHA MANIFEST_SHA RUNTIME_SHA CONFIG_SHA CONTRACT_SHA LEARNER_IDENTITY_SHA; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ $CAMPAIGN_ID == dagger_prod_run1 && $MACRO_ROUND == 0 ]] || fatal "campaign/round identity is invalid"
[[ $PROJECT == roborender-round-dagger-validation ]] || fatal "W&B project changed"
[[ -d $REMOTE_SOURCE && ! -L $REMOTE_SOURCE ]] || fatal "sealed source is unsafe"
[[ -f $CONTAINER && ! -L $CONTAINER ]] || fatal "container is unsafe"
[[ $(sha256sum "$CONTAINER" | awk '{print $1}') == "$CONTAINER_SHA" ]] || fatal "container digest mismatch"
[[ $REMOTE_CONTRACT == "$REMOTE_SOURCE/configs/round_dagger/validation_learner_1x8h100_heldout100.json" && \
   -f $REMOTE_CONTRACT && ! -L $REMOTE_CONTRACT ]] || fatal "heldout100 contract is non-canonical"
[[ $(sha256sum "$REMOTE_CONTRACT" | awk '{print $1}') == "$CONTRACT_SHA" ]] || fatal "contract digest mismatch"
for path in "$REMOTE_CAMPAIGN" "$REMOTE_OUTPUT" "$REMOTE_JOB" "$REMOTE_LOG"; do
    [[ -d $path && ! -L $path ]] || fatal "required directory is unsafe: $path"
    canonical_remote_dir "$path"
done
[[ $REMOTE_OUTPUT != "$REMOTE_CAMPAIGN" ]] || fatal "remote output aliases campaign"
tree_disjoint "$REMOTE_OUTPUT" "$REMOTE_JOB" || fatal "remote output/job roots overlap"
tree_disjoint "$REMOTE_OUTPUT" "$REMOTE_LOG" || fatal "remote output/log roots overlap"
tree_disjoint "$REMOTE_JOB" "$REMOTE_LOG" || fatal "remote job/log roots overlap"
tree_disjoint "$REMOTE_CAMPAIGN" "$REMOTE_JOB" || fatal "remote job root overlaps campaign"
tree_disjoint "$REMOTE_CAMPAIGN" "$REMOTE_LOG" || fatal "remote log root overlaps campaign"
[[ -f $NETRC && ! -L $NETRC ]] || fatal "W&B netrc is unsafe"
case $(stat -c %a "$NETRC") in 400|600) ;; *) fatal "W&B netrc mode must be 0400 or 0600" ;; esac
[[ $LOCAL_OUTPUT == "$LOCAL_CAMPAIGN"/* ]] || fatal "output mount is not nested"
OUTPUT_RELATIVE=${LOCAL_OUTPUT#"$LOCAL_CAMPAIGN"/}
[[ -d $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE && ! -L $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE ]] || fatal "nested output target is absent"
[[ $(realpath -e -- "$REMOTE_CAMPAIGN/$OUTPUT_RELATIVE") == "$REMOTE_OUTPUT" ]] || \
    fatal "remote output does not equal the canonical nested campaign target"
PYTHONDONTWRITEBYTECODE=1 python3 \
    "$REMOTE_SOURCE/scripts/round_dagger/h100_heldout100_validation.py" remote-layout \
    --campaign "$REMOTE_CAMPAIGN" --output "$REMOTE_OUTPUT" \
    --job "$REMOTE_JOB" --log "$REMOTE_LOG" \
    --output-relative-path "$OUTPUT_RELATIVE" >/dev/null
[[ -z $(find "$REMOTE_OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || fatal "learner output must start empty"
[[ -z $(find "$REMOTE_JOB" -mindepth 1 -maxdepth 1 -print -quit) ]] || fatal "job scratch must start empty"

PYTHONDONTWRITEBYTECODE=1 python3 "$REMOTE_SOURCE/scripts/round_dagger/verify_sealed_source_snapshot.py" \
    --root "$REMOTE_SOURCE" --expected-inventory-sha256 "$SOURCE_SHA" >/dev/null

MOUNTS="$REMOTE_SOURCE:/workspace/fast_roborender:ro"
MOUNTS+=",$REMOTE_CAMPAIGN:$LOCAL_CAMPAIGN:ro"
MOUNTS+=",$REMOTE_OUTPUT:$LOCAL_OUTPUT"
MOUNTS+=",$REMOTE_JOB:/round_dagger_heldout100_job"
MOUNTS+=",$NETRC:/home/ldu/.netrc:ro"
MOUNTS+=",$CONTAINER:/round_dagger_heldout100_container.sqsh:ro"
COMMON_ENV=(
    HOME=/home/ldu
    GR00T_ROOT=/workspace/fast_roborender/GR00T-VisualSim2Real
    ROUND_DAGGER_H100_PYTHON=/opt/viral-h100/bin/python
    ROUND_DAGGER_HELDOUT100_CAMPAIGN_ROOT="$LOCAL_CAMPAIGN"
    ROUND_DAGGER_HELDOUT100_OUTPUT_ROOT="$LOCAL_OUTPUT"
    ROUND_DAGGER_HELDOUT100_JOB_ROOT=/round_dagger_heldout100_job
    ROUND_DAGGER_HELDOUT100_SOURCE_INVENTORY_SHA256="$SOURCE_SHA"
    ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT="$SNAPSHOT"
    ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_FILE_SHA256="$SNAPSHOT_FILE_SHA"
    ROUND_DAGGER_HELDOUT100_REPLAY_SNAPSHOT_SHA256="$SNAPSHOT_SHA"
    ROUND_DAGGER_HELDOUT100_HELDOUT_SNAPSHOT="$HELDOUT_SNAPSHOT"
    ROUND_DAGGER_HELDOUT100_POLICY_CHECKPOINT="$CHECKPOINT"
    ROUND_DAGGER_HELDOUT100_POLICY_SHA256="$CHECKPOINT_SHA"
    ROUND_DAGGER_HELDOUT100_POLICY_SPEC="$SPEC"
    ROUND_DAGGER_HELDOUT100_POLICY_SPEC_SHA256="$SPEC_SHA"
    ROUND_DAGGER_HELDOUT100_POLICY_MANIFEST="$MANIFEST"
    ROUND_DAGGER_HELDOUT100_POLICY_MANIFEST_SHA256="$MANIFEST_SHA"
    ROUND_DAGGER_HELDOUT100_RUNTIME_POLICY_SHA256="$RUNTIME_SHA"
    ROUND_DAGGER_HELDOUT100_LEARNER_CONFIG_SHA256="$CONFIG_SHA"
    ROUND_DAGGER_HELDOUT100_CONTRACT=/workspace/fast_roborender/configs/round_dagger/validation_learner_1x8h100_heldout100.json
    ROUND_DAGGER_HELDOUT100_CONTRACT_SHA256="$CONTRACT_SHA"
    ROUND_DAGGER_HELDOUT100_LEARNER_IDENTITY_SHA256="$LEARNER_IDENTITY_SHA"
    ROUND_DAGGER_HELDOUT100_CAMPAIGN_ID="$CAMPAIGN_ID"
    ROUND_DAGGER_HELDOUT100_MACRO_ROUND="$MACRO_ROUND"
    ROUND_DAGGER_HELDOUT100_CONTAINER_HOST_PATH=/round_dagger_heldout100_container.sqsh
    ROUND_DAGGER_HELDOUT100_CONTAINER_SHA256="$CONTAINER_SHA"
    ROUND_DAGGER_HELDOUT100_WANDB_RUN_ID="$WANDB_RUN_ID"
    ROUND_DAGGER_WANDB_PROJECT="$PROJECT"
    ROUND_DAGGER_WANDB_ENTITY=nvidia
    ROUND_DAGGER_WANDB_EXPECTED_EMAIL=ldu@nvidia.com
    ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE=team
    CUBLAS_WORKSPACE_CONFIG=:4096:8
)

srun --nodes=1 --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=8 \
    --mem=32G --exact --exclusive --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/fast_roborender \
    --output="$REMOTE_LOG/preflight_%j.out" --error="$REMOTE_LOG/preflight_%j.err" \
    env "${COMMON_ENV[@]}" bash /workspace/fast_roborender/containers/h100/trainer_exec.sh \
    bash /workspace/fast_roborender/scripts/round_dagger/run_oci_nrt_h100_heldout100_container.sh preflight

srun --nodes=1 --ntasks=1 --gpus=8 --gpus-per-task=8 --cpus-per-task=64 \
    --mem=0 --exact --exclusive --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/fast_roborender \
    --output="$REMOTE_LOG/learner_%j.out" --error="$REMOTE_LOG/learner_%j.err" \
    env "${COMMON_ENV[@]}" ROUND_DAGGER_LEARNER_NNODES=1 \
    ROUND_DAGGER_LEARNER_NPROC_PER_NODE=8 ROUND_DAGGER_LEARNER_BATCH_SIZE=1 \
    ROUND_DAGGER_LEARNER_MAX_STEPS=100 ROUND_DAGGER_LEARNER_SAVE_EVERY=5 \
    bash /workspace/fast_roborender/containers/h100/trainer_exec.sh \
    bash /workspace/fast_roborender/scripts/round_dagger/run_oci_nrt_h100_heldout100_container.sh learner-sequence

ATTESTATION="$REMOTE_OUTPUT/success_attestation.json"
[[ -f $ATTESTATION && ! -L $ATTESTATION && $(stat -c %a "$ATTESTATION") == 444 ]] || \
    fatal "heldout100 success attestation is missing or mutable"
echo "OCI_NRT_H100_HELDOUT100_JOB_PASS job_id=${SLURM_JOB_ID:-unknown} attestation=$ATTESTATION"
