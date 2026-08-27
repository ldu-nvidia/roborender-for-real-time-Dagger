#!/bin/bash
# One resumable, camera-free 8xH100 round-DAgger learner allocation.
# Conservative standalone scheduler defaults.  The trusted RTX submit/wait
# bridge supplies and pins the approved live account, partition, and resources
# on the sbatch command line, which override these boundary-safe defaults.
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=round_dagger_h100_learner
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

BATCH_DRY_RUN=${ROUND_DAGGER_BATCH_DRY_RUN:-0}
[[ $BATCH_DRY_RUN == 0 || $BATCH_DRY_RUN == 1 ]] || \
    fatal "ROUND_DAGGER_BATCH_DRY_RUN must be exactly 0 or 1"

required=(
    ROUND_DAGGER_REMOTE_SOURCE_ROOT ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256
    ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT
    ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT
    ROUND_DAGGER_CAMPAIGN_ROOT
    ROUND_DAGGER_REMOTE_LEARNER_OUTPUT
    ROUND_DAGGER_REMOTE_CONTAINER ROUND_DAGGER_REMOTE_CONTAINER_SHA256
    ROUND_DAGGER_REMOTE_WANDB_NETRC ROUND_DAGGER_REMOTE_JOB_ROOT
    ROUND_DAGGER_CAMPAIGN_ID ROUND_DAGGER_MACRO_ROUND
    ROUND_DAGGER_POLICY_SPEC ROUND_DAGGER_POLICY_MANIFEST
    ROUND_DAGGER_POLICY_SPEC_SHA256 ROUND_DAGGER_POLICY_MANIFEST_SHA256
    ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
    ROUND_DAGGER_RUNTIME_POLICY_SHA256 ROUND_DAGGER_REPLAY_SNAPSHOT
    ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256
    ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_OUTPUT_DIR
    ROUND_DAGGER_LEARNER_RESULT ROUND_DAGGER_LEARNER_CODE_SHA256
    ROUND_DAGGER_LEARNER_CONFIG_SHA256 ROUND_DAGGER_LEARNER_CONTRACT_SHA256
    ROUND_DAGGER_LEARNER_IDENTITY_SHA256
    ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256
    ROUND_DAGGER_BASE_POLICY_SPEC ROUND_DAGGER_LEARNER_BATCH_SIZE
    ROUND_DAGGER_LEARNER_MAX_STEPS ROUND_DAGGER_LEARNER_SAVE_EVERY
    ROUND_DAGGER_LEARNER_LEARNING_RATE ROUND_DAGGER_LEARNER_WEIGHT_DECAY
    ROUND_DAGGER_LEARNER_SEED ROUND_DAGGER_LEARNER_PARTITION_SEED
    ROUND_DAGGER_WANDB_PROJECT ROUND_DAGGER_WANDB_ENTITY
    ROUND_DAGGER_WANDB_RUN_ID ROUND_DAGGER_WANDB_EXPECTED_EMAIL
    ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE
    ROUND_DAGGER_REMOTE_SUCCESS_MARKER
    ROUND_DAGGER_TARGET_ENVIRONMENTS ROUND_DAGGER_COHORT_COUNT
    ROUND_DAGGER_ENVIRONMENTS_PER_COHORT ROUND_DAGGER_STEPS_PER_ENVIRONMENT
    ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND
)
for name in "${required[@]}"; do
    [[ -n ${!name:-} ]] || fatal "missing required environment: $name"
done
[[ $ROUND_DAGGER_WANDB_EXPECTED_EMAIL == ldu@nvidia.com && \
   $ROUND_DAGGER_WANDB_ENTITY == nvidia && \
   $ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE == team ]] || \
    fatal "W&B identity authorization contract changed"
[[ "$ROUND_DAGGER_TARGET_ENVIRONMENTS:$ROUND_DAGGER_COHORT_COUNT:$ROUND_DAGGER_ENVIRONMENTS_PER_COHORT:$ROUND_DAGGER_STEPS_PER_ENVIRONMENT:$ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND" == \
   "256:4:64:512:131072" ]] || fatal "learner topology is not exactly 4x64x512"

[[ ${SLURM_JOB_NUM_NODES:-} == 1 ]] || fatal "learner allocation must use one node"
[[ ${SLURM_GPUS_ON_NODE:-8} =~ (^|[^0-9])8([^0-9]|$)|^8$ ]] || \
    fatal "learner allocation must expose eight GPUs"
for path in \
    "$ROUND_DAGGER_REMOTE_SOURCE_ROOT" "$ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT" \
    "$ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY" \
    "$ROUND_DAGGER_REMOTE_LEARNER_OUTPUT" "$ROUND_DAGGER_REMOTE_JOB_ROOT" \
    "$ROUND_DAGGER_REMOTE_CONTAINER" "$ROUND_DAGGER_REMOTE_WANDB_NETRC" \
    "$ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT" "$ROUND_DAGGER_CAMPAIGN_ROOT" \
    "$ROUND_DAGGER_REMOTE_SUCCESS_MARKER"; do
    [[ $path == /* && $path != *,* && $path != *:* ]] || \
        fatal "all source/campaign/job paths must be safe and absolute"
done
[[ $ROUND_DAGGER_CAMPAIGN_ROOT == "$ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT" ]] || \
    fatal "container campaign root differs from the projected mount path"
[[ $ROUND_DAGGER_REMOTE_SUCCESS_MARKER == \
   "$ROUND_DAGGER_REMOTE_JOB_ROOT/learner_success.json" ]] || \
    fatal "remote success marker path is non-canonical"
for name in ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256 \
            ROUND_DAGGER_REMOTE_CONTAINER_SHA256 ROUND_DAGGER_POLICY_SPEC_SHA256 \
            ROUND_DAGGER_POLICY_MANIFEST_SHA256 ROUND_DAGGER_POLICY_SHA256 \
            ROUND_DAGGER_RUNTIME_POLICY_SHA256 \
            ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256 \
            ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256 ROUND_DAGGER_LEARNER_CODE_SHA256 \
            ROUND_DAGGER_LEARNER_CONFIG_SHA256 \
            ROUND_DAGGER_LEARNER_CONTRACT_SHA256 \
            ROUND_DAGGER_LEARNER_IDENTITY_SHA256 \
            ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ $ROUND_DAGGER_MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "macro round is invalid"
[[ $ROUND_DAGGER_CAMPAIGN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "campaign ID is malformed"
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
        "runtime_container_sha256=$ROUND_DAGGER_REMOTE_CONTAINER_SHA256" \
        "learner_contract_sha256=$ROUND_DAGGER_LEARNER_CONTRACT_SHA256" | \
        sha256sum | awk '{print $1}'
)
[[ $ROUND_DAGGER_LEARNER_IDENTITY_SHA256 == \
   "$EXPECTED_LEARNER_IDENTITY_SHA256" ]] || \
    fatal "learner identity SHA-256 does not bind the exact semantic run"
EXPECTED_WANDB_RUN_ID="rrd-m${ROUND_PADDED}-id${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}"
[[ $ROUND_DAGGER_WANDB_RUN_ID == "$EXPECTED_WANDB_RUN_ID" ]] || \
    fatal "W&B run ID is not bound to the exact learner identity"
((${#ROUND_DAGGER_WANDB_RUN_ID} <= 128)) || fatal "W&B run ID is too long"
EXPECTED_JOB_NAME="rr-rd-${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}"
[[ ${SLURM_JOB_NAME:-} == "$EXPECTED_JOB_NAME" ]] || \
    fatal "Slurm job name is not bound to the exact learner identity"
IDENTITY_NAMESPACE="identity_${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}"
REMOTE_RUN=${ROUND_DAGGER_REMOTE_JOB_ROOT%/job}
[[ $REMOTE_RUN != "$ROUND_DAGGER_REMOTE_JOB_ROOT" && \
   $REMOTE_RUN == */"$IDENTITY_NAMESPACE" ]] || \
    fatal "remote job root is outside the learner identity namespace"
[[ $ROUND_DAGGER_REMOTE_LEARNER_OUTPUT == "$REMOTE_RUN/learner_output" && \
   $ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT == "$REMOTE_RUN/campaign_input" && \
   $ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY == \
   "$REMOTE_RUN/campaign_input.inventory.json" ]] || \
    fatal "remote learner paths do not share the exact identity namespace"
[[ -d $ROUND_DAGGER_REMOTE_SOURCE_ROOT && ! -L $ROUND_DAGGER_REMOTE_SOURCE_ROOT ]] || \
    fatal "sealed source root is missing or unsafe"
[[ -d $ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT && ! -L $ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT ]] || \
    fatal "remote campaign tree is missing or unsafe"
[[ -f $ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY && \
   ! -L $ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY ]] || \
    fatal "remote campaign inventory is missing or unsafe"
[[ -f $ROUND_DAGGER_REMOTE_CONTAINER && ! -L $ROUND_DAGGER_REMOTE_CONTAINER ]] || \
    fatal "trainer container is missing or unsafe"
[[ $(sha256sum "$ROUND_DAGGER_REMOTE_CONTAINER" | awk '{print $1}') == \
   "$ROUND_DAGGER_REMOTE_CONTAINER_SHA256" ]] || fatal "trainer container SHA mismatch"
[[ -f $ROUND_DAGGER_REMOTE_WANDB_NETRC && ! -L $ROUND_DAGGER_REMOTE_WANDB_NETRC ]] || \
    fatal "remote W&B credential is missing or unsafe"
case $(stat -c %a "$ROUND_DAGGER_REMOTE_WANDB_NETRC") in 400|600) ;; \
    *) fatal "remote W&B credential permissions must be 0400 or 0600" ;; esac
[[ $ROUND_DAGGER_POLICY_SPEC == "${ROUND_DAGGER_POLICY_CHECKPOINT}.policy_spec.json" && \
   $ROUND_DAGGER_POLICY_MANIFEST == "${ROUND_DAGGER_POLICY_CHECKPOINT}.manifest.json" && \
   $ROUND_DAGGER_BASE_POLICY_SPEC == "$ROUND_DAGGER_POLICY_SPEC" ]] || \
    fatal "active policy bundle paths are non-canonical"
policy_local_paths=(
    "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_POLICY_SPEC"
    "$ROUND_DAGGER_POLICY_MANIFEST"
)
policy_digests=(
    "$ROUND_DAGGER_POLICY_SHA256" "$ROUND_DAGGER_POLICY_SPEC_SHA256"
    "$ROUND_DAGGER_POLICY_MANIFEST_SHA256"
)
policy_remote_paths=()
for index in "${!policy_local_paths[@]}"; do
    local_path=${policy_local_paths[$index]}
    [[ $local_path == "$ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT"/* ]] || \
        fatal "active policy bundle is outside the campaign mount"
    relative=${local_path#"$ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT"/}
    [[ -n $relative && $relative != ../* ]] || fatal "unsafe policy relative path"
    path="$ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT/$relative"
    policy_remote_paths+=("$path")
    [[ -f $path && ! -L $path ]] || fatal "active policy bundle file is unsafe"
    [[ $(sha256sum "$path" | awk '{print $1}') == "${policy_digests[$index]}" ]] || \
        fatal "active policy bundle digest mismatch"
done

# Production source is immutable, non-symlinked, and contains no executable
# bytecode.  This is deliberately rechecked inside the allocation.
[[ -f $ROUND_DAGGER_REMOTE_SOURCE_ROOT/.source_snapshot_manifest.json ]] || \
    fatal "sealed source manifest is missing"
[[ -z $(find "$ROUND_DAGGER_REMOTE_SOURCE_ROOT" -xdev \
    \( -type l -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \
       -o -type d -name __pycache__ -o -perm /222 \) -print -quit) ]] || \
    fatal "sealed source tree is writable or unsafe"
[[ $ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256 == \
   "$ROUND_DAGGER_LEARNER_CODE_SHA256" ]] || \
    fatal "learner code identity is not the sealed source inventory"
VERIFY_SOURCE="$ROUND_DAGGER_REMOTE_SOURCE_ROOT/scripts/round_dagger/verify_sealed_source_snapshot.py"
[[ -f $VERIFY_SOURCE && ! -L $VERIFY_SOURCE ]] || \
    fatal "sealed source lacks its verifier"

REMOTE_TRANSFER="$ROUND_DAGGER_REMOTE_SOURCE_ROOT/scripts/round_dagger/remote_campaign_transfer.py"
[[ -f $REMOTE_TRANSFER && ! -L $REMOTE_TRANSFER ]] || \
    fatal "sealed source lacks remote transfer verifier"
# Dependency-bearing validation is deliberately deferred to the mounted
# trainer container. OCI compute-node host Python is not part of the sealed
# runtime and is not required to provide NumPy or PyTorch.

mkdir -p "$ROUND_DAGGER_REMOTE_JOB_ROOT/logs" "$ROUND_DAGGER_REMOTE_LEARNER_OUTPUT"
[[ ! -L $ROUND_DAGGER_REMOTE_LEARNER_OUTPUT ]] || \
    fatal "remote learner output is a symlink"

CONTAINER_ROOT=/workspace/fast_roborender
CONTAINER_HOME=/home/ldu
CONTAINER_VERIFY_CAMPAIGN_ROOT=/round_dagger_campaign_input
CONTAINER_VERIFY_CAMPAIGN_INVENTORY=/round_dagger_campaign_input.inventory.json
CONTAINER_SUCCESS_MARKER=/round_dagger_job/learner_success.json
MOUNTS="$ROUND_DAGGER_REMOTE_SOURCE_ROOT:$CONTAINER_ROOT:ro"
MOUNTS+=",$ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT:$ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT:ro"
MOUNTS+=",$ROUND_DAGGER_REMOTE_LEARNER_OUTPUT:$ROUND_DAGGER_LEARNER_OUTPUT_DIR"
MOUNTS+=",$ROUND_DAGGER_REMOTE_JOB_ROOT:/round_dagger_job"
MOUNTS+=",$ROUND_DAGGER_REMOTE_WANDB_NETRC:$CONTAINER_HOME/.netrc:ro"
MOUNTS+=",$ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT:$CONTAINER_VERIFY_CAMPAIGN_ROOT:ro"
MOUNTS+=",$ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY:$CONTAINER_VERIFY_CAMPAIGN_INVENTORY:ro"

if ((BATCH_DRY_RUN)); then
    echo "OCI_NRT_ROUND_LEARNER_HOST_DRY_RUN_OK validation=containerized host_python=unused"
    exit 0
fi

set +e
srun --nodes=1 --ntasks=1 --gpus=8 --gpus-per-task=8 \
    --cpus-per-task=64 --mem=0 --exact --exclusive --kill-on-bad-exit=1 \
    --container-image="$ROUND_DAGGER_REMOTE_CONTAINER" \
    --container-mounts="$MOUNTS" \
    --container-workdir="$CONTAINER_ROOT" \
    --output="$ROUND_DAGGER_REMOTE_JOB_ROOT/logs/learner_%j.out" \
    --error="$ROUND_DAGGER_REMOTE_JOB_ROOT/logs/learner_%j.err" \
    env \
      HOME="$CONTAINER_HOME" \
      GR00T_ROOT="$CONTAINER_ROOT/GR00T-VisualSim2Real" \
      ROUND_DAGGER_H100_PYTHON=/opt/viral-h100/bin/python \
      ROUND_DAGGER_WANDB_RUNTIME_DIR=/round_dagger_job/wandb \
      ROUND_DAGGER_CONTAINER_VERIFY_CAMPAIGN_ROOT="$CONTAINER_VERIFY_CAMPAIGN_ROOT" \
      ROUND_DAGGER_CONTAINER_VERIFY_CAMPAIGN_INVENTORY="$CONTAINER_VERIFY_CAMPAIGN_INVENTORY" \
      ROUND_DAGGER_CONTAINER_SUCCESS_MARKER="$CONTAINER_SUCCESS_MARKER" \
      ROUND_DAGGER_LEARNER_NNODES=1 \
      ROUND_DAGGER_LEARNER_NPROC_PER_NODE=8 \
      ROUND_DAGGER_LEARNER_NODE_RANK=0 \
      ROUND_DAGGER_LEARNER_MASTER_ADDR=127.0.0.1 \
      ROUND_DAGGER_LEARNER_MASTER_PORT=29631 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      bash "$CONTAINER_ROOT/containers/h100/trainer_exec.sh" \
      bash "$CONTAINER_ROOT/scripts/round_dagger/run_oci_nrt_learner_container.sh"
status=$?
set -e

if ((status != 0)); then
    echo "H100 learner allocation exited status=$status; durable restart is retained" >&2
    exit "$status"
fi

[[ $ROUND_DAGGER_LEARNER_RESULT == "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"/* ]] || \
    fatal "learner result is outside the output mount"
RESULT_RELATIVE=${ROUND_DAGGER_LEARNER_RESULT#"$ROUND_DAGGER_LEARNER_OUTPUT_DIR"/}
[[ -n $RESULT_RELATIVE && $RESULT_RELATIVE != */* ]] || \
    fatal "learner result is not a canonical basename"
[[ -f $ROUND_DAGGER_REMOTE_LEARNER_OUTPUT/$RESULT_RELATIVE && \
   ! -L $ROUND_DAGGER_REMOTE_LEARNER_OUTPUT/$RESULT_RELATIVE ]] || \
    fatal "learner exited without the canonical result"
COMPLETION_ATTESTATION="$ROUND_DAGGER_REMOTE_LEARNER_OUTPUT/learner_completion_round_${ROUND_PADDED}.json"
[[ -f $COMPLETION_ATTESTATION && ! -L $COMPLETION_ATTESTATION ]] || \
    fatal "learner exited without the canonical completion attestation"
[[ -f $ROUND_DAGGER_REMOTE_SUCCESS_MARKER && \
   ! -L $ROUND_DAGGER_REMOTE_SUCCESS_MARKER ]] || \
    fatal "container exited without the verified success marker"
[[ -z $(find "$ROUND_DAGGER_REMOTE_SUCCESS_MARKER" -perm /222 -print -quit) ]] || \
    fatal "container success marker is writable"
echo "OCI_NRT_ROUND_LEARNER_COMPLETE job_id=${SLURM_JOB_ID:-unknown}"
