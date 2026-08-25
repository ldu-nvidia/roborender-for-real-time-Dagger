#!/bin/bash
# Validation-only staged live gate on one OCI-NRT 8xH100 node.
# These are conservative standalone defaults required at the scheduler file
# boundary.  The audited submission wrapper pins the live OCI-NRT account,
# partition, topology, and time limit explicitly on its sbatch command line.
#SBATCH --account=general_cs_infra
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=rr_h100_validation
set -Eeuo pipefail

PATH=/cm/shared/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
fatal() { echo "FATAL: $*" >&2; exit 2; }
safe_path() {
    [[ $1 =~ ^/[A-Za-z0-9_./-]+$ && $1 != *//* && \
       $1 != */../* && $1 != */./* && $1 != */.. && $1 != */. ]]
}

[[ $# == 27 ]] || fatal "expected exactly 27 validation arguments"
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

[[ ${SLURM_JOB_NUM_NODES:-} == 1 ]] || fatal "validation requires exactly one node"
[[ ${SLURM_GPUS_ON_NODE:-8} =~ (^|[^0-9])8([^0-9]|$)|^8$ ]] || \
    fatal "validation allocation must expose eight H100s"
for path in "$REMOTE_SOURCE" "$CONTAINER" "$REMOTE_CAMPAIGN" "$LOCAL_CAMPAIGN" \
            "$REMOTE_OUTPUT" "$LOCAL_OUTPUT" "$REMOTE_JOB" "$NETRC" \
            "$REMOTE_LOG" "$SNAPSHOT" "$CHECKPOINT" "$SPEC" "$MANIFEST" \
            "$REMOTE_CONTRACT"; do
    safe_path "$path" || fatal "unsafe validation path: $path"
done
for name in SOURCE_SHA CONTAINER_SHA SNAPSHOT_FILE_SHA SNAPSHOT_SHA CHECKPOINT_SHA \
            SPEC_SHA MANIFEST_SHA RUNTIME_SHA CONFIG_SHA CONTRACT_SHA \
            LEARNER_IDENTITY_SHA; do
    [[ ${!name} =~ ^[0-9a-f]{64}$ ]] || fatal "$name is not a lowercase SHA-256"
done
[[ $CAMPAIGN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "validation campaign ID is malformed"
[[ $MACRO_ROUND =~ ^[0-9]+$ ]] || fatal "macro round is invalid"
[[ $PROJECT =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "W&B project token is malformed"
[[ $WANDB_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "canonical W&B run ID is malformed"
[[ -d $REMOTE_SOURCE && ! -L $REMOTE_SOURCE ]] || fatal "sealed source is unsafe"
[[ $REMOTE_CONTRACT == "$REMOTE_SOURCE/configs/round_dagger/validation_learner_1x8h100_2step.json" && \
   -f $REMOTE_CONTRACT && ! -L $REMOTE_CONTRACT ]] || \
    fatal "validation contract is non-canonical"
[[ $(sha256sum "$REMOTE_CONTRACT" | awk '{print $1}') == "$CONTRACT_SHA" ]] || \
    fatal "validation contract digest mismatch"
[[ -f $CONTAINER && ! -L $CONTAINER ]] || fatal "container is unsafe"
[[ $(sha256sum "$CONTAINER" | awk '{print $1}') == "$CONTAINER_SHA" ]] || \
    fatal "container digest mismatch"
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
    fatal "container learner output is not nested below campaign"
OUTPUT_RELATIVE=${LOCAL_OUTPUT#"$LOCAL_CAMPAIGN"/}
[[ -n $OUTPUT_RELATIVE && $OUTPUT_RELATIVE != ../* && $OUTPUT_RELATIVE != */../* ]] || \
    fatal "container learner output relative path is unsafe"
[[ -d $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE && ! -L $REMOTE_CAMPAIGN/$OUTPUT_RELATIVE ]] || \
    fatal "read-only campaign lacks the nested output mount target"
[[ -z $(find "$REMOTE_OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fatal "validation learner output must be empty"
[[ -z $(find "$REMOTE_JOB" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fatal "validation job scratch must be empty"

# Slurm creates the batch stdout/stderr files before this script starts.  They
# live in the dedicated sibling log root and are the only entries permitted at
# this frontier; job scratch therefore remains exactly empty for the runtime.
for entry in "$REMOTE_LOG"/*; do
    [[ -e $entry || -L $entry ]] || continue
    [[ -f $entry && ! -L $entry ]] || fatal "scheduler log root contains an unsafe entry"
    case $(basename "$entry") in
        "slurm_${SLURM_JOB_ID}.out"|"slurm_${SLURM_JOB_ID}.err") ;;
        *) fatal "scheduler log root contains an unexpected entry" ;;
    esac
done

VERIFY_SOURCE="$REMOTE_SOURCE/scripts/round_dagger/verify_sealed_source_snapshot.py"
[[ -f $VERIFY_SOURCE && ! -L $VERIFY_SOURCE ]] || fatal "source verifier is missing"
PYTHONDONTWRITEBYTECODE=1 python3 "$VERIFY_SOURCE" --root "$REMOTE_SOURCE" \
    --expected-inventory-sha256 "$SOURCE_SHA" >/dev/null

mapfile -t DERIVED_IDENTITY < <(
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REMOTE_SOURCE" python3 - \
        "$CAMPAIGN_ID" "$MACRO_ROUND" "$SOURCE_SHA" "$CONFIG_SHA" \
        "$SNAPSHOT_FILE_SHA" "$SNAPSHOT_SHA" "$CHECKPOINT_SHA" "$RUNTIME_SHA" \
        "$CONTAINER_SHA" "$CONTRACT_SHA" <<'PY'
import sys
from round_dagger.learner_identity import learner_identity_sha256, learner_wandb_run_id
(campaign, macro_round, code, config, replay_file, replay, parent, parent_runtime,
 container, contract) = sys.argv[1:]
identity = learner_identity_sha256(
    campaign_id=campaign, macro_round=int(macro_round), learner_code_sha256=code,
    learner_config_sha256=config, replay_snapshot_file_sha256=replay_file,
    replay_snapshot_sha256=replay, parent_checkpoint_sha256=parent,
    parent_runtime_policy_sha256=parent_runtime,
    runtime_container_sha256=container, learner_contract_sha256=contract,
)
print(identity)
print(learner_wandb_run_id(
    campaign_id=campaign, macro_round=int(macro_round),
    learner_identity_sha256=identity,
))
PY
)
[[ ${DERIVED_IDENTITY[0]:-} == "$LEARNER_IDENTITY_SHA" && \
   ${DERIVED_IDENTITY[1]:-} == "$WANDB_RUN_ID" && \
   ${#DERIVED_IDENTITY[@]} == 2 ]] || fatal "validation learner identity mismatch"

MOUNTS="$REMOTE_SOURCE:/workspace/fast_roborender:ro"
MOUNTS+=",$REMOTE_CAMPAIGN:$LOCAL_CAMPAIGN:ro"
MOUNTS+=",$REMOTE_OUTPUT:$LOCAL_OUTPUT"
MOUNTS+=",$REMOTE_JOB:/round_dagger_validation_job"
MOUNTS+=",$NETRC:/home/ldu/.netrc:ro"
MOUNTS+=",$CONTAINER:/round_dagger_validation_container.sqsh:ro"

COMMON_ENV=(
    HOME=/home/ldu
    GR00T_ROOT=/workspace/fast_roborender/GR00T-VisualSim2Real
    ROUND_DAGGER_H100_PYTHON=/opt/viral-h100/bin/python
    ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT="$LOCAL_CAMPAIGN"
    ROUND_DAGGER_VALIDATION_OUTPUT_ROOT="$LOCAL_OUTPUT"
    ROUND_DAGGER_VALIDATION_JOB_ROOT=/round_dagger_validation_job
    ROUND_DAGGER_VALIDATION_SOURCE_INVENTORY_SHA256="$SOURCE_SHA"
    ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT="$SNAPSHOT"
    ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_FILE_SHA256="$SNAPSHOT_FILE_SHA"
    ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_SHA256="$SNAPSHOT_SHA"
    ROUND_DAGGER_VALIDATION_POLICY_CHECKPOINT="$CHECKPOINT"
    ROUND_DAGGER_VALIDATION_POLICY_SHA256="$CHECKPOINT_SHA"
    ROUND_DAGGER_VALIDATION_POLICY_SPEC="$SPEC"
    ROUND_DAGGER_VALIDATION_POLICY_SPEC_SHA256="$SPEC_SHA"
    ROUND_DAGGER_VALIDATION_POLICY_MANIFEST="$MANIFEST"
    ROUND_DAGGER_VALIDATION_POLICY_MANIFEST_SHA256="$MANIFEST_SHA"
    ROUND_DAGGER_VALIDATION_RUNTIME_POLICY_SHA256="$RUNTIME_SHA"
    ROUND_DAGGER_VALIDATION_LEARNER_CONFIG_SHA256="$CONFIG_SHA"
    ROUND_DAGGER_VALIDATION_CONTRACT=/workspace/fast_roborender/configs/round_dagger/validation_learner_1x8h100_2step.json
    ROUND_DAGGER_VALIDATION_CONTRACT_SHA256="$CONTRACT_SHA"
    ROUND_DAGGER_VALIDATION_LEARNER_IDENTITY_SHA256="$LEARNER_IDENTITY_SHA"
    ROUND_DAGGER_VALIDATION_CAMPAIGN_ID="$CAMPAIGN_ID"
    ROUND_DAGGER_VALIDATION_MACRO_ROUND="$MACRO_ROUND"
    ROUND_DAGGER_VALIDATION_CONTAINER_HOST_PATH=/round_dagger_validation_container.sqsh
    ROUND_DAGGER_VALIDATION_CONTAINER_SHA256="$CONTAINER_SHA"
    ROUND_DAGGER_WANDB_PROJECT="$PROJECT"
    ROUND_DAGGER_WANDB_ENTITY=nvidia
    ROUND_DAGGER_WANDB_EXPECTED_EMAIL=ldu@nvidia.com
    ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE=team
    ROUND_DAGGER_VALIDATION_WANDB_RUN_ID="$WANDB_RUN_ID"
    CUBLAS_WORKSPACE_CONFIG=:4096:8
)

# Gate A: one visible H100, exact nested ro/rw Pyxis mounts, CUDA/torch/GR00T,
# and a live read-only W&B identity/team query.  W&B runtime is mounted under
# /round_dagger_validation_job, never under the returned learner output.
srun --nodes=1 --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=8 \
    --mem=32G --exact --exclusive --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/fast_roborender \
    --output="$REMOTE_LOG/preflight_%j.out" \
    --error="$REMOTE_LOG/preflight_%j.err" \
    env "${COMMON_ENV[@]}" \
    bash /workspace/fast_roborender/containers/h100/trainer_exec.sh \
    bash /workspace/fast_roborender/scripts/round_dagger/run_oci_nrt_h100_validation_container.sh \
    preflight

# Gates B/C: every launch below is torchrun with exactly eight CUDA ranks.
# It runs uninterrupted 2-update training, intentionally stops a second run at
# update 1, resumes that same W&B run with resume=must to update 2, then requires
# semantic/result identity and authenticated remote telemetry readback.
srun --nodes=1 --ntasks=1 --gpus=8 --gpus-per-task=8 --cpus-per-task=64 \
    --mem=0 --exact --exclusive --kill-on-bad-exit=1 \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/fast_roborender \
    --output="$REMOTE_LOG/learner_%j.out" \
    --error="$REMOTE_LOG/learner_%j.err" \
    env "${COMMON_ENV[@]}" \
    ROUND_DAGGER_LEARNER_NNODES=1 ROUND_DAGGER_LEARNER_NPROC_PER_NODE=8 \
    ROUND_DAGGER_LEARNER_BATCH_SIZE=1 ROUND_DAGGER_LEARNER_MAX_STEPS=2 \
    ROUND_DAGGER_LEARNER_SAVE_EVERY=1 \
    bash /workspace/fast_roborender/containers/h100/trainer_exec.sh \
    bash /workspace/fast_roborender/scripts/round_dagger/run_oci_nrt_h100_validation_container.sh \
    learner-sequence

ATTESTATION="$REMOTE_OUTPUT/success_attestation.json"
[[ -f $ATTESTATION && ! -L $ATTESTATION ]] || fatal "success attestation is missing"
[[ $(stat -c %a "$ATTESTATION") == 444 ]] || fatal "success attestation is not read-only"
echo "OCI_NRT_H100_VALIDATION_JOB_PASS job_id=${SLURM_JOB_ID:-unknown} attestation=$ATTESTATION"
