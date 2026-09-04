#!/usr/bin/env bash
# Trusted coordinator boundary for the explicit Lepton H200 learner backend.
set -Eeuo pipefail

fatal() { echo "production Lepton learner hook failed: $*" >&2; exit 2; }

[[ ${ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH:-} == 1 ]] || \
  fatal "trusted sealed train dispatch was not established"
[[ ${ROUND_DAGGER_LEARNER_BACKEND:-} == lepton_job_v1 ]] || \
  fatal "learner backend is not exactly lepton_job_v1"
[[ ${ROUND_DAGGER_TRUSTED_HOOK_SOURCE_ROOT:-} == /* ]] || \
  fatal "trusted hook source root is missing"
[[ ${ROUND_DAGGER_TRUSTED_HOOK_SOURCE_INVENTORY_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || \
  fatal "trusted hook inventory identity is malformed"

required=(
  ROUND_DAGGER_CAMPAIGN_ROOT
  ROUND_DAGGER_CAMPAIGN_ID
  ROUND_DAGGER_MACRO_ROUND
  ROUND_DAGGER_REMOTE_HANDOFF_ROOT
  ROUND_DAGGER_REPLAY_SNAPSHOT
  ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256
  ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256
  ROUND_DAGGER_POLICY_CHECKPOINT
  ROUND_DAGGER_POLICY_SHA256
  ROUND_DAGGER_RUNTIME_POLICY_SHA256
  ROUND_DAGGER_BASE_POLICY_SPEC
  ROUND_DAGGER_BETA
  ROUND_DAGGER_BETA_SHA256
  ROUND_DAGGER_SCENE_SAMPLE_MANIFEST
  ROUND_DAGGER_LEARNER_OUTPUT_DIR
  ROUND_DAGGER_LEARNER_RESULT
  ROUND_DAGGER_LEARNER_CODE_SHA256
  ROUND_DAGGER_LEARNER_CONFIG_SHA256
  ROUND_DAGGER_LEARNER_CONTRACT_SHA256
  ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256
  ROUND_DAGGER_LEARNER_IDENTITY_SHA256
  ROUND_DAGGER_LEPTON_BACKEND_CONTRACT
  ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256
  ROUND_DAGGER_LEPTON_REMOTE_BACKEND_CONTRACT
  ROUND_DAGGER_LEPTON_CONTROL_EXEC_PROGRAM
  ROUND_DAGGER_LEPTON_SDK_PYTHON
  ROUND_DAGGER_LEARNER_BATCH_SIZE
  ROUND_DAGGER_LEARNER_MAX_STEPS
  ROUND_DAGGER_LEARNER_SAVE_EVERY
  ROUND_DAGGER_LEARNER_LEARNING_RATE
  ROUND_DAGGER_LEARNER_WEIGHT_DECAY
  ROUND_DAGGER_LEARNER_SEED
  ROUND_DAGGER_LEARNER_PARTITION_SEED
  ROUND_DAGGER_TARGET_ENVIRONMENTS
  ROUND_DAGGER_COHORT_COUNT
  ROUND_DAGGER_ENVIRONMENTS_PER_COHORT
  ROUND_DAGGER_STEPS_PER_ENVIRONMENT
  ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND
  ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT
  ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256
  ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256
  ROUND_DAGGER_HELDOUT_EVAL_INTERVAL
  ROUND_DAGGER_WANDB_PROJECT
  ROUND_DAGGER_WANDB_ENTITY
  ROUND_DAGGER_WANDB_RUN_ID
  ROUND_DAGGER_WANDB_EXPECTED_EMAIL
  ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE
)
for name in "${required[@]}"; do
  [[ -n ${!name:-} ]] || fatal "missing required orchestrator environment: $name"
done

source_root=$ROUND_DAGGER_TRUSTED_HOOK_SOURCE_ROOT
bridge="$source_root/scripts/round_dagger/production_lepton_learner_bridge.py"
[[ -f $bridge && ! -L $bridge ]] || fatal "sealed production bridge is missing or unsafe"

python=${ROUND_DAGGER_CONTROL_PYTHON:-}
[[ $python == /* && -x $python ]] || \
  fatal "ROUND_DAGGER_CONTROL_PYTHON is not an absolute executable"
python_real=$(realpath -e -- "$python") || fatal "control Python cannot be resolved"
[[ -f $python_real && -x $python_real ]] || fatal "resolved control Python is unsafe"

sdk_python=${ROUND_DAGGER_LEPTON_SDK_PYTHON:-}
[[ $sdk_python == /* && -x $sdk_python ]] || \
  fatal "ROUND_DAGGER_LEPTON_SDK_PYTHON is not an absolute executable"
sdk_python_real=$(realpath -e -- "$sdk_python") || fatal "Lepton SDK Python cannot be resolved"
[[ -f $sdk_python_real && -x $sdk_python_real ]] || \
  fatal "resolved Lepton SDK Python is unsafe"

handoff=${ROUND_DAGGER_REMOTE_HANDOFF_ROOT:-}
[[ $handoff == /* ]] || fatal "remote handoff root is not absolute"
case "$handoff" in
  "$ROUND_DAGGER_CAMPAIGN_ROOT"|"$ROUND_DAGGER_CAMPAIGN_ROOT"/*)
    fatal "Lepton handoff root must be outside the mutable campaign"
    ;;
esac

exec "$python" "$bridge"
