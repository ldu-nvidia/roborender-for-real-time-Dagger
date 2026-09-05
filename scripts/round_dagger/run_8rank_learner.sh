#!/usr/bin/env bash
# Advance one BC/DAgger learner stage to one validation frontier on all 8 H200s.
set -Eeuo pipefail

fatal() { echo "native H200 learner: $*" >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T_ROOT=$ROOT/GR00T-VisualSim2Real
CAMPAIGN=/workspace/project/roborender/data
MOTIONS=$GR00T_ROOT/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775
LEARNER_PYTHON=${ROUND_DAGGER_LEARNER_PYTHON:-/workspace/project/roborender/envs/h200_learner/bin/python}

required=(
  ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_CAMPAIGN_ID
  ROUND_DAGGER_GENERATION_ID ROUND_DAGGER_MACRO_ROUND
  ROUND_DAGGER_POLICY_CHECKPOINT ROUND_DAGGER_POLICY_SHA256
  ROUND_DAGGER_REPLAY_SNAPSHOT ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256
  ROUND_DAGGER_LEARNER_BUNDLE ROUND_DAGGER_LEARNER_BUNDLE_SHA256
  ROUND_DAGGER_LEARNER_OUTPUT_DIR ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES
  ROUND_DAGGER_VALIDATION_INTERVAL_OPTIMIZER_EXAMPLES
  ROUND_DAGGER_LEARNER_IDENTITY_SHA256 ROUND_DAGGER_RUNTIME_ENVIRONMENT_SHA256
  ROUND_DAGGER_LEARNER_CONTRACT_SHA256
)
for name in "${required[@]}"; do
  [[ -n ${!name:-} ]] || fatal "missing required environment: $name"
done
[[ $ROUND_DAGGER_CAMPAIGN_ROOT == "$CAMPAIGN" && \
   $ROUND_DAGGER_CAMPAIGN_ID == dagger-redo-v001 ]] || fatal "campaign identity differs"
[[ -x $LEARNER_PYTHON && -d $GR00T_ROOT && ! -L $GR00T_ROOT ]] || \
  fatal "local learner runtime is unavailable"
for value in ROUND_DAGGER_POLICY_SHA256 ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256 \
             ROUND_DAGGER_LEARNER_BUNDLE_SHA256 ROUND_DAGGER_LEARNER_IDENTITY_SHA256 \
             ROUND_DAGGER_RUNTIME_ENVIRONMENT_SHA256 ROUND_DAGGER_LEARNER_CONTRACT_SHA256; do
  [[ ${!value} =~ ^[0-9a-f]{64}$ ]] || fatal "$value is not a SHA-256"
done
[[ $ROUND_DAGGER_MACRO_ROUND =~ ^[0-5]$ ]] || fatal "macro round must be 0..5"
case $ROUND_DAGGER_MACRO_ROUND in
  0) expected_generation=gen-000-bc; round_name=BC ;;
  1) expected_generation=gen-001-dagger-beta-0p80; round_name=dagger_1 ;;
  2) expected_generation=gen-002-dagger-beta-0p60; round_name=dagger_2 ;;
  3) expected_generation=gen-003-dagger-beta-0p40; round_name=dagger_3 ;;
  4) expected_generation=gen-004-dagger-beta-0p20; round_name=dagger_4 ;;
  5) expected_generation=gen-005-dagger-beta-0p00; round_name=dagger_5 ;;
esac
[[ $ROUND_DAGGER_GENERATION_ID == "$expected_generation" ]] || \
  fatal "generation does not match macro round"
[[ $ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES =~ ^[1-9][0-9]*$ ]] || \
  fatal "target optimizer examples must be positive"
[[ $ROUND_DAGGER_VALIDATION_INTERVAL_OPTIMIZER_EXAMPLES =~ ^[1-9][0-9]*$ ]] || \
  fatal "validation interval optimizer examples must be positive"

generation_root=$CAMPAIGN/$round_name
expected_output=$generation_root/checkpoints/candidates
[[ $ROUND_DAGGER_LEARNER_OUTPUT_DIR == "$expected_output" ]] || \
  fatal "learner output must be the generation candidate directory"
for path in "$ROUND_DAGGER_POLICY_CHECKPOINT" "$ROUND_DAGGER_REPLAY_SNAPSHOT" \
            "$ROUND_DAGGER_LEARNER_BUNDLE"; do
  [[ $path == /* && -f $path && ! -L $path ]] || fatal "unsafe learner input: $path"
done
[[ $(sha256sum "$ROUND_DAGGER_POLICY_CHECKPOINT" | awk '{print $1}') == \
   "$ROUND_DAGGER_POLICY_SHA256" ]] || fatal "parent checkpoint digest differs"
[[ $(sha256sum "$ROUND_DAGGER_REPLAY_SNAPSHOT" | awk '{print $1}') == \
   "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256" ]] || fatal "replay snapshot digest differs"
[[ $(sha256sum "$ROUND_DAGGER_LEARNER_BUNDLE" | awk '{print $1}') == \
   "$ROUND_DAGGER_LEARNER_BUNDLE_SHA256" ]] || fatal "learner bundle digest differs"

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,compute_cap \
  --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} == 8 ]] || fatal "exactly eight physical GPUs are required"
for index in "${!gpu_rows[@]}"; do
  [[ ${gpu_rows[$index]} == "$index, NVIDIA H200, 9.0" ]] || \
    fatal "GPU $index is not an H200 with compute capability 9.0"
done

mkdir -p "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"
[[ -d $ROUND_DAGGER_LEARNER_OUTPUT_DIR && ! -L $ROUND_DAGGER_LEARNER_OUTPUT_DIR ]] || \
  fatal "learner output directory is unsafe"
printf -v frontier_name 'frontier-examples-%012d.pt' \
  "$((10#$ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES))"
frontier=$ROUND_DAGGER_LEARNER_OUTPUT_DIR/$frontier_name
restart=$ROUND_DAGGER_LEARNER_OUTPUT_DIR/restart.pt
telemetry=$generation_root/telemetry/learner.jsonl
mkdir -p "$(dirname "$telemetry")"

# Validate the bundle and train lineage before mounting held-out paths away.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT:$GR00T_ROOT" "$LEARNER_PYTHON" - \
  "$ROUND_DAGGER_LEARNER_BUNDLE" "$ROUND_DAGGER_GENERATION_ID" \
  "$ROUND_DAGGER_MACRO_ROUND" "$ROUND_DAGGER_POLICY_SHA256" \
  "$ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES" \
  "$ROUND_DAGGER_VALIDATION_INTERVAL_OPTIMIZER_EXAMPLES" <<'PYVERIFY'
import sys
from round_dagger.learner import LearnerBundleConfig
bundle, generation, round_number, parent, target, interval = sys.argv[1:]
config = LearnerBundleConfig.load(bundle)
if config.generation_id != generation or config.macro_round != int(round_number):
    raise SystemExit("learner bundle generation differs")
if config.initial_checkpoint_sha256 != parent or config.model_lineage_sha256 != parent:
    raise SystemExit("learner bundle parent differs")
if int(target) > config.target_optimizer_examples:
    raise SystemExit("frontier exceeds stage maximum")
if int(target) % (config.batch_size * 8):
    raise SystemExit("frontier is not aligned to the eight-rank global batch")
if int(interval) != config.checkpoint_every_optimizer_examples:
    raise SystemExit("validation interval differs from learner bundle")
PYVERIFY

command=(
  "$LEARNER_PYTHON" -m torch.distributed.run
  --nnodes 1 --nproc-per-node 8 --master-addr 127.0.0.1
  --master-port "${ROUND_DAGGER_LEARNER_MASTER_PORT:-29631}"
  "$ROOT/scripts/round_dagger/train_learner.py"
  --bundle "$ROUND_DAGGER_LEARNER_BUNDLE"
  --snapshot "$ROUND_DAGGER_REPLAY_SNAPSHOT"
  --artifact-root "$CAMPAIGN"
  --output-dir "$ROUND_DAGGER_LEARNER_OUTPUT_DIR"
  --frontier-output "$frontier"
  --model-factory round_dagger.gr00t_policy:create_vision_actor_obj_pred
  --adapter-factory round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory
  --initial-checkpoint "$ROUND_DAGGER_POLICY_CHECKPOINT"
  --expected-snapshot-file-sha256 "$ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
  --learner-identity-sha256 "$ROUND_DAGGER_LEARNER_IDENTITY_SHA256"
  --runtime-container-sha256 "$ROUND_DAGGER_RUNTIME_ENVIRONMENT_SHA256"
  --learner-contract-sha256 "$ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
  --target-optimizer-examples "$ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES"
  --checkpoint-every-optimizer-examples \
    "$ROUND_DAGGER_VALIDATION_INTERVAL_OPTIMIZER_EXAMPLES"
  --device cuda --telemetry-jsonl "$telemetry" --no-publish-result
)
[[ ! -e $restart ]] || command+=(--resume-from "$restart")

# The optimizer can see train replay and model code only. Validation/test data,
# all source trajectories, and validation outputs are hidden in its namespace.
masks=(
  "$MOTIONS" "$CAMPAIGN/BC/contracts"
  "$CAMPAIGN/validation" "$CAMPAIGN/test"
  "$ROOT/configs/round_dagger/native_h200_group_registry.json"
  "$ROOT/configs/round_dagger/native_h200_group_registry_audit.json"
  "$ROOT/configs/round_dagger/native_h200_validation_panel.json"
  "$ROOT/configs/round_dagger/native_h200_validation_panel_audit.json"
)
for mask in "${masks[@]}"; do
  [[ -e $mask && ! -L $mask ]] || fatal "mask target is unsafe: $mask"
done

unshare --mount --propagation private --pid --fork --kill-child \
  --mount-proc=/proc -- /bin/bash -Eeuo pipefail -s -- \
  "$ROOT" "$GR00T_ROOT" "${#masks[@]}" "${masks[@]}" \
  "${command[@]}" <<'NAMESPACE'
source_root=$1; gr00t_root=$2; count=$3; shift 3
mask_root=$(mktemp -d /tmp/native-h200-learner-mask.XXXXXX)
chmod 000 "$mask_root"
masked_targets=()
for ((index=0; index<count; index++)); do
  target=$1; shift
  masked_targets+=("$target")
  if [[ -d $target ]]; then
    mount --bind "$mask_root" "$target"
  else
    mount --bind /dev/null "$target"
  fi
  mount -o remount,bind,ro,nosuid,nodev,noexec "$target"
done
rmdir "$mask_root"
exec setpriv --no-new-privs --bounding-set=-sys_admin,-dac_override,-dac_read_search \
  --inh-caps=-all --ambient-caps=-all -- \
  env -i PATH=/usr/bin:/bin PYTHONPATH="$source_root:$gr00t_root" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /bin/bash -Eeuo pipefail -c '
      count=$1; shift
      for ((index=0; index<count; index++)); do
        target=$1; shift
        [[ ! -r $target && ! -x $target ]] || {
          echo "native H200 learner: held-out mask is readable: $target" >&2
          exit 2
        }
      done
      exec "$@"
    ' native-h200-learner "$count" "${masked_targets[@]}" "${@}"
NAMESPACE

[[ -f $frontier && ! -L $frontier ]] || fatal "frontier checkpoint was not published"
frontier_sha=$(sha256sum "$frontier" | awk '{print $1}')
echo "NATIVE_H200_LEARNER_FRONTIER_OK generation=$ROUND_DAGGER_GENERATION_ID examples=$ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES checkpoint=$frontier sha256=$frontier_sha"
