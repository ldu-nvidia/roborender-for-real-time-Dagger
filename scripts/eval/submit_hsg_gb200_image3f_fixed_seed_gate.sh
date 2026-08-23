#!/bin/bash
# Submit the resource-minimal, production Image3F-in-loop fixed-seed gate.
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:?TRAIN_RUN_DIR is required}
BASELINE_CKPT=${BASELINE_CKPT:-$TRAIN_RUN_DIR/initial_policy.pt}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$TRAIN_RUN_DIR/config.yaml}
RENDERER_CONTAINER=${RENDERER_CONTAINER:-$STAGE_ROOT/containers/roborender-image3f-arm64-sm100-cu130-trt10.13.3.9.sqsh}
TRAINER_CONTAINER=${TRAINER_CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
EXPECTED_RENDERER_CONTAINER_SHA256=${EXPECTED_RENDERER_CONTAINER_SHA256:-4c8261f246a650399c2cdf3d3567e1e14a643d3d65e239ff5a7f9f52442da963}
EXPECTED_TRAINER_CONTAINER_SHA256=${EXPECTED_TRAINER_CONTAINER_SHA256:-65ae3fec98e1bb9097bb852a28604ac43ba0bfeb1b169708e16ce1b28343b852}
IMAGE3F_ROOT=${IMAGE3F_ROOT:-$STAGE_ROOT/assets/roborender_image_model_accelerated}
ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models}
IMAGE3F_GENERATED_ROOT=${IMAGE3F_GENERATED_ROOT:-$STAGE_ROOT/runtime/image3f-generated}
ANCHORS=${ANCHORS:-$STAGE_ROOT/assets/anchor_pool_2048}
TEACHER_CKPT=${TEACHER_CKPT:-$STAGE_ROOT/assets/policies/expert_labeler/model_step_044500.pt}
EXPECTED_TEACHER_CKPT_SHA256=${EXPECTED_TEACHER_CKPT_SHA256:-943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38}
RESNET18_WEIGHTS=${RESNET18_WEIGHTS:-$STAGE_ROOT/assets/torch/hub/checkpoints/resnet18-f37072fd.pth}
EXPECTED_RESNET18_SHA256=${EXPECTED_RESNET18_SHA256:-f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec}
EXPECTED_IMAGE3F_RELEASE_ID=${EXPECTED_IMAGE3F_RELEASE_ID:-roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0}
EXPECTED_IMAGE3F_ADAPTER_SHA256=${EXPECTED_IMAGE3F_ADAPTER_SHA256:-c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588}
CAMERA_CALIBRATION_ACCEPTANCE=${CAMERA_CALIBRATION_ACCEPTANCE:-}
CAMERA_CALIBRATION_RUN_ID=${CAMERA_CALIBRATION_RUN_ID:-}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}
WANDB_ONLINE_VALIDATION=${WANDB_ONLINE_VALIDATION:-$STAGE_ROOT/runtime/validation/wandb_online_smoke_nvidia_acceptance.json}
FIXED_EVAL_SEEDS=${FIXED_EVAL_SEEDS:-1729,3253,4519,5987,7411,8863,10193,12203}
FIXED_EVAL_NUM_ENVS=${FIXED_EVAL_NUM_ENVS:-32}
EXPECTED_RENDERERS=${EXPECTED_RENDERERS:-7}
MIN_SUCCESS_DELTA=${MIN_SUCCESS_DELTA:-0.02}
PROJ=${PROJ:-roborender_dagger_hsg_image3f_fixed_seed_eval}
RUN_ID=${RUN_ID:-hsg_image3f_fixed_eval_$(date -u +%Y%m%dT%H%M%SZ)}
EVAL_RUN_ID=${EVAL_RUN_ID:-$RUN_ID}
EVAL_DIR=${EVAL_DIR:-$TRAIN_RUN_DIR/image3f_fixed_seed_eval/$RUN_ID}
FLEET_DIR=${FLEET_DIR:-$EVAL_DIR/fleet}
DRY_RUN=${DRY_RUN:-0}
SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/local/apps/slurm/current/bin}

fatal() { echo "FATAL: $*" >&2; exit 2; }
case "$DRY_RUN" in 0|1) ;; *) fatal "DRY_RUN must be 0 or 1" ;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fatal "RUN_ID is malformed"
[[ "$EVAL_RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || fatal "EVAL_RUN_ID is malformed"
[ -n "$CAMERA_CALIBRATION_ACCEPTANCE" ] || \
    fatal "CAMERA_CALIBRATION_ACCEPTANCE must explicitly select a versioned calibration acceptance"
CAMERA_CALIBRATION_ACCEPTANCE_BASENAME=${CAMERA_CALIBRATION_ACCEPTANCE##*/}
case "$CAMERA_CALIBRATION_ACCEPTANCE_BASENAME" in
    camera_calibration_acceptance_*.json) ;;
    *) fatal "CAMERA_CALIBRATION_ACCEPTANCE must name camera_calibration_acceptance_<run-id>.json" ;;
esac
DERIVED_CAMERA_CALIBRATION_RUN_ID=${CAMERA_CALIBRATION_ACCEPTANCE_BASENAME#camera_calibration_acceptance_}
DERIVED_CAMERA_CALIBRATION_RUN_ID=${DERIVED_CAMERA_CALIBRATION_RUN_ID%.json}
[[ "$DERIVED_CAMERA_CALIBRATION_RUN_ID" =~ ^[A-Za-z0-9_-]{1,128}$ ]] || \
    fatal "versioned camera calibration acceptance has a malformed run ID"
if [ -n "$CAMERA_CALIBRATION_RUN_ID" ]; then
    [ "$CAMERA_CALIBRATION_RUN_ID" = "$DERIVED_CAMERA_CALIBRATION_RUN_ID" ] || \
        fatal "CAMERA_CALIBRATION_RUN_ID does not match the selected acceptance filename"
else
    CAMERA_CALIBRATION_RUN_ID=$DERIVED_CAMERA_CALIBRATION_RUN_ID
fi
[ "$CAMERA_CALIBRATION_ACCEPTANCE" = \
  "$STAGE_ROOT/runtime/validation/camera_calibration_acceptance_${CAMERA_CALIBRATION_RUN_ID}.json" ] || \
    fatal "CAMERA_CALIBRATION_ACCEPTANCE must be the exact versioned validation artifact"
[ "$FIXED_EVAL_NUM_ENVS" -eq 32 ] || fatal "production gate is pinned to 32 environments"
[ "$EXPECTED_RENDERERS" -eq 7 ] || fatal "production gate is pinned to seven renderers"
[ "$MIN_SUCCESS_DELTA" = 0.02 ] || fatal "minimum success delta is pinned to 0.02"
[ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || fatal "W&B account must be ldu@nvidia.com"
[ "$WANDB_ENTITY" = nvidia ] || fatal "W&B entity must be the authorized nvidia team"
[ "$WANDB_AUTHORIZATION_SCOPE" = team ] || fatal "W&B authorization scope must be team"
[ "$RENDERER_CONTAINER" != "$TRAINER_CONTAINER" ] || fatal "renderer and trainer images must differ"
for command in sbatch scontrol; do
    [ -x "$SLURM_BIN_DIR/$command" ] || fatal "missing HSG Slurm command $command"
done
for path in "$ROOT" "$GR00T_ROOT" "$TRAIN_RUN_DIR" "$IMAGE3F_ROOT" \
            "$ROBORENDER_MODEL_BASE" "$IMAGE3F_GENERATED_ROOT" "$ANCHORS"; do
    [ -d "$path" ] || fatal "missing required directory $path"
done
for path in "$BASELINE_CKPT" "$REFERENCE_CONFIG" "$RENDERER_CONTAINER" \
            "$TRAINER_CONTAINER" "$TEACHER_CKPT" "$RESNET18_WEIGHTS" \
            "$CAMERA_CALIBRATION_ACCEPTANCE" "$WANDB_ONLINE_VALIDATION"; do
    [ -s "$path" ] || fatal "missing required file $path"
done
[ "$(sha256sum "$RENDERER_CONTAINER" | awk '{print $1}')" = "$EXPECTED_RENDERER_CONTAINER_SHA256" ] || \
    fatal "renderer container hash mismatch"
[ "$(sha256sum "$TRAINER_CONTAINER" | awk '{print $1}')" = "$EXPECTED_TRAINER_CONTAINER_SHA256" ] || \
    fatal "trainer container hash mismatch"
[ "$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')" = "$EXPECTED_TEACHER_CKPT_SHA256" ] || \
    fatal "teacher checkpoint hash mismatch"
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || \
    fatal "ResNet18 hash mismatch"

read_revision() {
    local marker=$1 value
    [ -s "$marker" ] || fatal "missing immutable revision marker $marker"
    value=$(tr -d '\r\n' < "$marker")
    [[ "$value" =~ ^[0-9a-f]{40}$ ]] || fatal "malformed revision marker $marker"
    printf '%s' "$value"
}
FAST_REVISION=$(read_revision "$ROOT/.source_revision")
GR00T_REVISION=$(read_revision "$GR00T_ROOT/.source_revision")
EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=$(sha256sum "$CAMERA_CALIBRATION_ACCEPTANCE" | awk '{print $1}')
python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
    --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
    --expected-run-id "$CAMERA_CALIBRATION_RUN_ID" \
    --expected-fast-roborender-revision "$FAST_REVISION" \
    --expected-gr00t-revision "$GR00T_REVISION" \
    --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" >/dev/null

NETRC="$STAGE_ROOT/runtime/home/.netrc"
[ -s "$NETRC" ] || fatal "W&B credential missing: $NETRC"
[ "$(stat -c %u "$NETRC")" -eq "$(id -u)" ] || fatal "W&B netrc owner mismatch"
case "$(stat -c %a "$NETRC")" in 400|600) ;; *) fatal "W&B netrc must be 0400 or 0600" ;; esac
WANDB_ENTITY=$(python3 "$ROOT/scripts/eval/wandb_team_contract.py" \
    --acceptance "$WANDB_ONLINE_VALIDATION" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
    --authorization-scope "$WANDB_AUTHORIZATION_SCOPE") || \
    fatal "authorized-team W&B acceptance is invalid"
WANDB_ONLINE_VALIDATION_SHA256=$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')

[ ! -e "$EVAL_DIR" ] || fatal "refusing to reuse evaluation directory $EVAL_DIR"
mkdir -p "$EVAL_DIR/logs"
EVAL_MANIFEST="$EVAL_DIR/manifest.json"
BASELINE_SHA256=$(sha256sum "$BASELINE_CKPT" | awk '{print $1}')
REFERENCE_CONFIG_SHA256=$(sha256sum "$REFERENCE_CONFIG" | awk '{print $1}')
python3 "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py" prepare \
    --train-run-dir "$TRAIN_RUN_DIR" --baseline-checkpoint "$BASELINE_CKPT" \
    --reference-config "$REFERENCE_CONFIG" --seeds "$FIXED_EVAL_SEEDS" \
    --num-envs "$FIXED_EVAL_NUM_ENVS" \
    --expected-baseline-sha256 "$BASELINE_SHA256" \
    --expected-reference-config-sha256 "$REFERENCE_CONFIG_SHA256" \
    --output "$EVAL_MANIFEST" >/dev/null
EXPECTED_EVAL_MANIFEST_SHA256=$(sha256sum "$EVAL_MANIFEST" | awk '{print $1}')
PREFLIGHT="$EVAL_DIR/hsg_image3f_preflight.json"
python3 "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py" preflight \
    --root "$ROOT" --gr00t-root "$GR00T_ROOT" --fast-revision "$FAST_REVISION" \
    --gr00t-revision "$GR00T_REVISION" --renderer-container "$RENDERER_CONTAINER" \
    --renderer-container-sha256 "$EXPECTED_RENDERER_CONTAINER_SHA256" \
    --trainer-container "$TRAINER_CONTAINER" \
    --trainer-container-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --image-root "$IMAGE3F_ROOT" --model-root "$ROBORENDER_MODEL_BASE" \
    --generated-root "$IMAGE3F_GENERATED_ROOT" --anchors "$ANCHORS" \
    --anchor-min-files 2048 --teacher "$TEACHER_CKPT" \
    --teacher-sha256 "$EXPECTED_TEACHER_CKPT_SHA256" \
    --release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
    --adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    --camera-acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
    --camera-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" \
    --camera-calibration-run-id "$CAMERA_CALIBRATION_RUN_ID" \
    --resnet18 "$RESNET18_WEIGHTS" --resnet18-sha256 "$EXPECTED_RESNET18_SHA256" \
    --wandb-acceptance "$WANDB_ONLINE_VALIDATION" \
    --wandb-acceptance-sha256 "$WANDB_ONLINE_VALIDATION_SHA256" \
    --fixed-seed-manifest "$EVAL_MANIFEST" \
    --fixed-seed-manifest-sha256 "$EXPECTED_EVAL_MANIFEST_SHA256" \
    --train-run-dir "$TRAIN_RUN_DIR" --expected-seeds "$FIXED_EVAL_SEEDS" \
    --expected-num-envs "$FIXED_EVAL_NUM_ENVS" > "$PREFLIGHT"
EXPECTED_PREFLIGHT_SHA256=$(sha256sum "$PREFLIGHT" | awk '{print $1}')
EXPECTED_FAST_SOURCE_TREE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fast_roborender_source_tree_sha256"])' "$PREFLIGHT")
EXPECTED_GR00T_SOURCE_TREE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gr00t_source_tree_sha256"])' "$PREFLIGHT")
EXPECTED_IMAGE3F_ENGINE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["engine_sha256"])' "$PREFLIGHT")

EXPORTS="PATH=$SLURM_BIN_DIR:/usr/local/bin:/usr/bin:/bin,SLURM_BIN_DIR=$SLURM_BIN_DIR,STAGE_ROOT=$STAGE_ROOT,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,EVAL_DIR=$EVAL_DIR,EVAL_RUN_ID=$EVAL_RUN_ID,FLEET_DIR=$FLEET_DIR,EVAL_MANIFEST=$EVAL_MANIFEST,EXPECTED_EVAL_MANIFEST_SHA256=$EXPECTED_EVAL_MANIFEST_SHA256,PREFLIGHT=$PREFLIGHT,EXPECTED_PREFLIGHT_SHA256=$EXPECTED_PREFLIGHT_SHA256,RENDERER_CONTAINER=$RENDERER_CONTAINER,EXPECTED_RENDERER_CONTAINER_SHA256=$EXPECTED_RENDERER_CONTAINER_SHA256,TRAINER_CONTAINER=$TRAINER_CONTAINER,EXPECTED_TRAINER_CONTAINER_SHA256=$EXPECTED_TRAINER_CONTAINER_SHA256,IMAGE3F_ROOT=$IMAGE3F_ROOT,ROBORENDER_MODEL_BASE=$ROBORENDER_MODEL_BASE,IMAGE3F_GENERATED_ROOT=$IMAGE3F_GENERATED_ROOT,ANCHORS=$ANCHORS,TEACHER_CKPT=$TEACHER_CKPT,EXPECTED_TEACHER_CKPT_SHA256=$EXPECTED_TEACHER_CKPT_SHA256,RESNET18_WEIGHTS=$RESNET18_WEIGHTS,EXPECTED_RESNET18_SHA256=$EXPECTED_RESNET18_SHA256,EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION,EXPECTED_GR00T_REVISION=$GR00T_REVISION,EXPECTED_FAST_SOURCE_TREE_SHA256=$EXPECTED_FAST_SOURCE_TREE_SHA256,EXPECTED_GR00T_SOURCE_TREE_SHA256=$EXPECTED_GR00T_SOURCE_TREE_SHA256,EXPECTED_IMAGE3F_RELEASE_ID=$EXPECTED_IMAGE3F_RELEASE_ID,EXPECTED_IMAGE3F_ADAPTER_SHA256=$EXPECTED_IMAGE3F_ADAPTER_SHA256,EXPECTED_IMAGE3F_ENGINE_SHA256=$EXPECTED_IMAGE3F_ENGINE_SHA256,CAMERA_CALIBRATION_ACCEPTANCE=$CAMERA_CALIBRATION_ACCEPTANCE,CAMERA_CALIBRATION_RUN_ID=$CAMERA_CALIBRATION_RUN_ID,EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256=$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256,FIXED_EVAL_SEEDS=$FIXED_EVAL_SEEDS,FIXED_EVAL_NUM_ENVS=$FIXED_EVAL_NUM_ENVS,EXPECTED_RENDERERS=$EXPECTED_RENDERERS,MIN_SUCCESS_DELTA=$MIN_SUCCESS_DELTA,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_ENTITY=$WANDB_ENTITY,WANDB_AUTHORIZATION_SCOPE=$WANDB_AUTHORIZATION_SCOPE,WANDB_ONLINE_VALIDATION=$WANDB_ONLINE_VALIDATION,WANDB_ONLINE_VALIDATION_SHA256=$WANDB_ONLINE_VALIDATION_SHA256,PROJ=$PROJ,EXP=$RUN_ID"
CMD=("$SLURM_BIN_DIR/sbatch" --parsable --account=llmservice_nemotron_ultra \
    --partition=batch_long --qos=normal --nodes=2 --gpus-per-node=4 \
    --cpus-per-gpu=32 --mem=800G --time=48:00:00 --signal=B:USR1@600 \
    --job-name=rr_i3f_fixed_eval --output="$EVAL_DIR/logs/packed_%j.out" \
    --error="$EVAL_DIR/logs/packed_%j.err" --export="$EXPORTS" \
    "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_policy_eval.sbatch")

cat > "$EVAL_DIR/submission_manifest.txt" <<EOF
status=planned
observation_mode=image3f_generated_rgb_3view
image3f_in_the_loop=true
production_observation_distribution=true
nodes=2
gpus=8
renderer_services=7
evaluator_gpus=1
checkpoint_execution=sequential_same_warm_fleet
checkpoint_tasks=4
seeds=$FIXED_EVAL_SEEDS
envs_per_seed=$FIXED_EVAL_NUM_ENVS
episodes_per_checkpoint=$((FIXED_EVAL_NUM_ENVS * 8))
partition=batch_long
qos=normal
time_limit=48:00:00
fast_roborender_revision=$FAST_REVISION
fast_roborender_source_tree_sha256=$EXPECTED_FAST_SOURCE_TREE_SHA256
gr00t_revision=$GR00T_REVISION
gr00t_source_tree_sha256=$EXPECTED_GR00T_SOURCE_TREE_SHA256
renderer_container_sha256=$EXPECTED_RENDERER_CONTAINER_SHA256
trainer_container_sha256=$EXPECTED_TRAINER_CONTAINER_SHA256
image3f_engine_sha256=$EXPECTED_IMAGE3F_ENGINE_SHA256
fixed_seed_manifest_sha256=$EXPECTED_EVAL_MANIFEST_SHA256
preflight_sha256=$EXPECTED_PREFLIGHT_SHA256
baseline_sha256=$BASELINE_SHA256
reference_config_sha256=$REFERENCE_CONFIG_SHA256
wandb_online_validation_sha256=$WANDB_ONLINE_VALIDATION_SHA256
camera_calibration_run_id=$CAMERA_CALIBRATION_RUN_ID
wandb_entity=$WANDB_ENTITY
wandb_authorization_scope=$WANDB_AUTHORIZATION_SCOPE
wandb_team_member=true
wandb_entity_is_team=true
wandb_personal=false
EOF
if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi
JOB_ID=$("${CMD[@]}")
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fatal "sbatch returned malformed job ID"
printf 'status=submitted\njob_id=%s\n' "$JOB_ID" >> "$EVAL_DIR/submission_manifest.txt"
echo "HSG GB200 Image3F production fixed-seed evaluation job: $JOB_ID"
