#!/bin/bash
# Prepare the secondary simulator-RGB policy-mechanics gate.  This deliberately
# does not claim to validate the Image3F-in-loop production observation stream.
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
ROOT=${ROOT:-$STAGE_ROOT/src/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:?TRAIN_RUN_DIR is required}
BASELINE_CKPT=${BASELINE_CKPT:-$TRAIN_RUN_DIR/initial_policy.pt}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$TRAIN_RUN_DIR/config.yaml}
TRAINER_CONTAINER=${TRAINER_CONTAINER:-$STAGE_ROOT/containers/roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh}
EXPECTED_TRAINER_CONTAINER_SHA256=${EXPECTED_TRAINER_CONTAINER_SHA256:-65ae3fec98e1bb9097bb852a28604ac43ba0bfeb1b169708e16ce1b28343b852}
RESNET18_WEIGHTS=${RESNET18_WEIGHTS:-$STAGE_ROOT/assets/torch/hub/checkpoints/resnet18-f37072fd.pth}
EXPECTED_RESNET18_SHA256=${EXPECTED_RESNET18_SHA256:-f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec}
FIXED_EVAL_SEEDS=${FIXED_EVAL_SEEDS:-1729,3253,4519,5987,7411,8863,10193,12203}
FIXED_EVAL_NUM_ENVS=${FIXED_EVAL_NUM_ENVS:-32}
MIN_SUCCESS_DELTA=${MIN_SUCCESS_DELTA:-0.02}
WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
WANDB_AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}
WANDB_ONLINE_VALIDATION=${WANDB_ONLINE_VALIDATION:-$STAGE_ROOT/runtime/validation/wandb_online_smoke_nvidia_acceptance.json}
PROJ=${PROJ:-roborender_dagger_hsg_simulator_rgb_fixed_seed_eval}
RUN_ID=${RUN_ID:-hsg_fixed_seed_eval_$(date -u +%Y%m%dT%H%M%SZ)}
EVAL_DIR=${EVAL_DIR:-$TRAIN_RUN_DIR/fixed_seed_eval/$RUN_ID}
DRY_RUN=${DRY_RUN:-0}
SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/local/apps/slurm/current/bin}

fatal() { echo "FATAL: $*" >&2; exit 2; }
case "$DRY_RUN" in 0|1) ;; *) fatal "DRY_RUN must be 0 or 1" ;; esac
[[ "$FIXED_EVAL_NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || fatal "FIXED_EVAL_NUM_ENVS must be positive"
[ "$FIXED_EVAL_NUM_ENVS" -eq 32 ] || fatal "HSG fixed evaluation is pinned to 32 envs per seed"
[ "$MIN_SUCCESS_DELTA" = 0.02 ] || fatal "HSG minimum success delta is pinned to 0.02"
[ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || fatal "HSG evaluation is pinned to ldu@nvidia.com"
[ "$WANDB_ENTITY" = nvidia ] || fatal "HSG evaluation is pinned to W&B team nvidia"
[ "$WANDB_AUTHORIZATION_SCOPE" = team ] || fatal "HSG W&B authorization scope is pinned to team"
for command in sbatch scontrol; do
    [ -x "$SLURM_BIN_DIR/$command" ] || fatal "missing HSG Slurm command $command"
done
for path in "$ROOT" "$GR00T_ROOT" "$TRAIN_RUN_DIR"; do [ -d "$path" ] || fatal "missing directory $path"; done
for path in "$BASELINE_CKPT" "$REFERENCE_CONFIG" "$TRAINER_CONTAINER" "$RESNET18_WEIGHTS"; do
    [ -s "$path" ] || fatal "missing required file $path"
done
[ "$(sha256sum "$TRAINER_CONTAINER" | awk '{print $1}')" = "$EXPECTED_TRAINER_CONTAINER_SHA256" ] || \
    fatal "SM100 trainer container hash mismatch"
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || \
    fatal "ResNet18 weight hash mismatch"

read_revision() {
    local marker=$1 value
    [ -s "$marker" ] || fatal "missing immutable revision marker $marker"
    value=$(tr -d '\r\n' < "$marker")
    [[ "$value" =~ ^[0-9a-f]{40}$ ]] || fatal "malformed revision marker $marker"
    printf '%s' "$value"
}
FAST_REVISION=$(read_revision "$ROOT/.source_revision")
GR00T_REVISION=$(read_revision "$GR00T_ROOT/.source_revision")
BASELINE_SHA256=$(sha256sum "$BASELINE_CKPT" | awk '{print $1}')
REFERENCE_CONFIG_SHA256=$(sha256sum "$REFERENCE_CONFIG" | awk '{print $1}')

NETRC="$STAGE_ROOT/runtime/home/.netrc"
[ -s "$NETRC" ] || fatal "W&B credential missing: $NETRC"
[ "$(stat -c %u "$NETRC")" -eq "$(id -u)" ] || fatal "W&B netrc owner mismatch"
case "$(stat -c %a "$NETRC")" in 400|600) ;; *) fatal "W&B netrc must be 0400 or 0600" ;; esac
[ -s "$WANDB_ONLINE_VALIDATION" ] || \
    fatal "authorized-team W&B scalar/image/video acceptance is missing"
WANDB_ENTITY=$(python3 "$ROOT/scripts/eval/wandb_team_contract.py" \
    --acceptance "$WANDB_ONLINE_VALIDATION" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
    --authorization-scope "$WANDB_AUTHORIZATION_SCOPE") || \
    fatal "authorized-team W&B acceptance is invalid"
WANDB_ONLINE_VALIDATION_SHA256=$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')

[ ! -e "$EVAL_DIR" ] || fatal "refusing to reuse existing evaluation directory $EVAL_DIR"
mkdir -p "$EVAL_DIR/logs"
EVAL_MANIFEST="$EVAL_DIR/manifest.json"
TASK_COUNT=$(python3 "$ROOT/scripts/eval/fixed_seed_gate.py" prepare \
    --train-run-dir "$TRAIN_RUN_DIR" --baseline-checkpoint "$BASELINE_CKPT" \
    --reference-config "$REFERENCE_CONFIG" --seeds "$FIXED_EVAL_SEEDS" \
    --num-envs "$FIXED_EVAL_NUM_ENVS" --expected-baseline-sha256 "$BASELINE_SHA256" \
    --expected-reference-config-sha256 "$REFERENCE_CONFIG_SHA256" --output "$EVAL_MANIFEST")
[ "$TASK_COUNT" = 4 ] || fatal "fixed-seed gate must produce four checkpoint tasks"
EXPECTED_EVAL_MANIFEST_SHA256=$(sha256sum "$EVAL_MANIFEST" | awk '{print $1}')
PREFLIGHT="$EVAL_DIR/hsg_preflight.json"
python3 "$ROOT/scripts/eval/hsg_gb200_fixed_seed_eval_gate.py" preflight \
    --root "$ROOT" --gr00t-root "$GR00T_ROOT" --fast-revision "$FAST_REVISION" \
    --gr00t-revision "$GR00T_REVISION" --trainer-container "$TRAINER_CONTAINER" \
    --trainer-container-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --fixed-seed-manifest "$EVAL_MANIFEST" \
    --fixed-seed-manifest-sha256 "$EXPECTED_EVAL_MANIFEST_SHA256" \
    --train-run-dir "$TRAIN_RUN_DIR" --expected-seeds "$FIXED_EVAL_SEEDS" \
    --expected-num-envs "$FIXED_EVAL_NUM_ENVS" \
    --wandb-acceptance "$WANDB_ONLINE_VALIDATION" \
    --wandb-acceptance-sha256 "$WANDB_ONLINE_VALIDATION_SHA256" > "$PREFLIGHT"
EXPECTED_PREFLIGHT_SHA256=$(sha256sum "$PREFLIGHT" | awk '{print $1}')
EXPECTED_FAST_SOURCE_TREE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fast_roborender_source_tree_sha256"])' "$PREFLIGHT")
EXPECTED_GR00T_SOURCE_TREE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gr00t_source_tree_sha256"])' "$PREFLIGHT")

EXPORTS="PATH=$SLURM_BIN_DIR:/usr/local/bin:/usr/bin:/bin,SLURM_BIN_DIR=$SLURM_BIN_DIR,STAGE_ROOT=$STAGE_ROOT,ROOT=$ROOT,GR00T_ROOT=$GR00T_ROOT,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,EVAL_DIR=$EVAL_DIR,EVAL_MANIFEST=$EVAL_MANIFEST,EXPECTED_EVAL_MANIFEST_SHA256=$EXPECTED_EVAL_MANIFEST_SHA256,PREFLIGHT=$PREFLIGHT,EXPECTED_PREFLIGHT_SHA256=$EXPECTED_PREFLIGHT_SHA256,TRAINER_CONTAINER=$TRAINER_CONTAINER,EXPECTED_TRAINER_CONTAINER_SHA256=$EXPECTED_TRAINER_CONTAINER_SHA256,EXPECTED_FAST_ROBORENDER_REVISION=$FAST_REVISION,EXPECTED_GR00T_REVISION=$GR00T_REVISION,EXPECTED_FAST_SOURCE_TREE_SHA256=$EXPECTED_FAST_SOURCE_TREE_SHA256,EXPECTED_GR00T_SOURCE_TREE_SHA256=$EXPECTED_GR00T_SOURCE_TREE_SHA256,RESNET18_WEIGHTS=$RESNET18_WEIGHTS,EXPECTED_RESNET18_SHA256=$EXPECTED_RESNET18_SHA256,FIXED_EVAL_SEEDS=$FIXED_EVAL_SEEDS,FIXED_EVAL_NUM_ENVS=$FIXED_EVAL_NUM_ENVS,MIN_SUCCESS_DELTA=$MIN_SUCCESS_DELTA,WANDB_EXPECTED_EMAIL=$WANDB_EXPECTED_EMAIL,WANDB_ENTITY=$WANDB_ENTITY,WANDB_AUTHORIZATION_SCOPE=$WANDB_AUTHORIZATION_SCOPE,WANDB_ONLINE_VALIDATION=$WANDB_ONLINE_VALIDATION,WANDB_ONLINE_VALIDATION_SHA256=$WANDB_ONLINE_VALIDATION_SHA256,PROJ=$PROJ,EXP=$RUN_ID"
CMD=("$SLURM_BIN_DIR/sbatch" --parsable --account=llmservice_nemotron_ultra \
    --partition=batch_long --qos=normal --nodes=1 --gpus-per-node=4 \
    --cpus-per-gpu=32 --mem=800G --time=08:00:00 --signal=B:USR1@300 \
    --job-name=rr_hsg_fixed_eval --output="$EVAL_DIR/logs/packed_%j.out" \
    --error="$EVAL_DIR/logs/packed_%j.err" --export="$EXPORTS" \
    "$ROOT/scripts/eval/hsg_gb200_fixed_seed_policy_eval.sbatch")

cat > "$EVAL_DIR/submission_manifest.txt" <<EOF
status=planned
observation_mode=simulator_rgb
image3f_in_the_loop=false
production_observation_distribution_validated=false
readiness_scope=secondary_policy_mechanics_gate
nodes=1
gpus=4
partition=batch_long
time_limit=08:00:00
checkpoint_tasks=4
seeds=$FIXED_EVAL_SEEDS
envs_per_seed=$FIXED_EVAL_NUM_ENVS
episodes_per_checkpoint=$((FIXED_EVAL_NUM_ENVS * 8))
fast_roborender_revision=$FAST_REVISION
gr00t_revision=$GR00T_REVISION
trainer_container_sha256=$EXPECTED_TRAINER_CONTAINER_SHA256
fixed_seed_manifest_sha256=$EXPECTED_EVAL_MANIFEST_SHA256
preflight_sha256=$EXPECTED_PREFLIGHT_SHA256
baseline_sha256=$BASELINE_SHA256
reference_config_sha256=$REFERENCE_CONFIG_SHA256
wandb_online_validation_sha256=$WANDB_ONLINE_VALIDATION_SHA256
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
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fatal "sbatch returned malformed job id"
printf 'status=submitted\njob_id=%s\n' "$JOB_ID" >> "$EVAL_DIR/submission_manifest.txt"
echo "HSG GB200 fixed-seed policy evaluation job: $JOB_ID"
