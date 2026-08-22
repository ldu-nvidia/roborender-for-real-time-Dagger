#!/bin/bash
# Secondary simulator-RGB policy-mechanics gate, packed onto one 4xGB200 node.
# This does not validate the Image3F-in-loop production observation distribution.
# Submit only through submit_hsg_gb200_fixed_seed_gate.sh.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch_long
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=800G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=rr_hsg_fixed_eval
set -Eeuo pipefail

required=(STAGE_ROOT ROOT GR00T_ROOT TRAIN_RUN_DIR EVAL_DIR EVAL_MANIFEST EXPECTED_EVAL_MANIFEST_SHA256
          PREFLIGHT EXPECTED_PREFLIGHT_SHA256 TRAINER_CONTAINER EXPECTED_TRAINER_CONTAINER_SHA256
          EXPECTED_FAST_ROBORENDER_REVISION EXPECTED_GR00T_REVISION
          EXPECTED_FAST_SOURCE_TREE_SHA256 EXPECTED_GR00T_SOURCE_TREE_SHA256
          RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256 MIN_SUCCESS_DELTA
          FIXED_EVAL_SEEDS FIXED_EVAL_NUM_ENVS WANDB_EXPECTED_EMAIL WANDB_ENTITY
          WANDB_AUTHORIZATION_SCOPE
          WANDB_ONLINE_VALIDATION WANDB_ONLINE_VALIDATION_SHA256 PROJ EXP SLURM_BIN_DIR)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "FATAL: missing sanitized launch variable $name" >&2; exit 2; }
done
export SLURM_EXPORT_ENV=ALL
fatal() { echo "FATAL: $*" >&2; exit 2; }
[ "${SLURM_NNODES:-0}" -eq 1 ] || fatal "fixed evaluation requires one node"
[ "${SLURM_JOB_PARTITION:-}" = batch_long ] || fatal "fixed evaluation requires partition=batch_long"
ACTUAL_QOS=${SLURM_JOB_QOS:-}
[ -n "$ACTUAL_QOS" ] || ACTUAL_QOS=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* QOS=\([^ ]*\).*/\1/p')
[ "$ACTUAL_QOS" = normal ] || fatal "fixed evaluation requires qos=normal"
ACTUAL_ACCOUNT=${SLURM_JOB_ACCOUNT:-}
[ -n "$ACTUAL_ACCOUNT" ] || ACTUAL_ACCOUNT=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* Account=\([^ ]*\).*/\1/p')
[ "$ACTUAL_ACCOUNT" = llmservice_nemotron_ultra ] || fatal "fixed evaluation account changed"
[ "$WANDB_EXPECTED_EMAIL" = ldu@nvidia.com ] || fatal "W&B account binding changed"
[ "$WANDB_ENTITY" = nvidia ] || fatal "W&B team entity binding changed"
[ "$WANDB_AUTHORIZATION_SCOPE" = team ] || fatal "W&B authorization scope changed"
[ "$(sha256sum "$TRAINER_CONTAINER" | awk '{print $1}')" = "$EXPECTED_TRAINER_CONTAINER_SHA256" ] || \
    fatal "SM100 trainer container changed"
[ "$(sha256sum "$EVAL_MANIFEST" | awk '{print $1}')" = "$EXPECTED_EVAL_MANIFEST_SHA256" ] || \
    fatal "fixed-seed manifest changed"
[ "$(sha256sum "$PREFLIGHT" | awk '{print $1}')" = "$EXPECTED_PREFLIGHT_SHA256" ] || \
    fatal "fixed-seed preflight changed"
[ "$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')" = \
  "$WANDB_ONLINE_VALIDATION_SHA256" ] || fatal "authorized-team W&B acceptance changed"
VALIDATED_WANDB_ENTITY=$(python3 "$ROOT/scripts/eval/wandb_team_contract.py" \
    --acceptance "$WANDB_ONLINE_VALIDATION" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
    --authorization-scope "$WANDB_AUTHORIZATION_SCOPE") || \
    fatal "authorized-team W&B acceptance is invalid"
[ "$VALIDATED_WANDB_ENTITY" = "$WANDB_ENTITY" ] || fatal "W&B acceptance entity changed"
[ "$(tr -d '\r\n' < "$ROOT/.source_revision")" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || \
    fatal "fast_roborender revision changed"
[ "$(tr -d '\r\n' < "$GR00T_ROOT/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || \
    fatal "GR00T revision changed"

CURRENT_PREFLIGHT="$EVAL_DIR/.preflight.revalidated.$SLURM_JOB_ID.json"
python3 "$ROOT/scripts/eval/hsg_gb200_fixed_seed_eval_gate.py" preflight \
    --root "$ROOT" --gr00t-root "$GR00T_ROOT" \
    --fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --gr00t-revision "$EXPECTED_GR00T_REVISION" \
    --trainer-container "$TRAINER_CONTAINER" \
    --trainer-container-sha256 "$EXPECTED_TRAINER_CONTAINER_SHA256" \
    --fixed-seed-manifest "$EVAL_MANIFEST" \
    --fixed-seed-manifest-sha256 "$EXPECTED_EVAL_MANIFEST_SHA256" \
    --train-run-dir "$TRAIN_RUN_DIR" --expected-seeds "$FIXED_EVAL_SEEDS" \
    --expected-num-envs "$FIXED_EVAL_NUM_ENVS" \
    --wandb-acceptance "$WANDB_ONLINE_VALIDATION" \
    --wandb-acceptance-sha256 "$WANDB_ONLINE_VALIDATION_SHA256" > "$CURRENT_PREFLIGHT"
cmp -s "$PREFLIGHT" "$CURRENT_PREFLIGHT" || fatal "HSG preflight changed after submission"
grep -F '"fast_roborender_source_tree_sha256": '"\"$EXPECTED_FAST_SOURCE_TREE_SHA256\"" \
    "$CURRENT_PREFLIGHT" >/dev/null || fatal "fast_roborender source-tree digest changed"
grep -F '"gr00t_source_tree_sha256": '"\"$EXPECTED_GR00T_SOURCE_TREE_SHA256\"" \
    "$CURRENT_PREFLIGHT" >/dev/null || fatal "GR00T source-tree digest changed"
rm -f "$CURRENT_PREFLIGHT"

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
WANDB_PREFLIGHT_OUTPUT=$(srun --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" \
    --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE=online \
    /usr/local/bin/roborender-trainer-sm100 python \
    "$ROOT/scripts/scaling/wandb_preflight.py" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
    --authorization-scope "$WANDB_AUTHORIZATION_SCOPE" --format shell)
[ "$(printf '%s\n' "$WANDB_PREFLIGHT_OUTPUT" | grep -c "^WANDB_ENTITY=$WANDB_ENTITY$")" -eq 1 ] || \
    fatal "containerized authorized-team W&B preflight failed"

RUN_DONE="$EVAL_DIR/RUN_DONE"
RUN_FAILED="$EVAL_DIR/RUN_FAILED"
trap 'rc=$?; trap - EXIT TERM INT USR1; if [ "$rc" -eq 0 ]; then : > "$RUN_DONE"; else printf "%s\n" "$rc" > "$RUN_FAILED"; fi; exit "$rc"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 124' USR1
mkdir -p "$EVAL_DIR/logs" "$EVAL_DIR/results" "$EVAL_DIR/stage" \
         "$STAGE_ROOT/runtime/home" "$STAGE_ROOT/runtime/cache" "$STAGE_ROOT/runtime/tmp"
export EVAL_MANIFEST EVAL_DIR EXPECTED_EVAL_MANIFEST_SHA256 ROOT GR00T_ROOT STAGE_ROOT
export EXPECTED_FAST_ROBORENDER_REVISION EXPECTED_GR00T_REVISION
export RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256 TORCH_HOME="$STAGE_ROOT/assets/torch"

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus=4 --gpus-per-task=1 \
    --cpus-per-task=24 --mem=720G --exact --exclusive --gpu-bind=closest \
    --cpu-bind=cores --kill-on-bad-exit=1 \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    --output="$EVAL_DIR/logs/eval_%t.out" --error="$EVAL_DIR/logs/eval_%t.err" \
    /usr/local/bin/roborender-trainer-sm100 bash \
    "$ROOT/scripts/eval/hsg_gb200_fixed_seed_task.sh"

SUMMARY="$EVAL_DIR/fixed_seed_improvement_report.json"
set +e
python3 "$ROOT/scripts/eval/fixed_seed_gate.py" summarize \
    --manifest "$EVAL_MANIFEST" --results-dir "$EVAL_DIR/results" \
    --min-success-delta "$MIN_SUCCESS_DELTA" --output "$SUMMARY"
GATE_RC=$?
set -e
[ "$GATE_RC" -eq 0 ] || { echo "POLICY_IMPROVEMENT_GATE_FAIL rc=$GATE_RC report=$SUMMARY" >&2; exit "$GATE_RC"; }

srun --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" \
    --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE=online \
    /usr/local/bin/roborender-trainer-sm100 python \
    "$ROOT/scripts/eval/hsg_gb200_fixed_seed_wandb.py" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
    --project "$PROJ" --run-id "hsgfixedeval${SLURM_JOB_ID}" \
    --run-name "${EXP}-fixed-seed-gate" --manifest "$EVAL_MANIFEST" \
    --summary "$SUMMARY" --wandb-acceptance "$WANDB_ONLINE_VALIDATION" \
    --wandb-acceptance-sha256 "$WANDB_ONLINE_VALIDATION_SHA256" \
    --output "$EVAL_DIR/wandb_upload.json"

WANDB_UPLOAD="$EVAL_DIR/wandb_upload.json"

ACCEPTANCE="$EVAL_DIR/hsg_fixed_seed_acceptance.json"
python3 "$ROOT/scripts/eval/hsg_gb200_fixed_seed_eval_gate.py" postrun \
    --preflight "$PREFLIGHT" --preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" \
    --fixed-seed-manifest "$EVAL_MANIFEST" \
    --fixed-seed-manifest-sha256 "$EXPECTED_EVAL_MANIFEST_SHA256" \
    --results-dir "$EVAL_DIR/results" --summary "$SUMMARY" \
    --min-success-delta "$MIN_SUCCESS_DELTA" --wandb-upload "$WANDB_UPLOAD" \
    --expected-wandb-email "$WANDB_EXPECTED_EMAIL" \
    --expected-wandb-entity "$WANDB_ENTITY" --output "$ACCEPTANCE" \
    > "$EVAL_DIR/postrun_gate.json"
echo "HSG_SIMULATOR_RGB_POLICY_MECHANICS_PASS acceptance=$ACCEPTANCE image3f_in_the_loop=false"
