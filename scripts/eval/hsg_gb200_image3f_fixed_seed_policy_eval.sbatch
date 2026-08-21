#!/bin/bash
# Production Image3F-in-loop fixed-seed gate: 7 renderers + 1 sequential evaluator.
# Submit only through submit_hsg_gb200_image3f_fixed_seed_gate.sh.
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --partition=batch_long
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=800G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --job-name=rr_i3f_fixed_eval
set -Eeuo pipefail

required=(STAGE_ROOT ROOT GR00T_ROOT TRAIN_RUN_DIR EVAL_DIR EVAL_RUN_ID EVAL_MANIFEST
          EXPECTED_EVAL_MANIFEST_SHA256 PREFLIGHT EXPECTED_PREFLIGHT_SHA256
          RENDERER_CONTAINER EXPECTED_RENDERER_CONTAINER_SHA256 TRAINER_CONTAINER
          EXPECTED_TRAINER_CONTAINER_SHA256 IMAGE3F_ROOT ROBORENDER_MODEL_BASE
          IMAGE3F_GENERATED_ROOT ANCHORS TEACHER_CKPT EXPECTED_TEACHER_CKPT_SHA256
          RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256 EXPECTED_FAST_ROBORENDER_REVISION
          EXPECTED_GR00T_REVISION EXPECTED_FAST_SOURCE_TREE_SHA256
          EXPECTED_GR00T_SOURCE_TREE_SHA256 EXPECTED_IMAGE3F_RELEASE_ID
          EXPECTED_IMAGE3F_ADAPTER_SHA256 EXPECTED_IMAGE3F_ENGINE_SHA256 CAMERA_CALIBRATION_ACCEPTANCE
          EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256 MIN_SUCCESS_DELTA
          FIXED_EVAL_SEEDS FIXED_EVAL_NUM_ENVS EXPECTED_RENDERERS WANDB_EXPECTED_EMAIL
          WANDB_ENTITY WANDB_ONLINE_VALIDATION WANDB_ONLINE_VALIDATION_SHA256
          PROJ EXP SLURM_BIN_DIR)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "FATAL: missing sanitized launch variable $name" >&2; exit 2; }
done
export SLURM_EXPORT_ENV=ALL
fatal() { echo "FATAL: $*" >&2; exit 2; }

[ "${SLURM_NNODES:-0}" -eq 2 ] || fatal "Image3F evaluation requires exactly two nodes"
[ "${SLURM_JOB_PARTITION:-}" = batch_long ] || fatal "partition must remain batch_long"
ACTUAL_QOS=${SLURM_JOB_QOS:-}
[ -n "$ACTUAL_QOS" ] || ACTUAL_QOS=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* QOS=\([^ ]*\).*/\1/p')
[ "$ACTUAL_QOS" = normal ] || fatal "qos must remain normal"
ACTUAL_ACCOUNT=${SLURM_JOB_ACCOUNT:-}
[ -n "$ACTUAL_ACCOUNT" ] || ACTUAL_ACCOUNT=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* Account=\([^ ]*\).*/\1/p')
[ "$ACTUAL_ACCOUNT" = llmservice_nemotron_ultra ] || fatal "account changed"
[ "$EXPECTED_RENDERERS" -eq 7 ] || fatal "renderer count changed"
[ "$FIXED_EVAL_NUM_ENVS" -eq 32 ] || fatal "environment count changed"
[ "$RENDERER_CONTAINER" != "$TRAINER_CONTAINER" ] || fatal "renderer/trainer images must differ"
[ "$(sha256sum "$RENDERER_CONTAINER" | awk '{print $1}')" = "$EXPECTED_RENDERER_CONTAINER_SHA256" ] || \
    fatal "renderer container changed"
[ "$(sha256sum "$TRAINER_CONTAINER" | awk '{print $1}')" = "$EXPECTED_TRAINER_CONTAINER_SHA256" ] || \
    fatal "trainer container changed"
[ "$(sha256sum "$EVAL_MANIFEST" | awk '{print $1}')" = "$EXPECTED_EVAL_MANIFEST_SHA256" ] || \
    fatal "evaluation manifest changed"
[ "$(sha256sum "$PREFLIGHT" | awk '{print $1}')" = "$EXPECTED_PREFLIGHT_SHA256" ] || \
    fatal "preflight changed"
[ "$(sha256sum "$WANDB_ONLINE_VALIDATION" | awk '{print $1}')" = "$WANDB_ONLINE_VALIDATION_SHA256" ] || \
    fatal "personal W&B acceptance changed"
[ "$(tr -d '\r\n' < "$ROOT/.source_revision")" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || \
    fatal "fast_roborender revision changed"
[ "$(tr -d '\r\n' < "$GR00T_ROOT/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || \
    fatal "GR00T revision changed"

python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" verify \
    --acceptance "$CAMERA_CALIBRATION_ACCEPTANCE" \
    --expected-fast-roborender-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --expected-gr00t-revision "$EXPECTED_GR00T_REVISION" \
    --expected-acceptance-sha256 "$EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256" >/dev/null

CURRENT_PREFLIGHT="$EVAL_DIR/.preflight.revalidated.$SLURM_JOB_ID.json"
python3 "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py" preflight \
    --root "$ROOT" --gr00t-root "$GR00T_ROOT" \
    --fast-revision "$EXPECTED_FAST_ROBORENDER_REVISION" \
    --gr00t-revision "$EXPECTED_GR00T_REVISION" \
    --renderer-container "$RENDERER_CONTAINER" \
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
    --resnet18 "$RESNET18_WEIGHTS" --resnet18-sha256 "$EXPECTED_RESNET18_SHA256" \
    --wandb-acceptance "$WANDB_ONLINE_VALIDATION" \
    --wandb-acceptance-sha256 "$WANDB_ONLINE_VALIDATION_SHA256" \
    --fixed-seed-manifest "$EVAL_MANIFEST" \
    --fixed-seed-manifest-sha256 "$EXPECTED_EVAL_MANIFEST_SHA256" \
    --train-run-dir "$TRAIN_RUN_DIR" --expected-seeds "$FIXED_EVAL_SEEDS" \
    --expected-num-envs "$FIXED_EVAL_NUM_ENVS" > "$CURRENT_PREFLIGHT"
cmp -s "$PREFLIGHT" "$CURRENT_PREFLIGHT" || fatal "preflight content changed after submission"
grep -F '"fast_roborender_source_tree_sha256": '"\"$EXPECTED_FAST_SOURCE_TREE_SHA256\"" "$PREFLIGHT" >/dev/null || \
    fatal "fast_roborender source tree changed"
grep -F '"gr00t_source_tree_sha256": '"\"$EXPECTED_GR00T_SOURCE_TREE_SHA256\"" "$PREFLIGHT" >/dev/null || \
    fatal "GR00T source tree changed"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#NODES[@]}" -eq 2 ] || fatal "node allocation changed"
MASTER_NODE=${NODES[0]}
EVAL_NODE=${NODES[1]}

WANDB_PREFLIGHT_OUTPUT=$(srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" \
    --gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE=online \
    /usr/local/bin/roborender-trainer-sm100 python \
    "$ROOT/scripts/scaling/wandb_preflight.py" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" --format shell)
[ "$(printf '%s\n' "$WANDB_PREFLIGHT_OUTPUT" | grep -c "^WANDB_ENTITY=$WANDB_ENTITY$")" -eq 1 ] || \
    fatal "containerized personal W&B preflight failed"

RUN_DONE="$EVAL_DIR/RUN_DONE"
RUN_FAILED="$EVAL_DIR/RUN_FAILED"
RENDER_STEP_PID=""
RENDER_SHUTDOWN_TIMEOUT_S=180
shutdown_renderer() {
    [ -n "$RENDER_STEP_PID" ] || return 0
    local pid=$RENDER_STEP_PID deadline
    kill -TERM "$pid" 2>/dev/null || true
    deadline=$((SECONDS + RENDER_SHUTDOWN_TIMEOUT_S))
    while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: renderer Slurm step did not stop within ${RENDER_SHUTDOWN_TIMEOUT_S}s" >&2
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        RENDER_STEP_PID=""
        return 72
    fi
    wait "$pid" 2>/dev/null || true
    RENDER_STEP_PID=""
    return 0
}
cleanup() {
    local rc=$?
    local renderer_rc=0
    trap - EXIT TERM INT USR1
    if shutdown_renderer; then
        renderer_rc=0
    else
        renderer_rc=$?
    fi
    if [ "$rc" -eq 0 ] && [ "$renderer_rc" -ne 0 ]; then
        rc=$renderer_rc
    fi
    if [ "$rc" -eq 0 ]; then : > "$RUN_DONE"; else printf '%s\n' "$rc" > "$RUN_FAILED"; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 124' USR1

mkdir -p "$EVAL_DIR/logs" "$EVAL_DIR/results" "$EVAL_DIR/stage" \
         "$FLEET_DIR/endpoints" "$FLEET_DIR/queues" "$FLEET_DIR/logs" \
         "$STAGE_ROOT/runtime/home" "$STAGE_ROOT/runtime/cache" "$STAGE_ROOT/runtime/tmp"
export HOME="$STAGE_ROOT/runtime/home"
export IMAGE3F_RUN_ID="$EVAL_RUN_ID"
export IMAGE3F_MAX_STREAMS=16
export STARTUP_STAGGER_SCOPE=local
export STARTUP_STAGGER_S=2
export PYTHONDONTWRITEBYTECODE=1

PLACEMENT_PLAN="$FLEET_DIR/renderer_placement_plan.json"
python3 - "$PLACEMENT_PLAN" "${NODES[0]}" "${NODES[1]}" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema": "fast_roborender.renderer_placement_plan.v1",
    "host_counts": {sys.argv[2]: 4, sys.argv[3]: 3},
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
export PLACEMENT_PLAN

srun --nodes=2 --ntasks=7 --ntasks-per-node=4 --distribution=cyclic \
    --gpus=7 --gpus-per-task=1 --cpus-per-task=8 --mem-per-gpu=64G \
    --exact --exclusive --gpu-bind=closest --cpu-bind=cores --kill-on-bad-exit=1 \
    --container-image="$RENDERER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$ROBORENDER_MODEL_BASE:/models:ro,$IMAGE3F_GENERATED_ROOT:/opt/roborender/assets/generated:ro" \
    --container-workdir=/opt/roborender \
    --output="$FLEET_DIR/logs/renderer_%t.out" --error="$FLEET_DIR/logs/renderer_%t.err" \
    bash -lc '
        set -Eeuo pipefail
        export ROBORENDER_CONTAINERIZED=1
        export ROBORENDER_STAGE_ROOT="$STAGE_ROOT"
        export SLURM_ARRAY_TASK_ID=$SLURM_PROCID
        export SLURM_ARRAY_JOB_ID=$SLURM_JOB_ID
        export IMAGE3F_ADVERTISE_HOST=${SLURMD_NODENAME:-$(hostname -f)}
        export IMAGE3F_ROOT=/opt/roborender
        export IMAGE3F_ENV=/opt/roborender-venv
        export IMAGE3F_PYTHON=/opt/roborender-venv/bin/python
        export ROBORENDER_MODEL_BASE=/models
        export DIFFSYNTH_MODEL_BASE_PATH=/models
        export DIFFSYNTH_SKIP_DOWNLOAD=true
        exec bash "$ROOT/scripts/server/image3f_server_array.sbatch"
    ' &
RENDER_STEP_PID=$!

READY_DEADLINE=$((SECONDS + 1800))
while :; do
    READY_COUNT=$(find "$FLEET_DIR/endpoints" -maxdepth 1 -type f -name 'renderer_*.txt' -printf . | wc -c)
    [ "$READY_COUNT" -eq 7 ] && break
    kill -0 "$RENDER_STEP_PID" 2>/dev/null || fatal "renderer step exited before readiness"
    [ "$SECONDS" -lt "$READY_DEADLINE" ] || fatal "timed out waiting for seven renderers"
    sleep 5
done

python3 "$ROOT/scripts/scaling/image3f_probe.py" \
    --endpoint-dir "$FLEET_DIR/endpoints" --expected 7 --wait-timeout-s 120 \
    --expected-release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
    --expected-adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    --expected-engine-sha256 "$EXPECTED_IMAGE3F_ENGINE_SHA256" \
    --expected-inference-steps 5 --expected-teacache-threshold 0.3 \
    --expected-cfg-scale 1.0 --expected-cuda-graph disabled \
    --expected-output-views head_color,hand_left,hand_right \
    --expected-output-layout vertical --expected-output-tile-shape 240,416,3 \
    --expected-output-stacked-shape 720,416,3 --require-tensorrt-engine \
    --write-manifest "$FLEET_DIR/selected_endpoints.txt" \
    --expected-host-counts "$PLACEMENT_PLAN" \
    --write-placement-report "$FLEET_DIR/renderer_placement_attestation.json" \
    --write-report "$FLEET_DIR/initial_renderer_capabilities.json" --hydra-list \
    > "$FLEET_DIR/endpoint_hydra_list.txt"
ENDPOINT_MANIFEST="$FLEET_DIR/selected_endpoints.txt"
EP_LIST=$(tr -d '\r\n' < "$FLEET_DIR/endpoint_hydra_list.txt")
[[ "$EP_LIST" == \[*\] ]] || fatal "Image3F Hydra endpoint list is malformed"
export EP_LIST ENDPOINT_MANIFEST
export FLEET_DIR EVAL_RUN_ID EVAL_DIR EVAL_MANIFEST EXPECTED_EVAL_MANIFEST_SHA256
export ROOT GR00T_ROOT STAGE_ROOT ANCHORS EXPECTED_RENDERERS FIXED_EVAL_NUM_ENVS
export EXPECTED_FAST_ROBORENDER_REVISION EXPECTED_GR00T_REVISION
export EXPECTED_IMAGE3F_RELEASE_ID EXPECTED_IMAGE3F_ADAPTER_SHA256 EXPECTED_IMAGE3F_ENGINE_SHA256
export RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256

srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --gpus=1 --gpus-per-task=1 \
    --cpus-per-task=32 --mem=500G --exact --exclusive --gpu-bind=closest \
    --cpu-bind=cores --kill-on-bad-exit=1 \
    --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    --output="$EVAL_DIR/logs/evaluator.out" --error="$EVAL_DIR/logs/evaluator.err" \
    /usr/local/bin/roborender-trainer-sm100 bash \
    "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_task.sh"

python3 "$ROOT/scripts/scaling/image3f_probe.py" \
    --endpoint-dir "$FLEET_DIR/endpoints" --expected 7 --wait-timeout-s 120 \
    --expected-release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
    --expected-adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
    --expected-engine-sha256 "$EXPECTED_IMAGE3F_ENGINE_SHA256" \
    --expected-inference-steps 5 --expected-teacache-threshold 0.3 \
    --expected-cfg-scale 1.0 --expected-cuda-graph disabled \
    --expected-output-views head_color,hand_left,hand_right \
    --expected-output-layout vertical --expected-output-tile-shape 240,416,3 \
    --expected-output-stacked-shape 720,416,3 --require-tensorrt-engine \
    --expected-host-counts "$PLACEMENT_PLAN" \
    --write-report "$FLEET_DIR/final_renderer_capabilities.json" --json >/dev/null
shutdown_renderer || fatal "renderer shutdown exceeded its bounded cleanup contract"

SUMMARY="$EVAL_DIR/fixed_seed_improvement_report.json"
set +e
python3 "$ROOT/scripts/eval/fixed_seed_gate.py" summarize \
    --manifest "$EVAL_MANIFEST" --results-dir "$EVAL_DIR/results" \
    --min-success-delta "$MIN_SUCCESS_DELTA" --output "$SUMMARY"
GATE_RC=$?
set -e
[ "$GATE_RC" -eq 0 ] || { echo "POLICY_IMPROVEMENT_GATE_FAIL rc=$GATE_RC" >&2; exit "$GATE_RC"; }

srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --gpus=0 --cpus-per-task=2 \
    --mem=4G --exact --exclusive --container-image="$TRAINER_CONTAINER" \
    --container-mounts="$STAGE_ROOT:$STAGE_ROOT,$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro" \
    --container-workdir=/opt/GR00T-VisualSim2Real \
    env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE=online \
    /usr/local/bin/roborender-trainer-sm100 python \
    "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_wandb.py" \
    --expected-email "$WANDB_EXPECTED_EMAIL" --entity "$WANDB_ENTITY" \
    --project "$PROJ" --run-id "i3ffixedeval${SLURM_JOB_ID}" \
    --run-name "${EXP}-image3f-fixed-seed-gate" --manifest "$EVAL_MANIFEST" \
    --summary "$SUMMARY" --results-dir "$EVAL_DIR/results" \
    --output "$EVAL_DIR/wandb_upload.json"

python3 "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py" postrun \
    --preflight "$PREFLIGHT" --preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" \
    --fixed-seed-manifest "$EVAL_MANIFEST" \
    --fixed-seed-manifest-sha256 "$EXPECTED_EVAL_MANIFEST_SHA256" \
    --results-dir "$EVAL_DIR/results" --summary "$SUMMARY" \
    --min-success-delta "$MIN_SUCCESS_DELTA" \
    --wandb-upload "$EVAL_DIR/wandb_upload.json" \
    --expected-wandb-email "$WANDB_EXPECTED_EMAIL" \
    --expected-wandb-entity "$WANDB_ENTITY" \
    --output "$EVAL_DIR/hsg_image3f_fixed_seed_acceptance.json" \
    > "$EVAL_DIR/postrun_gate.json"
echo "HSG_IMAGE3F_PRODUCTION_POLICY_GATE_PASS acceptance=$EVAL_DIR/hsg_image3f_fixed_seed_acceptance.json"
