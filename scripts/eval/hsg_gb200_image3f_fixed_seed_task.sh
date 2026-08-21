#!/bin/bash
# Run baseline/early/mid/final sequentially on one evaluator GPU and one warm fleet.
set -Eeuo pipefail

required=(STAGE_ROOT ROOT GR00T_ROOT EVAL_MANIFEST EVAL_DIR
          EXPECTED_EVAL_MANIFEST_SHA256 EXPECTED_FAST_ROBORENDER_REVISION
          EXPECTED_GR00T_REVISION RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256
          FLEET_DIR EP_LIST ENDPOINT_MANIFEST ANCHORS EXPECTED_RENDERERS EXPECTED_IMAGE3F_RELEASE_ID
          EXPECTED_IMAGE3F_ADAPTER_SHA256 EXPECTED_IMAGE3F_ENGINE_SHA256 PLACEMENT_PLAN
          FIXED_EVAL_NUM_ENVS EVAL_RUN_ID)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "FATAL: missing $name" >&2; exit 2; }
done

fatal() { echo "FATAL: $*" >&2; exit 2; }
[ "$EXPECTED_RENDERERS" -eq 7 ] || fatal "production evaluator requires seven renderers"
[ "$FIXED_EVAL_NUM_ENVS" -eq 32 ] || fatal "production evaluator requires 32 environments"
[ "$(sha256sum "$EVAL_MANIFEST" | awk '{print $1}')" = "$EXPECTED_EVAL_MANIFEST_SHA256" ] || \
    fatal "fixed-seed manifest changed"
[ "$(tr -d '\r\n' < "$ROOT/.source_revision")" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || \
    fatal "fast_roborender source revision changed"
[ "$(tr -d '\r\n' < "$GR00T_ROOT/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || \
    fatal "GR00T source revision changed"
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || \
    fatal "pinned ResNet18 weights changed"
[ -s "$ENDPOINT_MANIFEST" ] || fatal "selected endpoint manifest is missing"
[ "$(grep -cve '^[[:space:]]*$' "$ENDPOINT_MANIFEST")" -eq 7 ] || \
    fatal "selected endpoint manifest must contain exactly seven endpoints"

export TORCH_HOME="$STAGE_ROOT/assets/torch"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export WANDB_MODE=disabled
export IMAGE_DAGGER_EVAL_RELABEL=1
export IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1
export IMAGE_DAGGER_EXPECT_PRIMARY_VIEW=head_color
export IMAGE_DAGGER_EXPECT_POLICY_VIEWS=head_color,hand_left,hand_right
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT=vertical
export IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE=324,192,3
export IMAGE3F_ENDPOINT_DIR="$FLEET_DIR/endpoints"
export IMAGE3F_ENDPOINT_MANIFEST="$ENDPOINT_MANIFEST"
export IMAGE3F_LOCAL_ENVS=32
export IMAGE3F_REFERENCE_LAG=0
export IMAGE3F_EXPECT_OUTPUT_VIEWS=head_color,hand_left,hand_right
export IMAGE3F_EXPECT_OUTPUT_SHAPE=240,416,3
export IMAGE3F_REQUEST_TIMEOUT_S=120

probe() {
    local report=$1
    python "$ROOT/scripts/scaling/image3f_probe.py" \
        --endpoint-dir "$FLEET_DIR/endpoints" --expected "$EXPECTED_RENDERERS" \
        --wait-timeout-s 120 --connect-timeout-s 10 --poll-s 5 \
        --expected-release-id "$EXPECTED_IMAGE3F_RELEASE_ID" \
        --expected-adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256" \
        --expected-engine-sha256 "$EXPECTED_IMAGE3F_ENGINE_SHA256" \
        --expected-inference-steps 5 --expected-teacache-threshold 0.3 \
        --expected-cfg-scale 1.0 --expected-cuda-graph disabled \
        --expected-output-views head_color,hand_left,hand_right \
        --expected-output-layout vertical --expected-output-tile-shape 240,416,3 \
        --expected-output-stacked-shape 720,416,3 --require-tensorrt-engine \
        --expected-host-counts "$PLACEMENT_PLAN" \
        --write-report "$report" --json >/dev/null
}

field() {
    local index=$1 field_name=$2
    python "$ROOT/scripts/eval/fixed_seed_gate.py" task-field \
        --manifest "$EVAL_MANIFEST" --index "$index" --field "$field_name"
}

REFERENCE_CONFIG=$(python - "$EVAL_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["reference_config"])
PY
)
EXPECTED_CONFIG_SHA=$(python - "$EVAL_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["reference_config_sha256"])
PY
)

cd /opt/GR00T-VisualSim2Real
for TASK_INDEX in 0 1 2 3; do
    LABEL=$(field "$TASK_INDEX" label)
    CKPT=$(field "$TASK_INDEX" checkpoint)
    EXPECTED_CKPT_SHA=$(field "$TASK_INDEX" checkpoint_sha256)
    SEEDS_CSV=$(field "$TASK_INDEX" seeds)
    STAGE="$EVAL_DIR/stage/$LABEL"
    RESULTS="$EVAL_DIR/results/$LABEL"
    mkdir -p "$STAGE" "$RESULTS"
    STAGED_CKPT="$STAGE/$(basename "$CKPT")"
    python "$ROOT/scripts/eval/fixed_seed_gate.py" sanitize-checkpoint \
        --checkpoint "$CKPT" --expected-sha256 "$EXPECTED_CKPT_SHA" \
        --output "$STAGED_CKPT" --provenance "$STAGE/checkpoint_provenance.json"
    cp "$REFERENCE_CONFIG" "$STAGE/.config.yaml.tmp.$$"
    mv "$STAGE/.config.yaml.tmp.$$" "$STAGE/config.yaml"
    STAGED_CKPT_SHA=$(sha256sum "$STAGED_CKPT" | awk '{print $1}')
    STAGED_CONFIG_SHA=$(sha256sum "$STAGE/config.yaml" | awk '{print $1}')
    [ "$STAGED_CONFIG_SHA" = "$EXPECTED_CONFIG_SHA" ] || \
        fatal "staged reference config changed"

    IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
    for SEED in "${SEEDS[@]}"; do
        RESULT_DIR="$RESULTS/seed_$SEED"
        [ ! -e "$RESULT_DIR" ] || fatal "refusing to reuse result directory $RESULT_DIR"
        mkdir -p "$RESULT_DIR"
        export HOME="$EVAL_DIR/runtime/$LABEL/seed_$SEED/home"
        export XDG_CACHE_HOME="$EVAL_DIR/runtime/$LABEL/seed_$SEED/xdg-cache"
        export XDG_DATA_HOME="$EVAL_DIR/runtime/$LABEL/seed_$SEED/xdg-data"
        export TMPDIR="$EVAL_DIR/runtime/$LABEL/seed_$SEED/tmp"
        mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TMPDIR"
        export IMAGE3F_RUN_ID="${EVAL_RUN_ID}_${LABEL}_seed${SEED}"

        BEFORE="$RESULT_DIR/renderers_before.json"
        AFTER="$RESULT_DIR/renderers_after.json"
        probe "$BEFORE"
        METRICS_PENDING="$RESULT_DIR/.metrics_eval.pending.$$"
        python "$ROOT/scripts/eval/eval_agent_metrics_adapter.py" \
            --evaluator "$GR00T_ROOT/gr00t/rl/eval_agent_trl.py" \
            --metrics-output "$METRICS_PENDING" -- \
            +checkpoint="$STAGED_CKPT" +num_envs="$FIXED_EVAL_NUM_ENVS" \
            +headless=true +seed="$SEED" +eval_output_dir="$RESULT_DIR" \
            algo.config.eval.eval_num_envs_episodes=true \
            algo.config.eval.num_eval_episodes="$FIXED_EVAL_NUM_ENVS" \
            algo.config.eval.save_videos=false algo.config.eval.save_trajectories=false \
            algo.config.init_at_random_ep_len=false \
            algo.config.videogen.image3f_endpoint_files="$EP_LIST" \
            algo.config.videogen.anchor_pool_dir="$ANCHORS" \
            algo.config.videogen.anchor_index=0 \
            algo.config.videogen.raytraced_mix_prob=0.0 \
            'algo.config.videogen.view_names=[head_color,hand_left,hand_right]' \
            ++algo.config.videogen.reference_lag=0 \
            ++algo.config.videogen.primary_view=head_color \
            ++algo.config.videogen.seed="$SEED" \
            'simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]' \
            ++env.config.image_dagger_policy_view_count=3 \
            +simulator.config.render_results=false +use_wandb=false \
            hydra.run.dir="$RESULT_DIR/hydra" +output_dir="$RESULT_DIR/output"
        [ -s "$METRICS_PENDING" ] || fatal "evaluator did not produce metrics"
        mv "$METRICS_PENDING" "$RESULT_DIR/metrics_eval.json"
        probe "$AFTER"
        python "$ROOT/scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py" attest \
            --before "$BEFORE" --after "$AFTER" \
            --metrics "$RESULT_DIR/metrics_eval.json" \
            --endpoint-manifest "$ENDPOINT_MANIFEST" --label "$LABEL" \
            --seed "$SEED" --run-id "$IMAGE3F_RUN_ID" \
            --num-envs "$FIXED_EVAL_NUM_ENVS" --expected-renderers "$EXPECTED_RENDERERS" \
            --output "$RESULT_DIR/image3f_attestation.json" >/dev/null
        python "$ROOT/scripts/eval/fixed_seed_gate.py" record \
            --manifest "$EVAL_MANIFEST" --label "$LABEL" --seed "$SEED" \
            --metrics "$RESULT_DIR/metrics_eval.json" \
            --output "$RESULT_DIR/result.json" \
            --evaluated-checkpoint-sha256 "$STAGED_CKPT_SHA" \
            --evaluated-config-sha256 "$STAGED_CONFIG_SHA"
    done
    touch "$RESULTS/TASK_DONE"
done
echo "HSG_IMAGE3F_FIXED_SEED_TASKS_DONE roles=4 sequential=true"
