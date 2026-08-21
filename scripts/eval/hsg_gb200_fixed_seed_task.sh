#!/bin/bash
# Evaluate one of baseline/early/mid/final on one GB200 over all seed blocks.
set -Eeuo pipefail

required=(STAGE_ROOT ROOT GR00T_ROOT EVAL_MANIFEST EVAL_DIR EXPECTED_EVAL_MANIFEST_SHA256
          EXPECTED_FAST_ROBORENDER_REVISION EXPECTED_GR00T_REVISION
          RESNET18_WEIGHTS EXPECTED_RESNET18_SHA256)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || { echo "FATAL: missing $name" >&2; exit 2; }
done
TASK_INDEX=${SLURM_PROCID:?HSG packed evaluation requires SLURM_PROCID}
[[ "$TASK_INDEX" =~ ^[0-3]$ ]] || { echo "FATAL: task index must be 0..3" >&2; exit 2; }
[ "$(sha256sum "$EVAL_MANIFEST" | awk '{print $1}')" = "$EXPECTED_EVAL_MANIFEST_SHA256" ] || {
    echo "FATAL: fixed-seed manifest changed" >&2; exit 2;
}
[ "$(tr -d '\r\n' < "$ROOT/.source_revision")" = "$EXPECTED_FAST_ROBORENDER_REVISION" ] || {
    echo "FATAL: fast_roborender source revision changed" >&2; exit 2;
}
[ "$(tr -d '\r\n' < "$GR00T_ROOT/.source_revision")" = "$EXPECTED_GR00T_REVISION" ] || {
    echo "FATAL: GR00T source revision changed" >&2; exit 2;
}
[ "$(sha256sum "$RESNET18_WEIGHTS" | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256" ] || {
    echo "FATAL: pinned ResNet18 weights changed" >&2; exit 2;
}

field() {
    python "$ROOT/scripts/eval/fixed_seed_gate.py" task-field \
        --manifest "$EVAL_MANIFEST" --index "$TASK_INDEX" --field "$1"
}
LABEL=$(field label)
CKPT=$(field checkpoint)
EXPECTED_CKPT_SHA=$(field checkpoint_sha256)
SEEDS_CSV=$(field seeds)
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
NUM_ENVS=$(python - "$EVAL_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["num_envs_per_seed"])
PY
)

export HOME="$EVAL_DIR/runtime/$LABEL/home"
export XDG_CACHE_HOME="$EVAL_DIR/runtime/$LABEL/xdg-cache"
export XDG_DATA_HOME="$EVAL_DIR/runtime/$LABEL/xdg-data"
export TMPDIR="$EVAL_DIR/runtime/$LABEL/tmp"
export TORCH_HOME
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export WANDB_MODE=disabled
export IMAGE_DAGGER_EVAL_RELABEL=0
export IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1
export IMAGE_DAGGER_EXPECT_PRIMARY_VIEW=head_color
export IMAGE_DAGGER_EXPECT_POLICY_VIEWS=head_color,hand_left,hand_right
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT=vertical
export IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE=324,192,3

STAGE="$EVAL_DIR/stage/$LABEL"
RESULTS="$EVAL_DIR/results/$LABEL"
mkdir -p "$STAGE" "$RESULTS" "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
STAGED_CKPT="$STAGE/$(basename "$CKPT")"
python "$ROOT/scripts/eval/fixed_seed_gate.py" sanitize-checkpoint \
    --checkpoint "$CKPT" --expected-sha256 "$EXPECTED_CKPT_SHA" \
    --output "$STAGED_CKPT" --provenance "$STAGE/checkpoint_provenance.json"
cp "$REFERENCE_CONFIG" "$STAGE/.config.yaml.tmp.$$"
mv "$STAGE/.config.yaml.tmp.$$" "$STAGE/config.yaml"
STAGED_CKPT_SHA=$(sha256sum "$STAGED_CKPT" | awk '{print $1}')
STAGED_CONFIG_SHA=$(sha256sum "$STAGE/config.yaml" | awk '{print $1}')
[ "$STAGED_CONFIG_SHA" = "$EXPECTED_CONFIG_SHA" ] || {
    echo "FATAL: staged reference config changed" >&2; exit 2;
}

cd /opt/GR00T-VisualSim2Real
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
    RESULT_DIR="$RESULTS/seed_$SEED"
    RESULT_JSON="$RESULT_DIR/result.json"
    if [ -s "$RESULT_JSON" ] && python "$ROOT/scripts/eval/fixed_seed_gate.py" validate-result \
            --manifest "$EVAL_MANIFEST" --label "$LABEL" --seed "$SEED" \
            --result "$RESULT_JSON" \
            --evaluated-checkpoint-sha256 "$STAGED_CKPT_SHA" \
            --evaluated-config-sha256 "$STAGED_CONFIG_SHA"; then
        echo "SKIP label=$LABEL seed=$SEED validated_result=true"
        continue
    fi
    mkdir -p "$RESULT_DIR"
    METRICS_PENDING="$RESULT_DIR/.metrics_eval.pending.$$"
    python "$ROOT/scripts/eval/eval_agent_metrics_adapter.py" \
        --evaluator "$GR00T_ROOT/gr00t/rl/eval_agent_trl.py" \
        --metrics-output "$METRICS_PENDING" -- \
        +checkpoint="$STAGED_CKPT" +num_envs="$NUM_ENVS" +headless=true \
        +seed="$SEED" +eval_output_dir="$RESULT_DIR" \
        +algo.config.eval.eval_num_envs_episodes=true \
        algo.config.eval.num_eval_episodes="$NUM_ENVS" \
        algo.config.eval.save_videos=false algo.config.eval.save_trajectories=false \
        +simulator.config.render_results=false +use_wandb=false \
        hydra.run.dir="$RESULT_DIR/hydra" +output_dir="$RESULT_DIR/output"
    [ -s "$METRICS_PENDING" ] || { echo "FATAL: evaluator did not produce metrics" >&2; exit 2; }
    mv "$METRICS_PENDING" "$RESULT_DIR/metrics_eval.json"
    python "$ROOT/scripts/eval/fixed_seed_gate.py" record \
        --manifest "$EVAL_MANIFEST" --label "$LABEL" --seed "$SEED" \
        --metrics "$RESULT_DIR/metrics_eval.json" --output "$RESULT_JSON" \
        --evaluated-checkpoint-sha256 "$STAGED_CKPT_SHA" \
        --evaluated-config-sha256 "$STAGED_CONFIG_SHA"
done
touch "$RESULTS/TASK_DONE"
echo "HSG_FIXED_SEED_TASK_DONE label=$LABEL"
