#!/bin/bash
# One array task evaluates one checkpoint over every fixed seed block.  Keeping
# each seed to exactly one episode per environment preserves matched reset RNGs.
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=rr_fixed_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --exclude=viscam10,viscam12
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/rr_fixed_eval_%A_%a.log
set -Eeuo pipefail

ROOT=${ROOT:-/svl/u/ravenh/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
EVAL_MANIFEST=${EVAL_MANIFEST:?EVAL_MANIFEST is required}
EVAL_DIR=${EVAL_DIR:?EVAL_DIR is required}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?run this script as a Slurm array}
PYTHON=${PYTHON:-/svl/u/ravenh/miniconda3/envs/viral/bin/python}

[ -f "$EVAL_MANIFEST" ] || { echo "FATAL: missing manifest: $EVAL_MANIFEST"; exit 2; }
[ -x "$PYTHON" ] || { echo "FATAL: evaluation Python is not executable: $PYTHON"; exit 2; }

field() {
    "$PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" task-field \
        --manifest "$EVAL_MANIFEST" --index "$TASK_INDEX" --field "$1"
}

LABEL=$(field label)
CKPT=$(field checkpoint)
EXPECTED_CKPT_SHA=$(field checkpoint_sha256)
SEEDS_CSV=$(field seeds)
REFERENCE_CONFIG=$(
    "$PYTHON" - "$EVAL_MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["reference_config"])
PY
)
EXPECTED_CONFIG_SHA=$(
    "$PYTHON" - "$EVAL_MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["reference_config_sha256"])
PY
)
NUM_ENVS=$(
    "$PYTHON" - "$EVAL_MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["num_envs_per_seed"])
PY
)

[ -f "$CKPT" ] || { echo "FATAL: missing $LABEL checkpoint: $CKPT"; exit 2; }
[ -f "$REFERENCE_CONFIG" ] || { echo "FATAL: missing reference config: $REFERENCE_CONFIG"; exit 2; }
ACTUAL_CONFIG_SHA=$(sha256sum "$REFERENCE_CONFIG" | awk '{print $1}')
[ "$ACTUAL_CONFIG_SHA" = "$EXPECTED_CONFIG_SHA" ] || {
    echo "FATAL: reference config changed after manifest creation"
    exit 2
}
export HOME=${HOME_OVERRIDE:-/sailhome/ravenh}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/svl/u/ravenh/.cache_xdg}
export XDG_DATA_HOME=${XDG_DATA_HOME:-/svl/u/ravenh/.share_xdg}
export TMPDIR=${TMPDIR:-/svl/u/ravenh/tmp}
export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE=disabled

STAGE="$EVAL_DIR/stage/$LABEL"
RESULTS="$EVAL_DIR/results/$LABEL"
mkdir -p "$STAGE" "$RESULTS"
STAGED_CKPT="$STAGE/$(basename "$CKPT")"
"$PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" sanitize-checkpoint \
    --checkpoint "$CKPT" \
    --expected-sha256 "$EXPECTED_CKPT_SHA" \
    --output "$STAGED_CKPT" \
    --provenance "$STAGE/checkpoint_provenance.json"
STAGED_CONFIG_TMP="$STAGE/.config.yaml.tmp.$$"
cp "$REFERENCE_CONFIG" "$STAGED_CONFIG_TMP"
mv "$STAGED_CONFIG_TMP" "$STAGE/config.yaml"
STAGED_CKPT_SHA=$(sha256sum "$STAGED_CKPT" | awk '{print $1}')
STAGED_CONFIG_SHA=$(sha256sum "$STAGE/config.yaml" | awk '{print $1}')
[ "$STAGED_CONFIG_SHA" = "$EXPECTED_CONFIG_SHA" ] || {
    echo "FATAL: staged config bytes do not match the manifest"
    exit 2
}

source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
conda activate viral
cd "$GR00T_ROOT"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
    RESULT_DIR="$RESULTS/seed_$SEED"
    RESULT_JSON="$RESULT_DIR/result.json"
    if [ -s "$RESULT_JSON" ] && \
       "$PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" validate-result \
            --manifest "$EVAL_MANIFEST" \
            --label "$LABEL" \
            --seed "$SEED" \
            --result "$RESULT_JSON" \
            --evaluated-checkpoint-sha256 "$STAGED_CKPT_SHA" \
            --evaluated-config-sha256 "$STAGED_CONFIG_SHA"; then
        echo "SKIP $LABEL seed=$SEED: validated result already exists"
        continue
    fi
    mkdir -p "$RESULT_DIR"
    echo "EVAL $LABEL seed=$SEED envs=$NUM_ENVS checkpoint=$CKPT"
    METRICS="$RESULT_DIR/metrics_eval.json"
    METRICS_PENDING="$RESULT_DIR/.metrics_eval.pending.$$"
    "$PYTHON" "$ROOT/scripts/eval/eval_agent_metrics_adapter.py" \
        --evaluator "$GR00T_ROOT/gr00t/rl/eval_agent_trl.py" \
        --metrics-output "$METRICS_PENDING" \
        -- \
        +checkpoint="$STAGED_CKPT" \
        +num_envs="$NUM_ENVS" \
        +headless=true \
        +seed="$SEED" \
        +eval_output_dir="$RESULT_DIR" \
        +algo.config.eval.eval_num_envs_episodes=true \
        algo.config.eval.num_eval_episodes="$NUM_ENVS" \
        algo.config.eval.save_videos=false \
        algo.config.eval.save_trajectories=false \
        +simulator.config.render_results=false \
        +use_wandb=false \
        hydra.run.dir="$RESULT_DIR/hydra" \
        +output_dir="$RESULT_DIR/output"

    [ -s "$METRICS_PENDING" ] || {
        echo "FATAL: metrics adapter did not write $METRICS_PENDING"
        exit 2
    }
    mv "$METRICS_PENDING" "$METRICS"
    "$PYTHON" "$ROOT/scripts/eval/fixed_seed_gate.py" record \
        --manifest "$EVAL_MANIFEST" \
        --label "$LABEL" \
        --seed "$SEED" \
        --metrics "$METRICS" \
        --output "$RESULT_JSON" \
        --evaluated-checkpoint-sha256 "$STAGED_CKPT_SHA" \
        --evaluated-config-sha256 "$STAGED_CONFIG_SHA"
done

touch "$RESULTS/TASK_DONE"
echo "FIXED_SEED_TASK_DONE label=$LABEL"
