#!/usr/bin/env bash
# Raven headline evaluation: saved BC environment config, ray-traced RGB, 300 episodes.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T=$ROOT/pipeline/GR00T-VisualSim2Real
QROOT=$ROOT/runtime/isaac_actor/qualified_h200_v1
PYTHON=${GR00T_PYTHON:-$QROOT/runtime/isaacsim-5.1/bin/python}
CHECKPOINT=${1:?usage: $0 CHECKPOINT OFFLINE_BC_BASELINE_CONFIG [NUM_ENVS] [RESULT_DIR]}
BASELINE_CONFIG=${2:?usage: $0 CHECKPOINT OFFLINE_BC_BASELINE_CONFIG [NUM_ENVS] [RESULT_DIR]}
NUM_ENVS=${3:-4}
RESULT_DIR=${4:-$ROOT/runtime/outputs/evaluation/raven_raytraced_300}

[[ $NUM_ENVS =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be positive" >&2; exit 2; }
[[ -f $CHECKPOINT ]] || { echo "checkpoint missing: $CHECKPOINT" >&2; exit 2; }
[[ -f $BASELINE_CONFIG ]] || { echo "baseline config missing: $BASELINE_CONFIG" >&2; exit 2; }
[[ -x $PYTHON ]] || { echo "GR00T Python missing: $PYTHON" >&2; exit 2; }
[[ $RESULT_DIR == /* ]] || { echo "RESULT_DIR must be absolute" >&2; exit 2; }

/usr/bin/python3 "$ROOT/pipeline/dagger/validate_raven_image3f_contract.py"
/usr/bin/python3 "$ROOT/pipeline/evaluation/validate_raven_baseline_config.py" \
    "$BASELINE_CONFIG"
mkdir -p "$ROOT/runtime/evaluation" "$RESULT_DIR"
STAGE=$(mktemp -d "$ROOT/runtime/evaluation/raven-eval-stage.XXXXXX")
cleanup() {
    case "$STAGE" in "$ROOT/runtime/evaluation/raven-eval-stage."*) rm -rf -- "$STAGE" ;; esac
}
trap cleanup EXIT
cp -- "$CHECKPOINT" "$STAGE/last.pt"
cp -- "$BASELINE_CONFIG" "$STAGE/config.yaml"

source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
unset IMAGE_DAGGER_EVAL_RELABEL

{
    echo "checkpoint=$CHECKPOINT"
    echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    echo "baseline_config=$BASELINE_CONFIG"
    echo "baseline_config_sha256=$(sha256sum "$BASELINE_CONFIG" | awk '{print $1}')"
    echo "observation_source=raytraced"
    echo "episodes=300"
    echo "num_envs=$NUM_ENVS"
    cd "$GR00T"
    "$PYTHON" gr00t/rl/eval_agent_trl.py \
        +checkpoint="$STAGE/last.pt" \
        +num_envs="$NUM_ENVS" \
        +headless=true \
        algo.config.eval.num_eval_episodes=300 \
        +simulator.config.render_results=true
} 2>&1 | tee "$RESULT_DIR/eval.log"
