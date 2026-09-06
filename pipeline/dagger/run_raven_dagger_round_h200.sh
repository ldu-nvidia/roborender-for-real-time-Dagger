#!/usr/bin/env bash
# Run one of four warm-started Raven-equivalent three-view Image3F DAgger rounds.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T=$ROOT/pipeline/GR00T-VisualSim2Real
QROOT=$ROOT/runtime/isaac_actor/qualified_h200_v1
PYTHON=${GR00T_PYTHON:-$QROOT/runtime/isaacsim-5.1/bin/python}
ROUND=${1:?usage: $0 ROUND(1..4) INPUT_CHECKPOINT [NUM_ENVS]}
CHECKPOINT=${2:?usage: $0 ROUND(1..4) INPUT_CHECKPOINT [NUM_ENVS]}
NUM_ENVS=${3:-16}
TEACHER=$ROOT/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt
ANCHORS=$ROOT/renderer/anchors/roborender_online_anchor_pool_teacher50_2048
OUTPUT=$ROOT/checkpoints/students/dagger_${ROUND}

[[ $ROUND =~ ^[1-4]$ ]] || { echo "ROUND must be 1..4" >&2; exit 2; }
[[ $NUM_ENVS =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be positive" >&2; exit 2; }
for path in "$PYTHON" "$CHECKPOINT" "$TEACHER"; do
    [[ -f $path || -L $path ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ -x $PYTHON ]] || { echo "GR00T Python is not executable: $PYTHON" >&2; exit 2; }

# Endpoint topology is operational, not an experiment parameter. Use a
# colon-separated list of files written by run_image3f_server_h200.sh.
: "${IMAGE3F_ENDPOINT_FILES:?set IMAGE3F_ENDPOINT_FILES to colon-separated endpoint files}"
IFS=: read -r -a ENDPOINTS <<<"$IMAGE3F_ENDPOINT_FILES"
(( ${#ENDPOINTS[@]} > 0 )) || { echo "no Image3F endpoints" >&2; exit 2; }
EP_LIST="["
for endpoint in "${ENDPOINTS[@]}"; do
    [[ $endpoint == /* && -f $endpoint ]] || {
        echo "missing absolute endpoint file: $endpoint" >&2
        exit 2
    }
    read -r host port extra <"$endpoint"
    [[ -n $host && $port =~ ^[0-9]+$ && -z ${extra:-} ]] || {
        echo "malformed endpoint: $endpoint" >&2
        exit 2
    }
    [[ $EP_LIST == "[" ]] || EP_LIST+=","
    EP_LIST+="$endpoint"
done
EP_LIST+="]"

/usr/bin/python3 "$ROOT/pipeline/dagger/validate_raven_image3f_contract.py"
mkdir -p "$OUTPUT"

# Rehydrate the already-qualified Isaac Sim 5.1 runtime. These paths relocate
# Raven's environment; they do not alter policy or DAgger parameters.
source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
export ROUND_DAGGER_EXPECT_POLICY_VIEW_COUNT=3
export IMAGE_DAGGER_DUMP_DIR=$ROOT/runtime/outputs/dagger_${ROUND}_previews
export PYTHONUNBUFFERED=1

cd "$GR00T"
exec "$PYTHON" -m accelerate.commands.launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger_3view \
    num_envs="$NUM_ENVS" \
    headless=true \
    checkpoint="$CHECKPOINT" \
    use_wandb=false \
    teacher_actor_path="$TEACHER" \
    algo.config.homie_walk_model_path="$ROOT/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt" \
    algo.config.homie_stand_model_path="$ROOT/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt" \
    algo.config.num_steps_per_env=81 \
    algo.config.num_mini_batches=1 \
    algo.config.value_loss_coef=0.0 \
    algo.config.entropy_coef=0.0 \
    algo.config.videogen.rollout_action_source=teacher \
    algo.config.videogen.raytraced_mix_prob=0.5 \
    algo.config.videogen.seed=42 \
    algo.config.videogen.anchor_index=0 \
    algo.config.videogen.anchor_pool_dir="$ANCHORS" \
    algo.config.videogen.image3f_endpoint_files="$EP_LIST" \
    base_dir="$OUTPUT" \
    experiment_name="raven_image3f_dagger_round_${ROUND}" \
    project_name="dagger_${ROUND}"
