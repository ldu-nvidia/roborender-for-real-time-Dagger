#!/bin/bash
# Raven-equivalent three-view offline BC. Only NUM_ENVS and execution topology
# are exposed; scientific hyperparameters remain in the pinned Hydra config.
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T_ROOT=$PROJECT_ROOT/pipeline/GR00T-VisualSim2Real
QUALIFIED_ROOT=$PROJECT_ROOT/runtime/isaac_actor/qualified_h200_v1
source "$PROJECT_ROOT/pipeline/common/setup_h200_isaac_runtime.sh"
NUM_ENVS=${NUM_ENVS:-16}

[[ $NUM_ENVS =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be positive" >&2; exit 2; }
[[ -x $GR00T_PYTHON ]] || { echo "GR00T_PYTHON is not executable" >&2; exit 2; }
"$GR00T_PYTHON" -c 'import accelerate, trl' 2>/dev/null || {
    echo "qualified runtime lacks Raven training dependencies: accelerate and/or trl" >&2
    exit 2
}

cd "$GR00T_ROOT"
exec "$GR00T_PYTHON" -m accelerate.commands.launch \
    --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_videogen_offline_bc_3view \
    num_envs="$NUM_ENVS" \
    headless=True \
    use_wandb=false \
    teacher_actor_path="$PROJECT_ROOT/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt" \
    algo.config.homie_walk_model_path="$PROJECT_ROOT/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt" \
    algo.config.homie_stand_model_path="$PROJECT_ROOT/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt" \
    algo.config.videogen.dataset_dir="$PROJECT_ROOT/data/BC/rendered" \
    algo.config.videogen.dump_dir="$PROJECT_ROOT/runtime/outputs/bc_training/videogen_frames" \
    base_dir="$PROJECT_ROOT/checkpoints/students/BC" \
    experiment_name=videogen_offline_bc \
    project_name=videogen_offline_bc
