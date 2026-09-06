#!/usr/bin/env bash
# Launch one independent H200 shard of Raven's exact BC collector.
set -Eeuo pipefail

if (( $# != 6 )); then
    echo "usage: $0 GPU RANK WORLD_SIZE ENVS_PER_RANK OUTPUT_DIR RUN_TAG" >&2
    exit 2
fi

gpu=$1
rank=$2
world_size=$3
envs_per_rank=$4
output_dir=$5
run_tag=$6

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GR00T=$ROOT/pipeline/GR00T-VisualSim2Real
RUNTIME=$ROOT/runtime/isaac_actor/qualified_h200_v1
PYTHON=$RUNTIME/runtime/isaacsim-5.1/bin/python
ACCELERATE=$RUNTIME/runtime/isaacsim-5.1/bin/accelerate
TEACHER=$ROOT/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt
WALK=$ROOT/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt
STAND=$ROOT/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt

for file in "$PYTHON" "$ACCELERATE" "$TEACHER" "$WALK" "$STAND"; do
    [[ -e $file ]] || { echo "missing required input: $file" >&2; exit 2; }
done
[[ $gpu =~ ^[0-9]+$ && $rank =~ ^[0-9]+$ && $world_size =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $envs_per_rank =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $run_tag =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
mkdir -p "$output_dir"
rank_tag=$(printf '%s_r%02d' "$run_tag" "$rank")
[[ -z $(find "$output_dir" -maxdepth 1 -type f -name "chunk_${rank_tag}_*.npz" -print -quit) ]] || {
    echo "rank/run output already exists in: $output_dir" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES=$gpu
export ROBORENDER_BC_RANK=$rank
export ROBORENDER_BC_WORLD_SIZE=$world_size
export ROBORENDER_BC_RUN_TAG=$run_tag
export ROBORENDER_BC_ISAAC_OUTPUT_DIR=$output_dir
# Raven uses SLURM_JOB_ID as the chunk prefix. A rank-unique local value is a
# filename/topology relocation only and prevents eight independent processes
# from overwriting one another.
export SLURM_JOB_ID=$rank_tag
export ROBORENDER_HEADLESS_EXPERIENCE=$RUNTIME/runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit

# Reuse only the qualified H200 Python/Isaac library environment. This does
# not alter Raven's environment, camera, rollout, or dataset parameters.
source "$ROOT/pipeline/common/setup_h200_isaac_runtime.sh"

cd "$GR00T"
exec "$ACCELERATE" launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_videogen_collect \
    "num_envs=$envs_per_rank" \
    "seed=$rank" \
    "teacher_actor_path=$TEACHER" \
    "algo.config.homie_walk_model_path=$WALK" \
    "algo.config.homie_stand_model_path=$STAND" \
    "algo.config.videogen.collect_dir=$output_dir" \
    "algo.config.videogen.dump_dir=null" \
    headless=True \
    use_wandb=false \
    "experiment_name=raven_bc_collect_${run_tag}_r$(printf '%02d' "$rank")" \
    project_name=roborender_bc_collection
