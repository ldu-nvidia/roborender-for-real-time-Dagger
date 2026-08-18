#!/bin/bash
# Distributed production DAgger: 256 global Isaac environments, 300 rollout steps.
# This remains the validated pure-BC DAgger objective (teacher action supervision),
# not PPO reward optimization. Submit via scripts/scaling/submit_production.sh.
#SBATCH --partition=viscam-hi
#SBATCH --account=viscam
#SBATCH --job-name=rr_dagger_256x300
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=viscam10,viscam12
#SBATCH --output=/svl/u/ravenh/fast_roborender/logs_archive/rr_dagger_256x300_%j.log
set -Eeuo pipefail

ROOT=${ROOT:-/svl/u/ravenh/fast_roborender}
GR00T_ROOT=${GR00T_ROOT:-$ROOT/GR00T-VisualSim2Real}
FLEET_DIR=${FLEET_DIR:?FLEET_DIR is required; use scripts/scaling/submit_production.sh}
GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}
MIX=${MIX:-0.0}
REFERENCE_LAG=${REFERENCE_LAG:-256}
ENVS_PER_RENDERER=${ENVS_PER_RENDERER:-5}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
NUM_TOTAL_BATCHES=${NUM_TOTAL_BATCHES:-1}
TRAINER_NODES=${SLURM_NNODES:?must run under Slurm}
RENDER_READY_TIMEOUT_S=${RENDER_READY_TIMEOUT_S:-3600}

FAST_REVISION=$(git -C "$ROOT" rev-parse HEAD)
FAST_DIRTY=$(git -C "$ROOT" status --porcelain)
if [ -n "$FAST_DIRTY" ] && [ "${ALLOW_DIRTY_FAST_ROBORENDER:-0}" != "1" ]; then
    echo "FATAL: fast_roborender worktree is dirty; commit this scaling branch first."
    exit 2
fi
if [ -n "${EXPECTED_FAST_ROBORENDER_REVISION:-}" ] && \
   [ "$FAST_REVISION" != "$EXPECTED_FAST_ROBORENDER_REVISION" ]; then
    echo "FATAL: fast_roborender revision $FAST_REVISION != $EXPECTED_FAST_ROBORENDER_REVISION"
    exit 2
fi
: "${EXPECTED_GR00T_REVISION:?EXPECTED_GR00T_REVISION must pin the Raven private trainer}"

PLAN=(
    --global-envs "$GLOBAL_NUM_ENVS"
    --rollout-steps "$ROLLOUT_STEPS"
    --trainer-nodes "$TRAINER_NODES"
    --raytraced-mix "$MIX"
    --envs-per-renderer "$ENVS_PER_RENDERER"
    --reference-lag "$REFERENCE_LAG"
    --num-mini-batches "$NUM_MINI_BATCHES"
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
)
LOCAL_NUM_ENVS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field local_envs)
REQUIRED_RENDERERS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field required_renderers)
GRADIENT_ACCUMULATION_STEPS=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field gradient_accumulation_steps)
TRANSITIONS_PER_UPDATE=$(python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --field transitions_per_update)
[[ "$NUM_TOTAL_BATCHES" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: NUM_TOTAL_BATCHES must be a positive integer, got $NUM_TOTAL_BATCHES"
    exit 2
}

# Until Raven's private trainer proves an episode-sticky mixed cohort, lagged
# streams are production-safe only when every env renders every tick (MIX=0)
# or no env renders (MIX=1). Per-tick Bernoulli mixing creates step gaps.
if [ "$MIX" != "0" ] && [ "$MIX" != "0.0" ] && \
   [ "$MIX" != "1" ] && [ "$MIX" != "1.0" ] && \
   [ "${ALLOW_EXPERIMENTAL_MIXED_LAG:-0}" != "1" ]; then
    echo "FATAL: MIX=$MIX is not validated with contiguous lag-$REFERENCE_LAG streams."
    echo "       Use MIX=0.0 (production default), MIX=1.0, or explicitly set"
    echo "       ALLOW_EXPERIMENTAL_MIXED_LAG=1 after verifying an episode-sticky mask."
    exit 2
fi

ALLOW_DIRTY_ARGS=()
if [ "${ALLOW_DIRTY_GR00T:-0}" = "1" ]; then
    ALLOW_DIRTY_ARGS+=(--allow-dirty)
fi
EXPECTED_REV_ARGS=()
if [ -n "${EXPECTED_GR00T_REVISION:-}" ]; then
    EXPECTED_REV_ARGS+=(--expected-revision "$EXPECTED_GR00T_REVISION")
fi
python3 "$ROOT/scripts/scaling/install_gr00t_integration.py" \
    --gr00t-root "$GR00T_ROOT" "${ALLOW_DIRTY_ARGS[@]}" "${EXPECTED_REV_ARGS[@]}"

CKPT=${CKPT:-$GR00T_ROOT/logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/last.pt}
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 2; }
[ -d "$GR00T_ROOT" ] || { echo "FATAL: GR00T tree not found: $GR00T_ROOT"; exit 2; }

SELECTED_ENDPOINT_MANIFEST="$FLEET_DIR/selected_endpoints.txt"
if [ "$REQUIRED_RENDERERS" -gt 0 ]; then
    EP_LIST=$(python3 "$ROOT/scripts/scaling/image3f_probe.py" \
        --endpoint-dir "$FLEET_DIR/endpoints" \
        --expected "$REQUIRED_RENDERERS" \
        --wait-timeout-s "$RENDER_READY_TIMEOUT_S" \
        --require-reference-history --required-reference-lag "$REFERENCE_LAG" \
        --write-manifest "$SELECTED_ENDPOINT_MANIFEST" --hydra-list)
else
    EP_LIST='[]'
    : > "$SELECTED_ENDPOINT_MANIFEST"
fi

PROJ=${PROJ:-roborender_dagger_scale}
EXP=${EXP:-dagger_${GLOBAL_NUM_ENVS}env_${ROLLOUT_STEPS}step_mix${MIX}}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-$GR00T_ROOT/logs_rl/$PROJ/${EXP}-${SLURM_JOB_ID}}
mkdir -p "$TRAIN_RUN_DIR/slurm" "$TRAIN_RUN_DIR/output"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))

export HOME=${HOME_OVERRIDE:-/sailhome/ravenh}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/svl/u/ravenh/.cache_xdg}
export XDG_DATA_HOME=${XDG_DATA_HOME:-/svl/u/ravenh/.share_xdg}
export TMPDIR=${TMPDIR:-/svl/u/ravenh/tmp}
export OMNI_KIT_ACCEPT_EULA=YES
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export IMAGE3F_ENDPOINT_DIR="$FLEET_DIR/endpoints"
export IMAGE3F_ENDPOINT_MANIFEST="$SELECTED_ENDPOINT_MANIFEST"
export IMAGE3F_LOCAL_ENVS="$LOCAL_NUM_ENVS"
export IMAGE3F_REFERENCE_LAG="$REFERENCE_LAG"
export IMAGE3F_RUN_ID=${IMAGE3F_RUN_ID:-$SLURM_JOB_ID}
export IMAGE3F_REQUEST_TIMEOUT_S=${IMAGE3F_REQUEST_TIMEOUT_S:-120}
export BASE_RENDER_SEED=${BASE_RENDER_SEED:-42}
export ROOT GR00T_ROOT FLEET_DIR GLOBAL_NUM_ENVS ROLLOUT_STEPS MIX REFERENCE_LAG
export LOCAL_NUM_ENVS REQUIRED_RENDERERS GRADIENT_ACCUMULATION_STEPS EP_LIST CKPT
export PROJ EXP TRAIN_RUN_DIR MASTER_ADDR MASTER_PORT TRAINER_NODES
export NUM_MINI_BATCHES PER_DEVICE_TRAIN_BATCH_SIZE
export NUM_TOTAL_BATCHES

python3 "$ROOT/scripts/scaling/scale_plan.py" "${PLAN[@]}" --format json \
    | tee "$TRAIN_RUN_DIR/scale_plan.json"
cat > "$TRAIN_RUN_DIR/launch_manifest.txt" <<EOF
fast_roborender_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
fast_roborender_dirty_status_sha256=$(git -C "$ROOT" status --porcelain | sha256sum | awk '{print $1}')
gr00t_revision=$(git -C "$GR00T_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
gr00t_dirty_status_sha256=$(git -C "$GR00T_ROOT" status --porcelain | sha256sum | awk '{print $1}')
checkpoint=$CKPT
fleet_dir=$FLEET_DIR
trainer_nodes=$TRAINER_NODES
local_envs=$LOCAL_NUM_ENVS
global_envs=$GLOBAL_NUM_ENVS
rollout_steps=$ROLLOUT_STEPS
transitions_per_update=$TRANSITIONS_PER_UPDATE
renderers=$REQUIRED_RENDERERS
reference_lag=$REFERENCE_LAG
num_total_batches=$NUM_TOTAL_BATCHES
EOF

echo "=================================================================="
echo " distributed DAgger: ${TRAINER_NODES} ranks x ${LOCAL_NUM_ENVS} envs = ${GLOBAL_NUM_ENVS}"
echo " rollout: ${ROLLOUT_STEPS} steps, ${TRANSITIONS_PER_UPDATE} transitions/update"
echo " optimizer batches: ${NUM_TOTAL_BATCHES}"
echo " renderers: ${REQUIRED_RENDERERS}, ray-traced mix=${MIX}, RGB lag=${REFERENCE_LAG}"
echo " checkpoint: $CKPT"
echo " output: $TRAIN_RUN_DIR"
echo "=================================================================="

srun --nodes="$TRAINER_NODES" --ntasks="$TRAINER_NODES" --ntasks-per-node=1 \
    --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores --kill-on-bad-exit=1 \
    --output="$TRAIN_RUN_DIR/slurm/rank_%t.out" \
    --error="$TRAIN_RUN_DIR/slurm/rank_%t.err" \
    bash -lc '
        set -Eeuo pipefail
        source /svl/u/ravenh/miniconda3/etc/profile.d/conda.sh
        conda activate viral
        cd "$GR00T_ROOT"
        RANK_RENDER_SEED=$((BASE_RENDER_SEED + SLURM_PROCID))
        exec accelerate launch \
            --multi_gpu \
            --num_machines "$TRAINER_NODES" \
            --num_processes "$TRAINER_NODES" \
            --machine_rank "$SLURM_PROCID" \
            --main_process_ip "$MASTER_ADDR" \
            --main_process_port "$MASTER_PORT" \
            --same_network --rdzv_backend static --max_restarts 0 \
            --num_cpu_threads_per_process "$SLURM_CPUS_PER_TASK" \
            gr00t/rl/train_agent_trl.py \
                +exp=loco_manip/wsdpt_image_dagger \
                num_envs="$LOCAL_NUM_ENVS" \
                algo.config.num_steps_per_env="$ROLLOUT_STEPS" \
                algo.config.num_mini_batches="$NUM_MINI_BATCHES" \
                algo.trl.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
                algo.trl.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
                algo.trl.num_total_batches="$NUM_TOTAL_BATCHES" \
                algo.trl.total_episodes=null \
                headless=true checkpoint="$CKPT" \
                algo.config.videogen.raytraced_mix_prob="$MIX" \
                ++algo.config.videogen.seed="$RANK_RENDER_SEED" \
                ++algo.config.videogen.reference_lag="$REFERENCE_LAG" \
                algo.config.videogen.image3f_endpoint_files="$EP_LIST" \
                algo.config.videogen.anchor_pool_dir="${ANCHORS:-/vision/u/ravenh/anchor_pool_2048}" \
                algo.config.videogen.anchor_index="${ANCHOR_IDX:-0}" \
                simulator.config.render_results=false \
                project_name="$PROJ" experiment_name="$EXP" \
                experiment_dir="$TRAIN_RUN_DIR" \
                hydra.run.dir="$TRAIN_RUN_DIR/.hydra/rank_${SLURM_PROCID}" \
                output_dir="$TRAIN_RUN_DIR/output/rank_${SLURM_PROCID}"
    '

echo RUN_DONE
