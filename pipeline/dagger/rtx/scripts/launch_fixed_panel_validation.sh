#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly RAVEN=/home/ldu/BC_render/rtx_BC_transfer
readonly IMAGE=roborender-round-actor:isaaclab2.3.0-checkpoint-compat-20260901
readonly IMAGE_ID=sha256:fcaa595df8f175763ac326187a97dc8bbddca55d6bcc3531cdfecd6be1705a43
readonly GPU_UUID=GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9
readonly BASE_CONFIG=/mnt/data1/ldu/research/artifacts/roborender_bc_corrected_norm_b28_20260909T193611Z/inputs/stage/config.yaml
readonly BASE_CONFIG_SHA256=557c3a64e7061553fcd689016b9a18f909ee6a41a173b2ad4bf55778ca0279e6
readonly METRICS_ADAPTER=/mnt/data1/ldu/research/fast_roborender/scripts/eval/eval_agent_metrics_adapter.py
readonly METRICS_ADAPTER_SHA256=947065989db0d1558881ec32c4559e61c1092525d9c8fac52723f49f92b30f07
readonly MOTION_INVENTORY="$ROOT/config/fixed_panel_motion_inventory.v1.json"
readonly MOTION_INVENTORY_SHA256=4ca087f1f39e99def133c73916904719175fb75262567c5e1d8154bf0b82db01
readonly HOMIE_WALK="$RAVEN/pipeline/GR00T-VisualSim2Real/models/model_walk.pt"
readonly HOMIE_WALK_SHA256=2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e
readonly HOMIE_STAND="$RAVEN/pipeline/GR00T-VisualSim2Real/models/model_stand.pt"
readonly HOMIE_STAND_SHA256=ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054
readonly PROVEN_ISAAC_APPS=/mnt/data1/ldu/research/artifacts/roborender_bc_pre_dagger_stock_eval_20260909T080415Z/cache/isaaclab-apps
readonly CACHE=/home/ldu/BC_render/BC_collection_data/persistent-cache/roborender-round-actor-isaaclab2.3.0-checkpoint-compat-20260901/physical-gpu-1

usage() {
  cat >&2 <<'EOF'
usage: launch_fixed_panel_validation.sh \
  --assignment PATH --assignment-sha256 SHA --evaluation-id ID --run-id ID \
  --round-id 1 --checkpoint PATH --checkpoint-sha256 SHA \
  --scenario-manifest-sha256 SHA --token-file PATH --output-root PATH [--resume]

Runs four sequential 64-environment Raven cohorts on physical RTX GPU1 only.
Each paired seed stays in one process so its native 64-lane RNG stream is not
restarted halfway through the sealed panel.
The H200 evaluation must already be registered on forwarded localhost:18614.
EOF
  exit 2
}

assignment= assignment_sha= evaluation_id= run_id= round_id=
checkpoint= checkpoint_sha= panel_sha= token_file= output_root= resume=0
while (($#)); do
  case "$1" in
    --assignment) assignment=${2:?}; shift 2 ;;
    --assignment-sha256) assignment_sha=${2:?}; shift 2 ;;
    --evaluation-id) evaluation_id=${2:?}; shift 2 ;;
    --run-id) run_id=${2:?}; shift 2 ;;
    --round-id) round_id=${2:?}; shift 2 ;;
    --checkpoint) checkpoint=${2:?}; shift 2 ;;
    --checkpoint-sha256) checkpoint_sha=${2:?}; shift 2 ;;
    --scenario-manifest-sha256) panel_sha=${2:?}; shift 2 ;;
    --token-file) token_file=${2:?}; shift 2 ;;
    --output-root) output_root=${2:?}; shift 2 ;;
    --resume) resume=1; shift ;;
    *) usage ;;
  esac
done
[[ -n $assignment && -n $assignment_sha && -n $evaluation_id && -n $run_id \
   && -n $round_id && -n $checkpoint && -n $checkpoint_sha && -n $panel_sha \
   && -n $token_file && -n $output_root ]] || usage

assignment=$(realpath -e "$assignment")
checkpoint=$(realpath -e "$checkpoint")
token_file=$(realpath -e "$token_file")
output_root=$(realpath -m "$output_root")
[[ $output_root == /mnt/data1/* || $output_root == /mnt/data2/* ]] || {
  echo "output root must be under /mnt/data1 or /mnt/data2" >&2; exit 2;
}
[[ ! -L $assignment && -f $assignment && ! -L $checkpoint && -f $checkpoint ]] || {
  echo "assignment/checkpoint must be regular non-symlink files" >&2; exit 2;
}
[[ $(stat -c '%a' "$token_file") =~ ^[46]00$ ]] || {
  echo "token file must be owner-only" >&2; exit 2;
}
[[ $(sha256sum "$assignment" | awk '{print $1}') == "$assignment_sha" ]] || {
  echo "assignment SHA-256 mismatch" >&2; exit 2;
}
[[ $(sha256sum "$checkpoint" | awk '{print $1}') == "$checkpoint_sha" ]] || {
  echo "checkpoint SHA-256 mismatch" >&2; exit 2;
}
[[ $(sha256sum "$BASE_CONFIG" | awk '{print $1}') == "$BASE_CONFIG_SHA256" \
   && $(sha256sum "$METRICS_ADAPTER" | awk '{print $1}') == "$METRICS_ADAPTER_SHA256" \
   && $(sha256sum "$MOTION_INVENTORY" | awk '{print $1}') == "$MOTION_INVENTORY_SHA256" \
   && $(stat -c '%s' "$HOMIE_WALK") == 9785586 \
   && $(sha256sum "$HOMIE_WALK" | awk '{print $1}') == "$HOMIE_WALK_SHA256" \
   && $(stat -c '%s' "$HOMIE_STAND") == 9785586 \
   && $(sha256sum "$HOMIE_STAND" | awk '{print $1}') == "$HOMIE_STAND_SHA256" ]] || {
  echo "fixed-panel source/config asset identity mismatch" >&2; exit 2;
}
[[ $(docker image inspect "$IMAGE" --format '{{.Id}}') == "$IMAGE_ID" ]] || {
  echo "certified Raven container image identity mismatch" >&2; exit 2;
}

gpu_identity=$(nvidia-smi -i "$GPU_UUID" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits)
[[ $(wc -l <<<"$gpu_identity") -eq 1 ]] || { echo "GPU1 UUID is not unique" >&2; exit 2; }
IFS=',' read -r gpu_index gpu_observed gpu_name gpu_memory gpu_util <<<"$gpu_identity"
gpu_index=${gpu_index//[[:space:]]/}; gpu_observed=${gpu_observed//[[:space:]]/}
gpu_memory=${gpu_memory//[[:space:]]/}; gpu_util=${gpu_util//[[:space:]]/}
[[ $gpu_index == 1 && $gpu_observed == "$GPU_UUID" && $gpu_name == *"RTX PRO 6000"* ]] || {
  echo "physical GPU1 identity gate failed" >&2; exit 2;
}
((gpu_memory < 1024 && gpu_util < 10)) || {
  echo "physical GPU1 is not idle: ${gpu_memory}MiB ${gpu_util}%" >&2; exit 3;
}

if [[ -e $output_root && $resume != 1 ]]; then
  echo "output root already exists; use a fresh path or explicit --resume" >&2
  exit 3
fi
install -d -m 0700 "$output_root" "$output_root/cohorts" "$output_root/outcome-ledger"
exec > >(tee -a "$output_root/launcher.log") 2>&1

common=(
  --assignment "$assignment"
  --assignment-sha256 "$assignment_sha"
  --evaluation-id "$evaluation_id"
  --run-id "$run_id"
  --round-id "$round_id"
  --checkpoint-sha256 "$checkpoint_sha"
  --scenario-manifest-sha256 "$panel_sha"
)
"$ROOT/.venv/bin/python" "$ROOT/scripts/fixed_panel_validation.py" plan \
  "${common[@]}" --output "$output_root/execution-plan.json"
"$ROOT/.venv/bin/python" "$ROOT/scripts/fixed_panel_validation.py" attest \
  "${common[@]}" --ledger-root "$output_root/outcome-ledger" --token-file "$token_file"

exec 9>"$CACHE/.writer.lock"
flock -n 9 || { echo "GPU1 Isaac cache has another writer" >&2; exit 3; }

mapfile -t cohort_rows < <("$ROOT/.venv/bin/python" - "$output_root/execution-plan.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
for row in value["cohorts"]:
    print(f'{row["cohort_index"]}\t{row["paired_seed"]}\t{row["cohort_id"]}')
PY
)
[[ ${#cohort_rows[@]} -eq 4 ]] || { echo "execution plan does not contain four cohorts" >&2; exit 2; }

motion_paths='[/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/record_2025-10-14_21-29-22_0.675m.npz,/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/record_2025-10-14_21-46-10_0.700m.npz,/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/record_2025-10-14_21-52-25_0.725m.npz,/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/record_2025-10-14_22-00-53_0.750m.npz,/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/record_2025-10-14_22-18-36_0.775m.npz]'

for row in "${cohort_rows[@]}"; do
  IFS=$'\t' read -r cohort_index seed cohort_id <<<"$row"
  cohort_root="$output_root/cohorts/$cohort_id"
  install -d -m 0700 "$cohort_root" "$cohort_root/runtime" "$cohort_root/results"
  if [[ -f $cohort_root/result.json ]]; then
    [[ $resume == 1 ]] || { echo "unexpected existing cohort result: $cohort_id" >&2; exit 3; }
    echo "FIXED_PANEL_COHORT_SKIP already complete $cohort_id"
    continue
  fi
  name="roborender-fixed-panel-${evaluation_id}-${cohort_id}"
  echo "FIXED_PANEL_COHORT_START index=$cohort_index seed=$seed B=64 $(date -u +%FT%TZ)"
  docker run --rm --name "$name" --gpus "device=$GPU_UUID" \
    --user "$(id -u):$(id -g)" --group-add 1234 \
    --shm-size 16g --ulimit memlock=-1 --tmpfs /tmp:rw,exec,nosuid,size=8g \
    --network host \
    --mount "type=bind,src=$RAVEN,dst=/workspace/rtx_BC_transfer,readonly" \
    --mount "type=bind,src=$RAVEN/pipeline/bc_collection/image_sitecustomize.py,dst=/opt/roborender-actor-compat/sitecustomize.py,readonly" \
    --mount "type=bind,src=$ROOT/code/src,dst=/opt/dagger-rtx-src,readonly" \
    --mount "type=bind,src=$output_root,dst=/eval" \
    --mount "type=bind,src=$assignment,dst=/eval/assignment.json,readonly" \
    --mount "type=bind,src=$checkpoint,dst=/eval/inputs/stage/last.pt,readonly" \
    --mount "type=bind,src=$BASE_CONFIG,dst=/eval/inputs/stage/config.yaml,readonly" \
    --mount "type=bind,src=$MOTION_INVENTORY,dst=/eval/motion-inventory.json,readonly" \
    --mount "type=bind,src=$token_file,dst=/run/secrets/dagger-evaluation.token,readonly" \
    --mount "type=bind,src=$METRICS_ADAPTER,dst=/opt/eval_agent_metrics_adapter.py,readonly" \
    --mount "type=bind,src=$PROVEN_ISAAC_APPS,dst=/workspace/isaaclab/apps" \
    --mount "type=bind,src=$CACHE/rank-cache,dst=/persistent-cache" \
    --mount "type=bind,src=$CACHE/optix-cache,dst=/var/tmp/OptixCache" \
    --workdir /workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real \
    --env HOME=/eval/cache/home \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    --env CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    --env PYTHONPATH=/opt/dagger-rtx-src:/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real:/opt/roborender-actor-compat \
    --env XDG_CACHE_HOME=/persistent-cache/xdg-cache \
    --env XDG_CONFIG_HOME=/persistent-cache/xdg-config \
    --env XDG_DATA_HOME=/persistent-cache/xdg-data \
    --env HF_HOME=/persistent-cache/hf \
    --env MPLCONFIGDIR=/persistent-cache/matplotlib \
    --env TORCH_HOME=/persistent-cache/torch \
    --env WARP_CACHE_PATH=/persistent-cache/warp \
    --env CUDA_CACHE_PATH=/persistent-cache/cuda \
    --env NVIDIA_SHADER_DISK_CACHE_PATH=/persistent-cache/cuda \
    --env OMNI_USER_CONFIG_PATH=/persistent-cache/omniverse \
    --env OMNI_CLIENT_CACHE_PATH=/persistent-cache/ov \
    --env OV_CACHE_ROOT=/persistent-cache/ov \
    --env HYDRA_FULL_ERROR=1 \
    --env WANDB_MODE=disabled \
    --env IMAGE_DAGGER_EVAL_RELABEL=0 \
    --env IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1 \
    --env IMAGE_DAGGER_EXPECT_PRIMARY_VIEW=head_color \
    --env IMAGE_DAGGER_EXPECT_POLICY_VIEWS=head_color,hand_left,hand_right \
    --env IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT=3 \
    --env IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT=vertical \
    --env IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE=324,192,3 \
    --env DAGGER_FIXED_PANEL_ASSIGNMENT=/eval/assignment.json \
    --env DAGGER_FIXED_PANEL_ASSIGNMENT_SHA256="$assignment_sha" \
    --env DAGGER_FIXED_PANEL_EVALUATION_ID="$evaluation_id" \
    --env DAGGER_FIXED_PANEL_RUN_ID="$run_id" \
    --env DAGGER_FIXED_PANEL_ROUND_ID="$round_id" \
    --env DAGGER_FIXED_PANEL_CHECKPOINT=/eval/inputs/stage/last.pt \
    --env DAGGER_FIXED_PANEL_CHECKPOINT_SHA256="$checkpoint_sha" \
    --env DAGGER_FIXED_PANEL_SCENARIO_MANIFEST_SHA256="$panel_sha" \
    --env DAGGER_FIXED_PANEL_MOTION_INVENTORY=/eval/motion-inventory.json \
    --env DAGGER_FIXED_PANEL_MOTION_INVENTORY_SHA256="$MOTION_INVENTORY_SHA256" \
    --env DAGGER_FIXED_PANEL_COHORT_INDEX="$cohort_index" \
    --env DAGGER_FIXED_PANEL_EFFECTIVE_SEED="$seed" \
    --env DAGGER_FIXED_PANEL_LEDGER_ROOT=/eval/outcome-ledger \
    --env DAGGER_FIXED_PANEL_TOKEN_FILE=/run/secrets/dagger-evaluation.token \
    --env DAGGER_FIXED_PANEL_COHORT_OUTPUT="/eval/cohorts/$cohort_id/result.json" \
    "$IMAGE" /opt/round-dagger-rtx/bin/python /opt/eval_agent_metrics_adapter.py \
      --evaluator /workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/eval_agent_trl.py \
      --metrics-output "/eval/cohorts/$cohort_id/results/metrics_eval.json" -- \
      +checkpoint=/eval/inputs/stage/last.pt \
      +num_envs=64 +seed="$seed" +headless=true \
      +trainer._target_=dagger_rtx.fixed_panel_raven.FixedPanelEvalTrainer \
      +env._target_=dagger_rtx.fixed_panel_raven.FixedPanelWalkStandPlaceGraspTurnHomie \
      +simulator._target_=dagger_rtx.fixed_panel_raven.NormalizedIsaacSim \
      +algo.config.init_at_random_ep_len=false \
      algo.config.eval.num_eval_episodes=64 \
      +algo.config.eval.eval_num_envs_episodes=true \
      algo.config.eval.save_videos=false \
      algo.config.eval.save_trajectories=false \
      +algo.config.use_dagger=false \
      +env.config.domain_rand.image_augmentation.enabled=false \
      +env.config.reset_from_dataset.use_motion_file_dir=false \
      "+env.config.reset_from_dataset.motion_files=$motion_paths" \
      +env.config.reset_from_dataset.motion_assignment_offset=0 \
      +simulator.config.render_results=true \
      +env.config.save_rendering_dir="/eval/cohorts/$cohort_id/runtime/renderings" \
      +use_wandb=false \
      +eval_output_dir="/eval/cohorts/$cohort_id/results" \
      hydra.run.dir="/eval/cohorts/$cohort_id/runtime/hydra" \
      +output_dir="/eval/cohorts/$cohort_id/runtime/output"
  [[ -f $cohort_root/result.json ]] || { echo "cohort result missing: $cohort_id" >&2; exit 3; }
  echo "FIXED_PANEL_COHORT_END index=$cohort_index $cohort_id $(date -u +%FT%TZ)"
done

"$ROOT/.venv/bin/python" "$ROOT/scripts/fixed_panel_validation.py" summarize \
  "${common[@]}" --ledger-root "$output_root/outcome-ledger" \
  --token-file "$token_file" --output "$output_root/summary.json"
sha256sum "$output_root/execution-plan.json" "$output_root/summary.json" \
  "$output_root"/cohorts/*/result.json > "$output_root/SHA256SUMS"
chmod 0400 "$output_root/SHA256SUMS"
echo "FIXED_PANEL_COMPLETE output=$output_root $(date -u +%FT%TZ)"
