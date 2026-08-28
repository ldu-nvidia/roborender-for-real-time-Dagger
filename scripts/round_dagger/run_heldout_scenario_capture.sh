#!/usr/bin/env bash
# One live, non-stepping Isaac/Image3F held-out scenario capture pass.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: run_heldout_scenario_capture.sh --plan FILE --seed N --pass-index 1|2
       --output-dir DIR --gr00t-root DIR --checkpoint FILE
       --checkpoint-sha256 SHA --evaluation-config FILE
       --motion-split FILE --motion-split-sha256 SHA
       --endpoint-dir DIR --expected-renderers N [--python PYTHON] [--smoke2]

Production runs capture one real Image3F response for 32 planned environments,
then exit before env.step.  --smoke2 runs exactly two planned environments and
publishes explicitly non-release evidence.  Run production twice per seed with
pass indices 1 and 2; round_dagger.scenario_capture attests the complete 8x32
campaign.

The caller must provide the production identity environment variables checked
below.  No identity is inferred from an interactive checkout.
EOF
}

fatal() { echo "FATAL: $*" >&2; exit 2; }
PLAN=
SEED=
PASS_INDEX=
OUTPUT_DIR=
GR00T_ROOT=
CHECKPOINT=
CHECKPOINT_SHA=
EVALUATION_CONFIG=
MOTION_SPLIT=
MOTION_SPLIT_SHA=
ENDPOINT_DIR=
EXPECTED_RENDERERS=
PYTHON_BIN=python3
SMOKE2=0
while (($#)); do
    case "$1" in
        --plan) PLAN=${2:?}; shift 2 ;;
        --seed) SEED=${2:?}; shift 2 ;;
        --pass-index) PASS_INDEX=${2:?}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
        --gr00t-root) GR00T_ROOT=${2:?}; shift 2 ;;
        --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
        --checkpoint-sha256) CHECKPOINT_SHA=${2:?}; shift 2 ;;
        --evaluation-config) EVALUATION_CONFIG=${2:?}; shift 2 ;;
        --motion-split) MOTION_SPLIT=${2:?}; shift 2 ;;
        --motion-split-sha256) MOTION_SPLIT_SHA=${2:?}; shift 2 ;;
        --endpoint-dir) ENDPOINT_DIR=${2:?}; shift 2 ;;
        --expected-renderers) EXPECTED_RENDERERS=${2:?}; shift 2 ;;
        --python) PYTHON_BIN=${2:?}; shift 2 ;;
        --smoke2) SMOKE2=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fatal "unknown argument: $1" ;;
    esac
done
for name in PLAN SEED PASS_INDEX OUTPUT_DIR GR00T_ROOT CHECKPOINT CHECKPOINT_SHA \
    EVALUATION_CONFIG MOTION_SPLIT MOTION_SPLIT_SHA ENDPOINT_DIR \
    EXPECTED_RENDERERS; do
    [[ -n ${!name} ]] || fatal "missing ${name,,}"
done
[[ $SEED =~ ^(0|[1-9][0-9]*)$ ]] || fatal "seed must be non-negative"
[[ $PASS_INDEX == 1 || $PASS_INDEX == 2 ]] || fatal "pass index must be 1 or 2"
[[ $EXPECTED_RENDERERS =~ ^[1-9][0-9]*$ ]] || fatal "renderer count is invalid"
[[ $CHECKPOINT_SHA =~ ^[0-9a-f]{64}$ ]] || fatal "checkpoint SHA is malformed"
[[ $MOTION_SPLIT_SHA =~ ^[0-9a-f]{64}$ ]] || fatal "motion split SHA is malformed"
for path in "$PLAN" "$CHECKPOINT" "$EVALUATION_CONFIG" "$MOTION_SPLIT"; do
    [[ $path == /* && -f $path && ! -L $path ]] || \
        fatal "input must be an absolute non-symlink regular file: $path"
done
for path in "$GR00T_ROOT" "$ENDPOINT_DIR"; do
    [[ $path == /* && -d $path && ! -L $path ]] || \
        fatal "input must be an absolute non-symlink directory: $path"
done
[[ $OUTPUT_DIR == /* && ! -e $OUTPUT_DIR ]] || \
    fatal "output directory must be a fresh absolute path"
[[ $(sha256sum "$CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA" ]] || \
    fatal "checkpoint SHA mismatch"
[[ $(sha256sum "$MOTION_SPLIT" | awk '{print $1}') == "$MOTION_SPLIT_SHA" ]] || \
    fatal "motion split SHA mismatch"
CHECKPOINT_PARENT=$(dirname "$CHECKPOINT")
LOADED_CONFIG="$CHECKPOINT_PARENT/config.yaml"
if [[ ! -f $LOADED_CONFIG || -L $LOADED_CONFIG ]]; then
    LOADED_CONFIG=$(dirname "$CHECKPOINT_PARENT")/config.yaml
fi
[[ -f $LOADED_CONFIG && ! -L $LOADED_CONFIG ]] || \
    fatal "checkpoint has no safe evaluator config sibling"
[[ $(realpath "$LOADED_CONFIG") == $(realpath "$EVALUATION_CONFIG") ]] || \
    fatal "provided evaluation config is not the config the evaluator will load"
PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
CONFIG_SHA=$(sha256sum "$EVALUATION_CONFIG" | awk '{print $1}')
[[ ${ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256:-} == "$PLAN_SHA" ]] || \
    fatal "held-out plan differs from the production campaign identity"
[[ ${ROUND_DAGGER_EVALUATOR_CONFIG_SHA256:-} == "$CONFIG_SHA" ]] || \
    fatal "evaluation config differs from pinned identity"
[[ ${ROUND_DAGGER_POLICY_SHA256:-} == "$CHECKPOINT_SHA" ]] || \
    fatal "checkpoint differs from pinned parent identity"
[[ ${ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256:-} == "$MOTION_SPLIT_SHA" ]] || \
    fatal "motion split differs from pinned identity"

required_sha_environment=(
    ROUND_DAGGER_RUNTIME_POLICY_SHA256
    ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256
    ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256
    ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256
    ROUND_DAGGER_RENDERER_RUNTIME_SHA256
    ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256
    ROUND_DAGGER_ANCHOR_INVENTORY_SHA256
    ROUND_DAGGER_RTX_IMAGE_ID_SHA256
    ROUND_DAGGER_RENDERER_SHA256
)
for name in "${required_sha_environment[@]}"; do
    [[ ${!name:-} =~ ^[0-9a-f]{64}$ ]] || fatal "$name must be a lowercase SHA-256"
done
for name in ROUND_DAGGER_FAST_ROBORENDER_REVISION ROUND_DAGGER_GR00T_REVISION; do
    [[ ${!name:-} =~ ^[0-9a-f]{40}$ ]] || fatal "$name must be a 40-character revision"
done

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $GR00T_ROOT == "$REPOSITORY_ROOT/GR00T-VisualSim2Real" ]] || \
    fatal "GR00T root must belong to this exact source checkout"
[[ ${ROUND_DAGGER_CAPTURE_EXECUTION_CONTEXT:-} == actor_docker ]] || \
    fatal "capture launcher must execute inside the validated actor Docker"
[[ ${ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH:-} == 1 ]] || \
    fatal "capture requires the trusted coordinator evaluation dispatcher"
[[ ${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:-} == "$REPOSITORY_ROOT" ]] || \
    fatal "capture source root differs from the sealed evaluation source"
[[ -f $REPOSITORY_ROOT/.source_revision && \
   ! -L $REPOSITORY_ROOT/.source_revision ]] || \
    fatal "sealed fast_roborender source revision is missing"
[[ -f $GR00T_ROOT/.source_revision && ! -L $GR00T_ROOT/.source_revision ]] || \
    fatal "sealed GR00T source revision is missing"
[[ $(tr -d '\r\n' < "$REPOSITORY_ROOT/.source_revision") == \
   "$ROUND_DAGGER_FAST_ROBORENDER_REVISION" ]] || \
    fatal "sealed fast_roborender source revision mismatch"
[[ $(tr -d '\r\n' < "$GR00T_ROOT/.source_revision") == \
   "$ROUND_DAGGER_GR00T_REVISION" ]] || \
    fatal "sealed GR00T source revision mismatch"
EVALUATOR="$GR00T_ROOT/gr00t/rl/eval_agent_trl.py"
PROBE="$REPOSITORY_ROOT/scripts/scaling/image3f_probe.py"
[[ -f $EVALUATOR && ! -L $EVALUATOR && -f $PROBE && ! -L $PROBE ]] || \
    fatal "capture evaluator/probe source is missing"

readarray -t ASSIGNMENT < <("$PYTHON_BIN" - "$PLAN" "$PLAN_SHA" "$SEED" "$SMOKE2" <<'PY'
import hashlib, json, sys
path, expected, seed, smoke = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
payload = open(path, "rb").read()
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("plan changed while reading")
plan = json.loads(payload)
if plan.get("schema") != "fast_roborender.round_dagger.heldout_motion_capture_plan.v2":
    raise SystemExit("plan schema mismatch")
cases = sorted(
    (case for case in plan.get("capture", {}).get("cases", []) if case.get("seed") == seed),
    key=lambda case: case.get("env_index", -1),
)
count = 2 if smoke else 32
cases = cases[:count]
if len(cases) != count or [case.get("env_index") for case in cases] != list(range(count)):
    raise SystemExit("plan has no complete capture block for seed/profile")
indices = [case.get("global_case_index") for case in cases]
if indices != list(range(indices[0], indices[0] + count)):
    raise SystemExit("capture block is not contiguous")
print(indices[0])
print(count)
PY
) || fatal "cannot derive held-out capture assignment"
[[ ${#ASSIGNMENT[@]} -eq 2 ]] || fatal "capture assignment output is incomplete"
OFFSET=${ASSIGNMENT[0]}
NUM_ENVS=${ASSIGNMENT[1]}
PROFILE=production32
((SMOKE2)) && PROFILE=smoke2

mkdir -p "$OUTPUT_DIR"
chmod 0700 "$OUTPUT_DIR"
RUNTIME_OUTPUT="$OUTPUT_DIR/scenario_capture_runtime.json"
BEFORE="$OUTPUT_DIR/renderer_before.json"
AFTER="$OUTPUT_DIR/renderer_after.json"
PASS_OUTPUT="$OUTPUT_DIR/scenario_capture_pass.json"
NONCE=$("$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)
RUN_ID="heldout-s${SEED}-p${PASS_INDEX}-${NONCE}"

mapfile -t ENDPOINT_FILES < <(find "$ENDPOINT_DIR" -maxdepth 1 -type f \
    -name 'renderer_*.txt' -print | LC_ALL=C sort)
[[ ${#ENDPOINT_FILES[@]} -eq $EXPECTED_RENDERERS ]] || \
    fatal "endpoint fleet does not contain the expected renderer count"
ENDPOINT_LIST="[$(IFS=,; echo "${ENDPOINT_FILES[*]}")]"

probe=("$PYTHON_BIN" "$PROBE" --endpoint-dir "$ENDPOINT_DIR"
    --expected "$EXPECTED_RENDERERS" --required-reference-lag 256
    --require-reference-history
    --expected-output-views head_color,hand_left,hand_right
    --expected-output-layout vertical --wait-timeout-s 120 --connect-timeout-s 10
    --json)
"${probe[@]}" --write-report "$BEFORE" >/dev/null

export PYTHONPATH="$REPOSITORY_ROOT:$GR00T_ROOT"
export TRAIN_RUN_DIR="$OUTPUT_DIR"
export IMAGE_DAGGER_SCENARIO_CAPTURE=1
export IMAGE_DAGGER_SCENARIO_CAPTURE_PROFILE="$PROFILE"
export IMAGE_DAGGER_SCENARIO_CAPTURE_PLAN="$PLAN"
export IMAGE_DAGGER_SCENARIO_CAPTURE_PASS_INDEX="$PASS_INDEX"
export IMAGE_DAGGER_SCENARIO_CAPTURE_RUN_ID="$RUN_ID"
export IMAGE_DAGGER_SCENARIO_CAPTURE_NONCE="$NONCE"
export IMAGE_DAGGER_SCENARIO_CAPTURE_OUTPUT="$RUNTIME_OUTPUT"
export IMAGE_DAGGER_EVAL_RELABEL=1
export IMAGE_DAGGER_PAIRED_EVAL=0
export IMAGE3F_RUN_ID="$RUN_ID"
export IMAGE3F_ENDPOINT_DIR="$ENDPOINT_DIR"
export IMAGE3F_LOCAL_ENVS="$NUM_ENVS"
export ROUND_DAGGER_SCENARIO_CAPTURE_PLAN_SHA256="$PLAN_SHA"

"$PYTHON_BIN" "$EVALUATOR" \
    ++checkpoint="$CHECKPOINT" ++num_envs="$NUM_ENVS" ++headless=true \
    ++seed="$SEED" ++eval_output_dir="$OUTPUT_DIR" \
    ++algo.config.eval.eval_num_envs_episodes=false \
    ++algo.config.eval.num_eval_episodes=1 \
    ++algo.config.eval.save_videos=false \
    ++algo.config.eval.save_trajectories=false \
    ++algo.config.eval.training_enabled=false \
    ++algo.config.init_at_random_ep_len=false \
    ++algo.config.videogen.enable=true \
    ++algo.config.videogen.mode=image_dagger \
    ++algo.config.videogen.raytraced_mix_prob=0.0 \
    ++algo.config.videogen.image3f_endpoint_files="$ENDPOINT_LIST" \
    '++algo.config.videogen.view_names=[head_color,hand_left,hand_right]' \
    ++algo.config.videogen.reference_lag=256 \
    ++algo.config.videogen.primary_view=head_color \
    ++algo.config.videogen.seed="$SEED" \
    ++algo.config.videogen.anchor_source=simulator_rgb \
    '++simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]' \
    ++simulator.config.cameras.policy_view_layout=vertical \
    ++env.config.image_dagger_policy_view_count=3 \
    ++env.config.reset_from_dataset.motion_split_manifest="$MOTION_SPLIT" \
    ++env.config.reset_from_dataset.motion_split_manifest_sha256="$MOTION_SPLIT_SHA" \
    ++env.config.reset_from_dataset.motion_split_role=eval \
    ++env.config.reset_from_dataset.motion_split_offset="$OFFSET" \
    ++env.config.reset_from_dataset.motion_split_limit="$NUM_ENVS" \
    ++env.config.reset_from_dataset.motion_assignment_offset=0 \
    ++env.config.reset_from_dataset.motion_split_verify_content=true \
    ++simulator.config.render_results=false ++use_wandb=false

[[ -f $RUNTIME_OUTPUT && ! -L $RUNTIME_OUTPUT ]] || \
    fatal "evaluator did not publish immutable scenario capture evidence"
"${probe[@]}" --write-report "$AFTER" >/dev/null

attest=("$PYTHON_BIN" -m round_dagger.scenario_capture attest-pass
    --plan "$PLAN" --runtime "$RUNTIME_OUTPUT"
    --renderer-before "$BEFORE" --renderer-after "$AFTER"
    --output "$PASS_OUTPUT")
((SMOKE2)) && attest+=(--smoke2)
"${attest[@]}" >/dev/null
verify=("$PYTHON_BIN" -m round_dagger.scenario_capture verify-pass
    --plan "$PLAN" --pass-artifact "$PASS_OUTPUT")
((SMOKE2)) && verify+=(--smoke2)
"${verify[@]}" >/dev/null
echo "HELDOUT_SCENARIO_CAPTURE_PASS seed=$SEED pass=$PASS_INDEX profile=$PROFILE envs=$NUM_ENVS artifact=$PASS_OUTPUT"
