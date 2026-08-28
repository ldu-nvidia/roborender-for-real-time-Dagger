#!/usr/bin/env bash
# Production-only collect hook: verify every static input and exact 256x512
# dynamic cohort identity before launching the two isolated RTX actor ranks.
set -Eeuo pipefail

DRY_RUN=${DRY_RUN:-0}
if [[ ${ROUND_DAGGER_TRUSTED_COLLECT_DISPATCH:-0} == 1 ]]; then
    ROOT=${ROUND_DAGGER_ACTOR_SOURCE_ROOT:?missing ROUND_DAGGER_ACTOR_SOURCE_ROOT}
    [[ $ROOT == /* && -d $ROOT && ! -L $ROOT ]] || {
        echo "production collect preflight failed: sealed actor source is unsafe" >&2
        exit 2
    }
    [[ $(cd "$ROOT" && pwd -P) == "$ROOT" ]] || {
        echo "production collect preflight failed: sealed actor source is not canonical" >&2
        exit 2
    }
else
    [[ $DRY_RUN == 1 ]] || {
        echo "production collect preflight failed: untrusted collect dispatch" >&2
        exit 2
    }
    # A direct dry-run remains useful for a zero-GPU preflight.  Production is
    # dispatched only from the coordinator's verified anonymous-memory copy.
    ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
PAIR_SCRIPT="$ROOT/scripts/round_dagger/run_actor_cohort_pair.sh"
ANCHOR_VERIFIER="$ROOT/scripts/round_dagger/verify_anchor_pool.py"

EXPECTED_TEACHER_SHA256=943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38
EXPECTED_ANCHOR_INVENTORY_SHA256=ee358a97581512a5d03730d06efd5804e53844f517112dc0a845ed86b6dae46f
EXPECTED_ACTOR_IMAGE_ID=sha256:65a0cbe64eb71ee2edb672bed3ec7bdaa8adfeb22298c0e99af8b91d74dc78f8
EXPECTED_MOTION_SPLIT_MANIFEST_SHA256=6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834
MOTION_SPLIT_RELATIVE=GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775_heldout_256_v1.json

fatal() { echo "production collect preflight failed: $*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

required=(
    ROUND_DAGGER_CAMPAIGN_ROOT ROUND_DAGGER_CAMPAIGN_ID
    ROUND_DAGGER_MACRO_ROUND ROUND_DAGGER_COHORT_INDEX
    ROUND_DAGGER_COHORT_DIR ROUND_DAGGER_GLOBAL_ENV_OFFSET
    ROUND_DAGGER_TARGET_ENVIRONMENTS ROUND_DAGGER_COHORT_COUNT
    ROUND_DAGGER_ENVIRONMENTS_PER_COHORT ROUND_DAGGER_LOGICAL_ENV_COUNT
    ROUND_DAGGER_STEPS_PER_ENVIRONMENT ROUND_DAGGER_EXPECTED_TRANSITIONS
    ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND
    ROUND_DAGGER_TEACHER_CHECKPOINT ROUND_DAGGER_TEACHER_SHA256
    ROUND_DAGGER_RENDERER_SHA256 IMAGE3F_ENDPOINT_MANIFEST ANCHORS
    ROUND_DAGGER_ENDPOINT_MANIFEST_SHA256
    IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE
    ROUND_DAGGER_RTX_IMAGE ROUND_DAGGER_DOCKER_BIN
    ROUND_DAGGER_CAMERA_ACCEPTANCE ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256
    ROUND_DAGGER_GR00T_REVISION ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256
    ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST
    ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256
    ROUND_DAGGER_CAMPAIGN_CONTRACT ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256
    ROUND_DAGGER_DATA_LINEAGE_SHA256
    ROUND_DAGGER_WANDB_NETRC
    ROUND_DAGGER_ACTOR_RUN_DIR TRAIN_RUN_DIR
)
for name in "${required[@]}"; do
    [ -n "${!name:-}" ] || fatal "missing required environment: $name"
done

case "$ROUND_DAGGER_CAMPAIGN_ROOT:$ROUND_DAGGER_COHORT_DIR:$ROUND_DAGGER_TEACHER_CHECKPOINT:$IMAGE3F_ENDPOINT_MANIFEST:$ANCHORS:$ROUND_DAGGER_CAMERA_ACCEPTANCE:$ROUND_DAGGER_WANDB_NETRC:$ROUND_DAGGER_ACTOR_RUN_DIR:$TRAIN_RUN_DIR:$ROUND_DAGGER_DOCKER_BIN:$ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST:$ROUND_DAGGER_CAMPAIGN_CONTRACT" in
    /*:/*:/*:/*:/*:/*:/*:/*:/*:/*:/*:/*) ;;
    *) fatal "all production paths must be absolute" ;;
esac
[[ "$ROUND_DAGGER_CAMPAIGN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "unsafe campaign ID"
[[ "$ROUND_DAGGER_MACRO_ROUND" =~ ^(0|[1-9][0-9]*)$ ]] || fatal "macro round must be a canonical integer"
[[ "$ROUND_DAGGER_COHORT_INDEX" =~ ^[0-3]$ ]] || fatal "cohort index must be one of 0,1,2,3"
[[ "$ROUND_DAGGER_TARGET_ENVIRONMENTS:$ROUND_DAGGER_COHORT_COUNT:$ROUND_DAGGER_ENVIRONMENTS_PER_COHORT:$ROUND_DAGGER_STEPS_PER_ENVIRONMENT:$ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND" == \
   "256:4:64:512:131072" ]] || fatal "production topology must be exactly 256 environments x 512 steps"
[[ "$ROUND_DAGGER_LOGICAL_ENV_COUNT:$ROUND_DAGGER_EXPECTED_TRANSITIONS" == "64:32768" ]] || \
    fatal "cohort topology must be exactly 64 environments and 32768 transitions"
EXPECTED_OFFSET=$((10#$ROUND_DAGGER_COHORT_INDEX * 64))
[[ "$ROUND_DAGGER_GLOBAL_ENV_OFFSET" == "$EXPECTED_OFFSET" ]] || fatal "cohort/global-environment identity mismatch"
[[ "$ROUND_DAGGER_ACTOR_RUN_DIR" == "$TRAIN_RUN_DIR" ]] || fatal "actor run directory must equal TRAIN_RUN_DIR"
[[ $IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE == simulator_rgb ]] || \
    fatal "production bootstrap anchor source must be exactly simulator_rgb"
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || fatal "DRY_RUN must be exactly 0 or 1"

if [[ ${ROUND_DAGGER_TRUSTED_COLLECT_DISPATCH:-0} == 1 ]]; then
    [[ ${ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || \
        fatal "sealed actor source inventory SHA-256 is malformed"
    [[ ${GR00T_ROOT:-} == "$ROOT/GR00T-VisualSim2Real" ]] || \
        fatal "GR00T root is not derived from the sealed actor source"
    [[ ${ROUND_DAGGER_TEACHER_CHECKPOINT:-} == "$ROOT/"* ]] || \
        fatal "teacher checkpoint is not derived from the sealed actor source"
    [[ ${ROUND_DAGGER_ACTOR_PAIR_EXECUTOR:-} == \
        "$ROOT/scripts/round_dagger/run_actor_rank_docker.sh" ]] || \
        fatal "actor executor is not derived from the sealed actor source"
    [[ ${ROUND_DAGGER_ACTOR_PAIR_RANK_SCRIPT:-} == \
        "$ROOT/scripts/round_dagger/run_actor_cohort_with_wandb.sh" ]] || \
        fatal "actor rank primitive is not derived from the sealed actor source"
fi

for file in "$ROUND_DAGGER_TEACHER_CHECKPOINT" "$IMAGE3F_ENDPOINT_MANIFEST" "$ROUND_DAGGER_CAMERA_ACCEPTANCE"; do
    [ -f "$file" ] && [ ! -L "$file" ] || fatal "unsafe or missing immutable input: $file"
done
[[ -f $ROUND_DAGGER_CAMPAIGN_CONTRACT && ! -L $ROUND_DAGGER_CAMPAIGN_CONTRACT ]] || \
    fatal "external campaign contract is unsafe or missing"
[[ $ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
    fatal "external campaign contract SHA-256 is malformed"
[[ $ROUND_DAGGER_DATA_LINEAGE_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
    fatal "production data-lineage SHA-256 is malformed"
[[ $(sha256_file "$ROUND_DAGGER_CAMPAIGN_CONTRACT") == \
   $ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256 ]] || \
    fatal "external campaign contract bytes changed"
[[ $ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST == "$ROOT/$MOTION_SPLIT_RELATIVE" ]] || \
    fatal "collection motion split is not derived from the sealed actor source"
[[ -f $ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST && \
   ! -L $ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST ]] || \
    fatal "collection motion split manifest is unsafe or missing"
[[ $ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256 == \
   $EXPECTED_MOTION_SPLIT_MANIFEST_SHA256 ]] || \
    fatal "collection motion split identity is not the production split"
[[ $(sha256_file "$ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST") == \
   $EXPECTED_MOTION_SPLIT_MANIFEST_SHA256 ]] || \
    fatal "collection motion split manifest bytes changed"
[[ -f $ROUND_DAGGER_WANDB_NETRC && ! -L $ROUND_DAGGER_WANDB_NETRC && \
   $(stat -c %a "$ROUND_DAGGER_WANDB_NETRC") == 600 ]] || \
    fatal "W&B netrc must be a mode-0600 regular file"
grep -q '^machine api.wandb.ai$' "$ROUND_DAGGER_WANDB_NETRC" && \
grep -q '^  login user$' "$ROUND_DAGGER_WANDB_NETRC" && \
grep -q '^  password wandb_v1_' "$ROUND_DAGGER_WANDB_NETRC" || \
    fatal "W&B netrc has no supported API credential"
[ -d "$ANCHORS" ] && [ ! -L "$ANCHORS" ] || fatal "unsafe or missing anchor pool"
[ -x "$ROUND_DAGGER_DOCKER_BIN" ] && [ ! -L "$ROUND_DAGGER_DOCKER_BIN" ] || fatal "unsafe Docker executable"
[ -x "$PAIR_SCRIPT" ] && [ ! -L "$PAIR_SCRIPT" ] || fatal "unsafe production actor pair primitive"
[ -f "$ANCHOR_VERIFIER" ] && [ ! -L "$ANCHOR_VERIFIER" ] || fatal "unsafe anchor verifier"

[[ "$ROUND_DAGGER_TEACHER_SHA256" == "$EXPECTED_TEACHER_SHA256" ]] || fatal "round teacher identity is not the accepted production teacher"
[[ $(sha256_file "$ROUND_DAGGER_TEACHER_CHECKPOINT") == "$EXPECTED_TEACHER_SHA256" ]] || fatal "production teacher checkpoint bytes changed"
[[ $ROUND_DAGGER_RENDERER_SHA256 =~ ^[0-9a-f]{64}$ ]] || fatal "round renderer identity is malformed"
[[ $ROUND_DAGGER_ENDPOINT_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || fatal "endpoint manifest identity is malformed"
[[ $(sha256_file "$IMAGE3F_ENDPOINT_MANIFEST") == "$ROUND_DAGGER_ENDPOINT_MANIFEST_SHA256" ]] || fatal "renderer endpoint manifest bytes changed"
[[ "$ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || fatal "camera acceptance SHA-256 is malformed"
[[ $(sha256_file "$ROUND_DAGGER_CAMERA_ACCEPTANCE") == "$ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256" ]] || fatal "camera acceptance bytes changed"
[[ "$ROUND_DAGGER_GR00T_REVISION" =~ ^[0-9a-f]{40}$ ]] || fatal "GR00T revision is malformed"
[[ "$ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] || fatal "GR00T source tree SHA-256 is malformed"

# The acceptance is evidence, not merely an opaque file hash.  Require the
# exact three-view semantic mapping the operator accepted.
python3 - "$ROUND_DAGGER_CAMERA_ACCEPTANCE" "$ROUND_DAGGER_GR00T_REVISION" \
    "$ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256" <<'PY' || fatal "camera acceptance semantic contract is invalid"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_gr00t_revision = sys.argv[2]
expected_gr00t_tree = sys.argv[3]
try:
    document = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid camera acceptance JSON: {exc}")
expected = {
    "schema": "fast_roborender.camera_calibration_acceptance.v2",
    "status": "accepted",
    "policy_views": ["head_color", "hand_left", "hand_right"],
    "columns": ["head_color", "hand_left", "hand_right"],
    "rows": ["depth_control", "simulator_rgb", "generated_rgb"],
    "reviewer": "ldu@nvidia.com",
    "confirmation": "I reviewed both three-view calibration panels and accept the camera mapping.",
}
for key, value in expected.items():
    if document.get(key) != value:
        raise SystemExit(f"camera acceptance {key} mismatch")
if document.get("gr00t_revision") != expected_gr00t_revision:
    raise SystemExit("camera acceptance GR00T revision mismatch")
if document.get("gr00t_source_tree_sha256") != expected_gr00t_tree:
    raise SystemExit("camera acceptance GR00T source tree mismatch")
panels = document.get("panels")
if not isinstance(panels, list) or len(panels) != 2:
    raise SystemExit("camera acceptance must bind exactly two panels")
PY

INVARIANT_PROFILE="$ROOT/configs/round_dagger/production_campaign_256x512.json"
[[ -f $INVARIANT_PROFILE && ! -L $INVARIANT_PROFILE ]] || \
    fatal "sealed invariant profile is unsafe or missing"
RUNTIME_CONTRACT_REPORT=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    -m round_dagger.production_data_lineage verify \
    --invariant-profile "$INVARIANT_PROFILE" \
    --campaign-contract "$ROUND_DAGGER_CAMPAIGN_CONTRACT" \
    --campaign-contract-sha256 "$ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256") || \
    fatal "external campaign/live-capture contract verification failed"
RUNTIME_LINEAGE_SHA=$(python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data_lineage_sha256"])' \
    <<<"$RUNTIME_CONTRACT_REPORT") || fatal "cannot parse runtime data lineage"
[[ $RUNTIME_LINEAGE_SHA == "$ROUND_DAGGER_DATA_LINEAGE_SHA256" ]] || \
    fatal "runtime campaign lineage differs from durable cohort"

ANCHOR_REPORT=$(python3 "$ANCHOR_VERIFIER" \
    --root "$ANCHORS" \
    --expected-count 2048 \
    --expected-sha256 "$EXPECTED_ANCHOR_INVENTORY_SHA256") || fatal "anchor inventory is not the accepted production pool"

ACTUAL_IMAGE_ID=$(
    "$ROUND_DAGGER_DOCKER_BIN" image inspect "$ROUND_DAGGER_RTX_IMAGE" --format '{{.Id}}'
) || fatal "cannot inspect production actor image"
[[ "$ACTUAL_IMAGE_ID" == "$EXPECTED_ACTOR_IMAGE_ID" ]] || fatal "actor image ID mismatch"

printf '%s\n' "$ANCHOR_REPORT"
printf '{"schema":"fast_roborender.round_dagger.production_collect_preflight.v3","status":"verified","cohort_index":%s,"global_env_offset":%s,"teacher_sha256":"%s","renderer_sha256":"%s","endpoint_manifest_sha256":"%s","actor_image_id":"%s","camera_acceptance_sha256":"%s","gr00t_revision":"%s","gr00t_source_tree_sha256":"%s","motion_split_manifest_sha256":"%s","campaign_contract_sha256":"%s","data_lineage_sha256":"%s"}\n' \
    "$ROUND_DAGGER_COHORT_INDEX" "$ROUND_DAGGER_GLOBAL_ENV_OFFSET" \
    "$EXPECTED_TEACHER_SHA256" "$ROUND_DAGGER_RENDERER_SHA256" "$ROUND_DAGGER_ENDPOINT_MANIFEST_SHA256" \
    "$EXPECTED_ACTOR_IMAGE_ID" "$ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256" \
    "$ROUND_DAGGER_GR00T_REVISION" "$ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256" \
    "$ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256" \
    "$ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256" "$ROUND_DAGGER_DATA_LINEAGE_SHA256"

export NETRC="$ROUND_DAGGER_WANDB_NETRC"
exec "$PAIR_SCRIPT"
