#!/usr/bin/env bash
# Candidate-aware, immutable production paired-evaluation hook.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }
require_regular() {
    local path=$1 label=$2
    [[ $path == /* && -f $path && ! -L $path ]] || \
        fatal "$label must be an absolute, non-symlink regular file"
}
require_directory() {
    local path=$1 label=$2
    [[ $path == /* && -d $path && ! -L $path ]] || \
        fatal "$label must be an absolute, non-symlink directory"
}
require_sha() {
    local value=$1 label=$2
    [[ $value =~ ^[0-9a-f]{64}$ ]] || fatal "$label must be a lowercase SHA-256"
}
print_command() {
    printf 'DRY_RUN'
    printf ' %q' "$@"
    printf '\n'
}

# Production execution is permitted only through the coordinator-resident
# dispatcher.  That dispatcher verifies the complete seal, latches this script
# on an O_NOFOLLOW descriptor, and executes /proc/self/fd/N.  BASH_SOURCE is
# therefore intentionally not a repository pathname; derive every source path
# from the independently pinned and verified evaluation root.
TRUSTED_DISPATCH=${ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH:?missing ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH}
[[ $TRUSTED_DISPATCH == 1 ]] || \
    fatal "ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH must be exactly 1"
EVALUATION_SOURCE_ROOT=${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:?missing ROUND_DAGGER_EVALUATION_SOURCE_ROOT}
require_directory "$EVALUATION_SOURCE_ROOT" "sealed evaluation source root"
REPOSITORY_ROOT=$(cd "$EVALUATION_SOURCE_ROOT" && pwd -P)
[[ $REPOSITORY_ROOT == "$EVALUATION_SOURCE_ROOT" ]] || \
    fatal "sealed evaluation source root must be canonical"

# Dynamic identities are provided by MacroRoundOrchestrator.  They are checked
# here again so a standalone or stale hook invocation cannot evaluate a
# different candidate or publish outside the canonical round directory.
CAMPAIGN_ROOT=${ROUND_DAGGER_CAMPAIGN_ROOT:?missing ROUND_DAGGER_CAMPAIGN_ROOT}
CAMPAIGN_ID=${ROUND_DAGGER_CAMPAIGN_ID:?missing ROUND_DAGGER_CAMPAIGN_ID}
MACRO_ROUND=${ROUND_DAGGER_MACRO_ROUND:?missing ROUND_DAGGER_MACRO_ROUND}
PARENT_CHECKPOINT=${ROUND_DAGGER_POLICY_CHECKPOINT:?missing ROUND_DAGGER_POLICY_CHECKPOINT}
PARENT_CHECKPOINT_SHA=${ROUND_DAGGER_POLICY_SHA256:?missing ROUND_DAGGER_POLICY_SHA256}
PARENT_RUNTIME_SHA=${ROUND_DAGGER_RUNTIME_POLICY_SHA256:?missing ROUND_DAGGER_RUNTIME_POLICY_SHA256}
CANDIDATE_MANIFEST=${ROUND_DAGGER_CANDIDATE_MANIFEST:?missing ROUND_DAGGER_CANDIDATE_MANIFEST}
CANDIDATE_MANIFEST_SHA=${ROUND_DAGGER_CANDIDATE_SHA256:?missing ROUND_DAGGER_CANDIDATE_SHA256}
EVALUATOR_CONFIG_SHA=${ROUND_DAGGER_EVALUATOR_CONFIG_SHA256:?missing ROUND_DAGGER_EVALUATOR_CONFIG_SHA256}
EVALUATOR_RESULT=${ROUND_DAGGER_EVALUATOR_RESULT:?missing ROUND_DAGGER_EVALUATOR_RESULT}

# Static deployment identities are deliberately explicit rather than inherited
# from an interactive shell.
EVALUATION_CONFIG=${ROUND_DAGGER_EVALUATION_CONFIG:?missing ROUND_DAGGER_EVALUATION_CONFIG}
SCENARIO_CASES=${ROUND_DAGGER_EVALUATION_SCENARIOS:?missing ROUND_DAGGER_EVALUATION_SCENARIOS}
SCENARIO_CASES_SHA=${ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256:?missing ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256}
THRESHOLDS=${ROUND_DAGGER_EVALUATION_THRESHOLDS:?missing ROUND_DAGGER_EVALUATION_THRESHOLDS}
THRESHOLDS_SHA=${ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256:?missing ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256}
MOTION_SPLIT=${ROUND_DAGGER_EVALUATION_MOTION_SPLIT_MANIFEST:?missing ROUND_DAGGER_EVALUATION_MOTION_SPLIT_MANIFEST}
MOTION_SPLIT_SHA=${ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256:?missing ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256}
HELDOUT_PLAN=${ROUND_DAGGER_HELDOUT_EVALUATION_PLAN:?missing ROUND_DAGGER_HELDOUT_EVALUATION_PLAN}
HELDOUT_PLAN_SHA=${ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256:?missing ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256}
TEACHER_CHECKPOINT=${ROUND_DAGGER_TEACHER_CHECKPOINT:?missing ROUND_DAGGER_TEACHER_CHECKPOINT}
TEACHER_SHA=${ROUND_DAGGER_TEACHER_SHA256:?missing ROUND_DAGGER_TEACHER_SHA256}
SEALED_TEACHER=${ROUND_DAGGER_EVALUATION_SEALED_TEACHER:?missing ROUND_DAGGER_EVALUATION_SEALED_TEACHER}
EVALUATION_SOURCE_INVENTORY_SHA=${ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256:?missing ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256}
ENDPOINT_DIR=${ROUND_DAGGER_EVALUATION_ENDPOINT_DIR:?missing ROUND_DAGGER_EVALUATION_ENDPOINT_DIR}
RENDERER_RUNTIME_SHA=${ROUND_DAGGER_RENDERER_RUNTIME_SHA256:?missing ROUND_DAGGER_RENDERER_RUNTIME_SHA256}
EXPECTED_RENDERERS=${ROUND_DAGGER_EXPECTED_RENDERERS:?missing ROUND_DAGGER_EXPECTED_RENDERERS}
CONTAINER_PYTHON=${GR00T_PYTHON:?missing GR00T_PYTHON}
DOCKER=${ROUND_DAGGER_DOCKER_BIN:?missing ROUND_DAGGER_DOCKER_BIN}
IMAGE=${ROUND_DAGGER_RTX_IMAGE:?missing ROUND_DAGGER_RTX_IMAGE}
EXPECTED_IMAGE_ID_SHA=${ROUND_DAGGER_RTX_IMAGE_ID_SHA256:?missing ROUND_DAGGER_RTX_IMAGE_ID_SHA256}
CONTROL_PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:-python3}
DRY_RUN=${DRY_RUN:-0}
RECOVERY_ID=${ROUND_DAGGER_EVALUATION_RECOVERY_ID:-}
ANCHOR_POOL=${ANCHORS:?missing ANCHORS}
ANCHOR_COUNT=${ROUND_DAGGER_MIN_ANCHORS:?missing ROUND_DAGGER_MIN_ANCHORS}
ANCHOR_INDEX=${ROUND_DAGGER_ANCHOR_INDEX:?missing ROUND_DAGGER_ANCHOR_INDEX}
ANCHOR_INVENTORY_SHA=${ROUND_DAGGER_ANCHOR_INVENTORY_SHA256:?missing ROUND_DAGGER_ANCHOR_INVENTORY_SHA256}
ANCHOR_SOURCE=${IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE:?missing IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE}
CAMPAIGN_CONTRACT=${ROUND_DAGGER_CAMPAIGN_CONTRACT:?missing ROUND_DAGGER_CAMPAIGN_CONTRACT}
CAMPAIGN_CONTRACT_SHA=${ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256:?missing ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256}
DATA_LINEAGE_SHA=${ROUND_DAGGER_DATA_LINEAGE_SHA256:?missing ROUND_DAGGER_DATA_LINEAGE_SHA256}
CAPTURE_ATTESTATION=${ROUND_DAGGER_CAPTURE_ATTESTATION:?missing ROUND_DAGGER_CAPTURE_ATTESTATION}
CAPTURE_ATTESTATION_SHA=${ROUND_DAGGER_CAPTURE_ATTESTATION_SHA256:?missing ROUND_DAGGER_CAPTURE_ATTESTATION_SHA256}
CAPTURE_EVIDENCE=${ROUND_DAGGER_CAPTURE_EVIDENCE:?missing ROUND_DAGGER_CAPTURE_EVIDENCE}
CAPTURE_EVIDENCE_SHA=${ROUND_DAGGER_CAPTURE_EVIDENCE_SHA256:?missing ROUND_DAGGER_CAPTURE_EVIDENCE_SHA256}
FIRST_CONDITION_BUNDLE_INVENTORY_SHA=${ROUND_DAGGER_FIRST_CONDITION_BUNDLE_INVENTORY_SHA256:?missing ROUND_DAGGER_FIRST_CONDITION_BUNDLE_INVENTORY_SHA256}

[[ $CAMPAIGN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
    fatal "campaign ID is unsafe"
[[ $MACRO_ROUND =~ ^(0|[1-9][0-9]*)$ ]] || fatal "macro round is non-canonical"
[[ $EXPECTED_RENDERERS =~ ^[1-9][0-9]*$ ]] || fatal "expected renderer count is invalid"
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || fatal "DRY_RUN must be exactly 0 or 1"
if [[ -n $RECOVERY_ID ]]; then
    [[ $RECOVERY_ID =~ ^hardened-[0-9a-f]{20}$ ]] || \
        fatal "ROUND_DAGGER_EVALUATION_RECOVERY_ID is non-canonical"
fi
require_sha "$PARENT_CHECKPOINT_SHA" ROUND_DAGGER_POLICY_SHA256
require_sha "$PARENT_RUNTIME_SHA" ROUND_DAGGER_RUNTIME_POLICY_SHA256
require_sha "$CANDIDATE_MANIFEST_SHA" ROUND_DAGGER_CANDIDATE_SHA256
require_sha "$EVALUATOR_CONFIG_SHA" ROUND_DAGGER_EVALUATOR_CONFIG_SHA256
require_sha "$SCENARIO_CASES_SHA" ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256
require_sha "$THRESHOLDS_SHA" ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256
require_sha "$MOTION_SPLIT_SHA" ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256
require_sha "$HELDOUT_PLAN_SHA" ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256
require_sha "$TEACHER_SHA" ROUND_DAGGER_TEACHER_SHA256
require_sha "$EVALUATION_SOURCE_INVENTORY_SHA" ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256
require_sha "$ANCHOR_INVENTORY_SHA" ROUND_DAGGER_ANCHOR_INVENTORY_SHA256
require_sha "$RENDERER_RUNTIME_SHA" ROUND_DAGGER_RENDERER_RUNTIME_SHA256
require_sha "$EXPECTED_IMAGE_ID_SHA" ROUND_DAGGER_RTX_IMAGE_ID_SHA256
require_sha "$CAMPAIGN_CONTRACT_SHA" ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256
require_sha "$DATA_LINEAGE_SHA" ROUND_DAGGER_DATA_LINEAGE_SHA256
require_sha "$CAPTURE_ATTESTATION_SHA" ROUND_DAGGER_CAPTURE_ATTESTATION_SHA256
require_sha "$CAPTURE_EVIDENCE_SHA" ROUND_DAGGER_CAPTURE_EVIDENCE_SHA256
require_sha "$FIRST_CONDITION_BUNDLE_INVENTORY_SHA" \
    ROUND_DAGGER_FIRST_CONDITION_BUNDLE_INVENTORY_SHA256
require_directory "$CAMPAIGN_ROOT" "campaign root"
[[ $REPOSITORY_ROOT == "$EVALUATION_SOURCE_ROOT" ]] || \
    fatal "production evaluation hook is not executing from the sealed source root"
GR00T_SOURCE_ROOT="$EVALUATION_SOURCE_ROOT/GR00T-VisualSim2Real"
require_directory "$GR00T_SOURCE_ROOT" "sealed GR00T root"
require_directory "$ENDPOINT_DIR" "renderer endpoint directory"
require_directory "$ANCHOR_POOL" "anchor pool"
[[ $ANCHOR_COUNT =~ ^[1-9][0-9]*$ ]] || fatal "anchor count is invalid"
[[ $ANCHOR_INDEX =~ ^(0|[1-9][0-9]*)$ ]] || fatal "anchor index is invalid"
[[ $ANCHOR_SOURCE == simulator_rgb ]] || \
    fatal "production bootstrap anchor source must be exactly simulator_rgb"
((10#$ANCHOR_INDEX < 10#$ANCHOR_COUNT)) || \
    fatal "anchor index must be smaller than anchor count"
require_regular "$PARENT_CHECKPOINT" "parent checkpoint"
require_regular "$CANDIDATE_MANIFEST" "candidate manifest"
require_regular "$EVALUATION_CONFIG" "evaluation config"
require_regular "$SCENARIO_CASES" "scenario cases"
require_regular "$THRESHOLDS" "evaluation thresholds"
require_regular "$MOTION_SPLIT" "evaluation motion split manifest"
require_regular "$HELDOUT_PLAN" "held-out evaluation plan"
require_regular "$TEACHER_CHECKPOINT" "configured teacher checkpoint"
require_regular "$SEALED_TEACHER" "sealed teacher checkpoint"
require_regular "$CAMPAIGN_CONTRACT" "external campaign contract"
require_regular "$CAPTURE_ATTESTATION" "live capture attestation"
require_regular "$CAPTURE_EVIDENCE" "live capture evidence"
[[ $TEACHER_CHECKPOINT != "$SEALED_TEACHER" ]] || \
    fatal "sealed teacher source must differ from the configured target"
[[ $(stat -c %a "$SEALED_TEACHER") == 444 ]] || \
    fatal "sealed teacher checkpoint mode must be exactly 0444"
[[ $(stat -c %h "$SEALED_TEACHER") == 1 ]] || \
    fatal "sealed teacher checkpoint must have exactly one hard link"
[[ $(stat -c '%d:%i' "$TEACHER_CHECKPOINT") != \
   "$(stat -c '%d:%i' "$SEALED_TEACHER")" ]] || \
    fatal "sealed teacher source and configured target must be distinct files"
[[ $CONTAINER_PYTHON == /* ]] || fatal "GR00T_PYTHON must be an absolute container path"
case "$DOCKER" in /*) ;; *) fatal "Docker executable must be absolute" ;; esac
[[ -x $DOCKER && ! -L $DOCKER ]] || fatal "Docker executable is unsafe or unavailable"
if [[ $CONTROL_PYTHON == */* ]]; then
    [[ $CONTROL_PYTHON == /* ]] || fatal "control Python path must be absolute"
else
    CONTROL_PYTHON=$(command -v "$CONTROL_PYTHON") || fatal "control Python is unavailable"
fi
# A virtual-environment interpreter is normally a symlink.  Validate its
# ultimate executable, but retain the venv entry point when invoking Python;
# replacing it with the real target discards the venv's site-packages.
CONTROL_PYTHON_TARGET=$(readlink -f -- "$CONTROL_PYTHON") || \
    fatal "cannot resolve control Python executable"
[[ $CONTROL_PYTHON == /* && -x $CONTROL_PYTHON && \
   $CONTROL_PYTHON_TARGET == /* && -f $CONTROL_PYTHON_TARGET && \
   ! -L $CONTROL_PYTHON_TARGET && -x $CONTROL_PYTHON_TARGET ]] || \
    fatal "control Python executable is unsafe or unavailable"

[[ $(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}') == "$PARENT_CHECKPOINT_SHA" ]] || \
    fatal "parent checkpoint SHA mismatch"
[[ $(sha256sum "$CANDIDATE_MANIFEST" | awk '{print $1}') == "$CANDIDATE_MANIFEST_SHA" ]] || \
    fatal "candidate manifest SHA mismatch"
[[ $(sha256sum "$EVALUATION_CONFIG" | awk '{print $1}') == "$EVALUATOR_CONFIG_SHA" ]] || \
    fatal "evaluation config SHA mismatch"
[[ $(sha256sum "$SCENARIO_CASES" | awk '{print $1}') == "$SCENARIO_CASES_SHA" ]] || \
    fatal "evaluation scenario cases SHA mismatch"
[[ $(sha256sum "$THRESHOLDS" | awk '{print $1}') == "$THRESHOLDS_SHA" ]] || \
    fatal "evaluation thresholds SHA mismatch"
[[ $(sha256sum "$MOTION_SPLIT" | awk '{print $1}') == "$MOTION_SPLIT_SHA" ]] || \
    fatal "evaluation motion split manifest SHA mismatch"
[[ $(sha256sum "$HELDOUT_PLAN" | awk '{print $1}') == "$HELDOUT_PLAN_SHA" ]] || \
    fatal "held-out evaluation plan SHA mismatch"
[[ $(sha256sum "$CAMPAIGN_CONTRACT" | awk '{print $1}') == "$CAMPAIGN_CONTRACT_SHA" ]] || \
    fatal "external campaign contract SHA mismatch"
[[ $(sha256sum "$CAPTURE_ATTESTATION" | awk '{print $1}') == "$CAPTURE_ATTESTATION_SHA" ]] || \
    fatal "live capture attestation SHA mismatch"
[[ $(sha256sum "$CAPTURE_EVIDENCE" | awk '{print $1}') == "$CAPTURE_EVIDENCE_SHA" ]] || \
    fatal "live capture evidence SHA mismatch"
[[ $MOTION_SPLIT == \
   "$GR00T_SOURCE_ROOT/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775_heldout_256_v1.json" ]] || \
    fatal "evaluation motion split path is non-canonical"
[[ $HELDOUT_PLAN == \
   "$EVALUATION_SOURCE_ROOT/configs/round_dagger/production_heldout_motion_plan_8x32.json" ]] || \
    fatal "held-out evaluation plan path is non-canonical"
[[ $(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}') == "$TEACHER_SHA" ]] || \
    fatal "configured teacher checkpoint SHA mismatch"
[[ $(sha256sum "$SEALED_TEACHER" | awk '{print $1}') == "$TEACHER_SHA" ]] || \
    fatal "sealed teacher checkpoint SHA mismatch"
EXPECTED_SEALED_TEACHER="$EVALUATION_SOURCE_ROOT/${ROUND_DAGGER_TEACHER_CHECKPOINT_RELATIVE:?missing ROUND_DAGGER_TEACHER_CHECKPOINT_RELATIVE}"
[[ $SEALED_TEACHER == "$EXPECTED_SEALED_TEACHER" ]] || \
    fatal "sealed teacher path was not derived from the evaluation source"

SOURCE_INVENTORY="$(dirname "$EVALUATION_SOURCE_ROOT")/$(basename "$EVALUATION_SOURCE_ROOT").inventory.json"
SOURCE_ATTESTATION="$(dirname "$EVALUATION_SOURCE_ROOT")/$(basename "$EVALUATION_SOURCE_ROOT").snapshot_attestation.json"
require_regular "$SOURCE_INVENTORY" "evaluation source inventory"
require_regular "$SOURCE_ATTESTATION" "evaluation source attestation"
export PYTHONPATH="$EVALUATION_SOURCE_ROOT:$GR00T_SOURCE_ROOT"

verify_evaluation_source() {
    "$CONTROL_PYTHON" "$EVALUATION_SOURCE_ROOT/scripts/round_dagger/verify_sealed_source_snapshot.py" \
        --root "$EVALUATION_SOURCE_ROOT" \
        --expected-inventory-sha256 "$EVALUATION_SOURCE_INVENTORY_SHA" \
        >/dev/null || fatal "sealed evaluation source verification failed"
}
verify_anchor_pool() {
    "$CONTROL_PYTHON" "$EVALUATION_SOURCE_ROOT/scripts/round_dagger/verify_anchor_pool.py" \
        --root "$ANCHOR_POOL" --expected-count "$ANCHOR_COUNT" \
        --expected-sha256 "$ANCHOR_INVENTORY_SHA" \
        >/dev/null || fatal "evaluation anchor pool verification failed"
}
verify_evaluation_source
verify_anchor_pool
MOTION_SPLIT_REPORT=$(
    "$CONTROL_PYTHON" -m round_dagger.motion_split_contract verify \
        --manifest "$MOTION_SPLIT" --manifest-sha256 "$MOTION_SPLIT_SHA" \
        --plan "$HELDOUT_PLAN" --plan-sha256 "$HELDOUT_PLAN_SHA" \
        --scenarios "$SCENARIO_CASES"
) || fatal "motion split/held-out evaluation contract verification failed"
printf '%s\n' "$MOTION_SPLIT_REPORT"

# The pinned Hydra config must name the host path that will be shadowed by the
# more-specific read-only file bind below.  Parsing the exact, digest-checked
# config prevents a valid teacher from being mounted at a path the evaluator
# never consumes.
mapfile -t CONFIGURED_INPUTS < <($CONTROL_PYTHON - "$EVALUATION_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    document = yaml.safe_load(handle)
if not isinstance(document, dict):
    raise SystemExit("evaluation config must contain a YAML mapping")
teacher = document.get("teacher_actor_path")
if not isinstance(teacher, str) or not Path(teacher).is_absolute():
    raise SystemExit("evaluation config teacher_actor_path must be absolute")
videogen = document.get("algo", {}).get("config", {}).get("videogen", {})
anchor = videogen.get("anchor_pool_dir")
index = videogen.get("anchor_index")
if not isinstance(anchor, str) or not Path(anchor).is_absolute():
    raise SystemExit("evaluation config anchor_pool_dir must be absolute")
if isinstance(index, bool) or not isinstance(index, int) or index < 0:
    raise SystemExit("evaluation config anchor_index must be non-negative")
print(teacher)
print(anchor)
print(index)
PY
) || fatal "cannot read teacher/anchor paths from the pinned evaluation config"
[[ ${#CONFIGURED_INPUTS[@]} -eq 3 ]] || fatal "pinned config input parse is incomplete"
CONFIGURED_TEACHER=${CONFIGURED_INPUTS[0]}
CONFIGURED_ANCHOR_POOL=${CONFIGURED_INPUTS[1]}
CONFIGURED_ANCHOR_INDEX=${CONFIGURED_INPUTS[2]}
[[ $CONFIGURED_TEACHER == "$TEACHER_CHECKPOINT" ]] || \
    fatal "pinned evaluation config teacher path mismatch"
[[ $CONFIGURED_ANCHOR_POOL == "$ANCHOR_POOL" ]] || \
    fatal "pinned evaluation config anchor pool path mismatch"
[[ $CONFIGURED_ANCHOR_INDEX == "$ANCHOR_INDEX" ]] || \
    fatal "pinned evaluation config anchor index mismatch"

SEALED_CAMPAIGN_CONTRACT="$EVALUATION_SOURCE_ROOT/configs/round_dagger/production_campaign_256x512.json"
require_regular "$SEALED_CAMPAIGN_CONTRACT" "sealed production campaign contract"
RUNTIME_CONTRACT_REPORT=$(
    "$CONTROL_PYTHON" -m round_dagger.production_data_lineage verify \
        --invariant-profile "$SEALED_CAMPAIGN_CONTRACT" \
        --campaign-contract "$CAMPAIGN_CONTRACT" \
        --campaign-contract-sha256 "$CAMPAIGN_CONTRACT_SHA"
) || fatal "external campaign/live-capture contract verification failed"
RUNTIME_LINEAGE_SHA=$(
    "$CONTROL_PYTHON" -c \
        'import json,sys; print(json.load(sys.stdin)["data_lineage_sha256"])' \
        <<<"$RUNTIME_CONTRACT_REPORT"
) || fatal "cannot parse runtime campaign lineage identity"
[[ $RUNTIME_LINEAGE_SHA == "$DATA_LINEAGE_SHA" ]] || \
    fatal "runtime campaign lineage differs from durable round"
verify_first_condition_bundles() {
    "$CONTROL_PYTHON" - "$CAMPAIGN_CONTRACT" "$CAMPAIGN_CONTRACT_SHA" \
        "$SCENARIO_CASES" "$SCENARIO_CASES_SHA" \
        "$CAPTURE_ATTESTATION" "$CAPTURE_ATTESTATION_SHA" \
        "$CAPTURE_EVIDENCE" "$CAPTURE_EVIDENCE_SHA" \
        "$FIRST_CONDITION_BUNDLE_INVENTORY_SHA" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


runtime_path = Path(sys.argv[1])
if digest(runtime_path) != sys.argv[2]:
    raise SystemExit("production runtime contract SHA mismatch")
with runtime_path.open(encoding="utf-8") as handle:
    capture = json.load(handle)["live_capture"]
expected = {
    "scenario_path": sys.argv[3],
    "scenario_sha256": sys.argv[4],
    "attestation_path": sys.argv[5],
    "attestation_sha256": sys.argv[6],
    "evidence_path": sys.argv[7],
    "evidence_sha256": sys.argv[8],
    "first_condition_bundle_inventory_sha256": sys.argv[9],
}
if any(capture.get(key) != value for key, value in expected.items()):
    raise SystemExit("evaluator inputs differ from production runtime contract")
rows = capture.get("first_condition_bundles")
keys = {"seed", "path", "sha256", "size", "case_count", "source_pass_index"}
seeds = [17, 29, 43, 59, 71, 89, 101, 127]
if not isinstance(rows, list) or len(rows) != len(seeds):
    raise SystemExit("first-condition bundle coverage is not exact 8-seed")
if hashlib.sha256(json.dumps(
    rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("utf-8")).hexdigest() != sys.argv[9]:
    raise SystemExit("first-condition bundle inventory SHA mismatch")
for index, (row, seed) in enumerate(zip(rows, seeds, strict=True)):
    if not isinstance(row, dict) or set(row) != keys:
        raise SystemExit(f"first-condition bundle row {index} fields differ")
    if (row["seed"] != seed or row["case_count"] != 32
            or row["source_pass_index"] != 1
            or isinstance(row["size"], bool) or not isinstance(row["size"], int)
            or row["size"] < 1):
        raise SystemExit(f"first-condition bundle row {index} is malformed")
    path = Path(row["path"])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.resolve(strict=True) != path):
        raise SystemExit(f"first-condition bundle {seed} path is unsafe")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit(f"first-condition bundle {seed} mode is not 0444")
    if metadata.st_size != row["size"] or digest(path) != row["sha256"]:
        raise SystemExit(f"first-condition bundle {seed} bytes differ")
for label, path_value, sha_value in (
    ("capture attestation", sys.argv[5], sys.argv[6]),
    ("capture evidence", sys.argv[7], sys.argv[8]),
):
    path = Path(path_value)
    if digest(path) != sha_value:
        raise SystemExit(f"{label} SHA mismatch")
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if (document.get("first_condition_bundles") != rows
            or document.get("first_condition_bundle_inventory_sha256")
            != sys.argv[9]):
        raise SystemExit(f"{label} bundle inventory differs")
for row in rows:
    print(row["path"])
PY
}
BUNDLE_PATH_OUTPUT=$(verify_first_condition_bundles) || \
    fatal "first-condition bundle inventory verification failed"
mapfile -t FIRST_CONDITION_BUNDLE_PATHS <<<"$BUNDLE_PATH_OUTPUT"
[[ ${#FIRST_CONDITION_BUNDLE_PATHS[@]} -eq 8 ]] || \
    fatal "first-condition bundle mount inventory is incomplete"
printf '%s\n' "$RUNTIME_CONTRACT_REPORT"
$CONTROL_PYTHON - "$SEALED_CAMPAIGN_CONTRACT" "$ANCHOR_POOL" \
    "$ANCHOR_COUNT" "$ANCHOR_INVENTORY_SHA" "$ANCHOR_SOURCE" <<'PY' || \
    fatal "sealed campaign anchor contract mismatch"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    campaign = json.load(handle)
anchor = campaign["artifacts"]["anchor_pool"]
expected = {
    "path": sys.argv[2],
    "anchor_count": int(sys.argv[3]),
    "inventory_sha256": sys.argv[4],
    "inventory_algorithm": "len64be(path_utf8)+path_utf8+size64be+sha256_binary",
    "usage": "compatibility_fallback_only",
}
if (
    anchor != expected
    or campaign["observation_contract"]["bootstrap_anchor_source"] != sys.argv[5]
):
    raise SystemExit(1)
PY

PADDED_ROUND=$(printf '%06d' "$MACRO_ROUND")
if [[ -n $RECOVERY_ID ]]; then
    EVALUATION_ROOT="$CAMPAIGN_ROOT/rounds/macro_round_$PADDED_ROUND/evaluation_recovery/$RECOVERY_ID"
else
    EVALUATION_ROOT="$CAMPAIGN_ROOT/rounds/macro_round_$PADDED_ROUND/evaluation"
fi
CONTRACT="$EVALUATION_ROOT/contract.json"
RESULTS_DIR="$EVALUATION_ROOT/results"
EXPECTED_RESULT="$EVALUATION_ROOT/result.json"
[[ $EVALUATOR_RESULT == "$EXPECTED_RESULT" ]] || \
    fatal "orchestrator evaluator result path is non-canonical"

# Every host input must be representable by an exact narrow Docker bind.  The
# source, evidence, config, endpoint, anchor, and teacher binds are read-only;
# only the campaign root is writable for staged inputs and evaluation results.
for path in "$REPOSITORY_ROOT" "$CAMPAIGN_ROOT" "$GR00T_SOURCE_ROOT" \
    "$EVALUATION_CONFIG" "$SCENARIO_CASES" "$THRESHOLDS" "$ENDPOINT_DIR" \
    "$MOTION_SPLIT" "$HELDOUT_PLAN" \
    "$TEACHER_CHECKPOINT" "$SEALED_TEACHER" "$ANCHOR_POOL" \
    "$SOURCE_INVENTORY" "$SOURCE_ATTESTATION" "$CAMPAIGN_CONTRACT" \
    "$CAPTURE_ATTESTATION" "$CAPTURE_EVIDENCE" \
    "${FIRST_CONDITION_BUNDLE_PATHS[@]}"; do
    case "$path" in
        /mnt/data1|/mnt/data1/*|/mnt/data2|/mnt/data2/*) ;;
        *) fatal "container input is outside /mnt/data1 and /mnt/data2: $path" ;;
    esac
done
[[ $THRESHOLDS == "$EVALUATION_SOURCE_ROOT"/* ]] || \
    fatal "threshold contract must come from the sealed source"

mapfile -t ENDPOINT_FILES < <(find "$ENDPOINT_DIR" -maxdepth 1 -type f \
    -name 'renderer_*.txt' -print | LC_ALL=C sort)
[[ ${#ENDPOINT_FILES[@]} -eq $EXPECTED_RENDERERS ]] || \
    fatal "renderer endpoint count differs from ROUND_DAGGER_EXPECTED_RENDERERS"

ACTUAL_IMAGE_ID=$("$DOCKER" image inspect --format '{{.Id}}' "$IMAGE") || \
    fatal "actor Docker image is unavailable"
[[ $ACTUAL_IMAGE_ID == "sha256:$EXPECTED_IMAGE_ID_SHA" ]] || \
    fatal "actor Docker image ID mismatch"

# The preflight uses the same strict campaign/candidate loaders and schema
# validators as contract publication, but performs no write.  Its sole output
# is the candidate runtime ABI hash needed by the immutable contract.
CANDIDATE_RUNTIME_SHA=$($CONTROL_PYTHON - \
    "$CAMPAIGN_ROOT" "$CAMPAIGN_ID" "$MACRO_ROUND" "$CANDIDATE_MANIFEST" \
    "$PARENT_CHECKPOINT" "$PARENT_RUNTIME_SHA" "$TEACHER_SHA" \
    "$SCENARIO_CASES" "$THRESHOLDS" <<'PY'
import sys
from pathlib import Path

from round_dagger.artifacts import load_strict_json
from round_dagger.evaluation import _validate_cases, _validate_thresholds
from round_dagger.promotion import load_candidate_manifest
from round_dagger.state import load_campaign, load_state

root = Path(sys.argv[1])
campaign = load_campaign(root)
state = load_state(root)
if campaign["campaign_id"] != sys.argv[2]:
    raise SystemExit("campaign ID differs from coordinator")
if campaign["teacher_sha256"] != sys.argv[7]:
    raise SystemExit("teacher checkpoint SHA differs from campaign")
macro_round = int(sys.argv[3])
if state.get("phase") != "TRAINED" or state.get("open_macro_round") != macro_round:
    raise SystemExit("coordinator is not in the requested TRAINED round")
candidate = load_candidate_manifest(
    root, sys.argv[4], expected_macro_round=macro_round, bind_state=True
)
active = state["active_policy"]
if str(root / active["checkpoint_path"]) != sys.argv[5]:
    raise SystemExit("parent checkpoint path differs from active policy")
if active["runtime_policy_sha256"] != sys.argv[6]:
    raise SystemExit("parent runtime ABI differs from active policy")
cases = _validate_cases(load_strict_json(sys.argv[8]))
if cases != load_strict_json(sys.argv[8]):
    raise SystemExit("scenario cases are not canonically ordered")
_validate_thresholds(load_strict_json(sys.argv[9]))
print(candidate["runtime_policy_sha256"])
PY
) || fatal "campaign/candidate/evaluation preflight failed"
require_sha "$CANDIDATE_RUNTIME_SHA" "candidate runtime policy identity"

EVALUATOR_DIGEST_JSON=$($CONTROL_PYTHON -m round_dagger.evaluation evaluator-digest \
    --repository-root "$REPOSITORY_ROOT" --gr00t-root "$GR00T_SOURCE_ROOT" \
    --evaluation-source-inventory-sha256 "$EVALUATION_SOURCE_INVENTORY_SHA") || \
    fatal "evaluator source digest failed"
EVALUATOR_CODE_SHA=$(printf '%s' "$EVALUATOR_DIGEST_JSON" | \
    $CONTROL_PYTHON -c 'import json,sys; print(json.load(sys.stdin)["sha256"])') || \
    fatal "cannot read evaluator source digest"
require_sha "$EVALUATOR_CODE_SHA" "evaluator source bundle identity"

PREPARE=("$CONTROL_PYTHON" -m round_dagger.evaluation prepare
    --root "$CAMPAIGN_ROOT"
    --candidate-manifest "$CANDIDATE_MANIFEST"
    --scenario-cases "$SCENARIO_CASES"
    --parent-runtime-policy-sha256 "$PARENT_RUNTIME_SHA"
    --candidate-runtime-policy-sha256 "$CANDIDATE_RUNTIME_SHA"
    --renderer-runtime-sha256 "$RENDERER_RUNTIME_SHA"
    --evaluation-config "$EVALUATION_CONFIG"
    --evaluator-code "$EVALUATOR_CODE_SHA"
    --expected-renderer-count "$EXPECTED_RENDERERS"
    --anchor-pool-path "$ANCHOR_POOL"
    --anchor-count "$ANCHOR_COUNT"
    --anchor-index "$ANCHOR_INDEX"
    --anchor-inventory-sha256 "$ANCHOR_INVENTORY_SHA"
    --production-runtime-contract "$CAMPAIGN_CONTRACT"
    --production-runtime-contract-sha256 "$CAMPAIGN_CONTRACT_SHA"
    --thresholds "$THRESHOLDS"
    --output "$CONTRACT")

RUNNER="$REPOSITORY_ROOT/scripts/round_dagger/run_paired_evaluation.sh"
require_regular "$RUNNER" "paired evaluation launcher"
CONTAINER_NAME="rde-${CAMPAIGN_ID}-m${MACRO_ROUND}"
ENV_ARGS=()
while IFS= read -r name; do
    case "$name" in
        ROUND_DAGGER_*|IMAGE3F_*|IMAGE_DAGGER_*|ROBORENDER_*|GR00T_PYTHON)
            ENV_ARGS+=(--env "$name")
            ;;
    esac
done < <(compgen -e | LC_ALL=C sort)
FIRST_CONDITION_BUNDLE_MOUNTS=()
for path in "${FIRST_CONDITION_BUNDLE_PATHS[@]}"; do
    FIRST_CONDITION_BUNDLE_MOUNTS+=(
        --mount "type=bind,src=$path,dst=$path,readonly"
    )
done
DOCKER_RUN=("$DOCKER" run --rm
    --name "$CONTAINER_NAME"
    --gpus '"device=0"'
    --ipc=private
    --shm-size=32g
    --network=host
    --workdir "$GR00T_SOURCE_ROOT"
    --mount "type=bind,src=$EVALUATION_SOURCE_ROOT,dst=$EVALUATION_SOURCE_ROOT,readonly"
    --mount "type=bind,src=$SOURCE_INVENTORY,dst=$SOURCE_INVENTORY,readonly"
    --mount "type=bind,src=$SOURCE_ATTESTATION,dst=$SOURCE_ATTESTATION,readonly"
    --mount "type=bind,src=$CAMPAIGN_ROOT,dst=$CAMPAIGN_ROOT,readonly"
    --mount "type=bind,src=$EVALUATION_ROOT,dst=$EVALUATION_ROOT"
    --mount "type=bind,src=$CONTRACT,dst=$CONTRACT,readonly"
    --mount "type=bind,src=$EVALUATION_CONFIG,dst=$EVALUATION_CONFIG,readonly"
    --mount "type=bind,src=$ENDPOINT_DIR,dst=$ENDPOINT_DIR,readonly"
    --mount "type=bind,src=$ANCHOR_POOL,dst=$ANCHOR_POOL,readonly"
    --mount "type=bind,src=$SEALED_TEACHER,dst=$SEALED_TEACHER,readonly"
    --mount "type=bind,src=$SEALED_TEACHER,dst=$TEACHER_CHECKPOINT,readonly"
    "${FIRST_CONDITION_BUNDLE_MOUNTS[@]}"
    --tmpfs /run/round-dagger-eval:rw,nosuid,nodev,noexec,size=1g
    --env "PYTHONPATH=$EVALUATION_SOURCE_ROOT:$GR00T_SOURCE_ROOT"
    --env "GR00T_ROOT=$GR00T_SOURCE_ROOT"
    "${ENV_ARGS[@]}"
    "$IMAGE"
    /usr/bin/env bash "$RUNNER"
    --root "$CAMPAIGN_ROOT"
    --contract "$CONTRACT"
    --results-dir "$RESULTS_DIR"
    --gr00t-root "$GR00T_SOURCE_ROOT"
    --evaluation-config "$EVALUATION_CONFIG"
    --endpoint-dir "$ENDPOINT_DIR"
    --teacher-checkpoint "$TEACHER_CHECKPOINT"
    --teacher-sha256 "$TEACHER_SHA"
    --sealed-teacher-checkpoint "$SEALED_TEACHER"
    --evaluation-source-inventory-sha256 "$EVALUATION_SOURCE_INVENTORY_SHA"
    --anchor-pool-dir "$ANCHOR_POOL"
    --anchor-count "$ANCHOR_COUNT"
    --anchor-index "$ANCHOR_INDEX"
    --anchor-inventory-sha256 "$ANCHOR_INVENTORY_SHA"
    --motion-split-manifest "$MOTION_SPLIT"
    --motion-split-manifest-sha256 "$MOTION_SPLIT_SHA"
    --heldout-plan "$HELDOUT_PLAN"
    --heldout-plan-sha256 "$HELDOUT_PLAN_SHA"
    --scratch-root /run/round-dagger-eval
    --require-teacher-readonly-mount
    --orchestrator-result "$EVALUATOR_RESULT"
    --python "$CONTAINER_PYTHON")

if ((DRY_RUN)); then
    [[ ! -e $EVALUATOR_RESULT ]] || fatal "canonical evaluator result already exists"
    # If an interrupted run already prepared the contract, verify it now.
    if [[ -e $CONTRACT ]]; then
        require_regular "$CONTRACT" "existing paired evaluation contract"
        "$CONTROL_PYTHON" -m round_dagger.evaluation verify-contract \
            --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" >/dev/null || \
            fatal "existing paired evaluation contract is invalid"
    fi
    verify_first_condition_bundles >/dev/null || \
        fatal "first-condition bundles changed before dry-run mount plan"
    print_command "${PREPARE[@]}"
    print_command "${DOCKER_RUN[@]}"
    echo "PRODUCTION_PAIRED_EVALUATION_DRY_RUN_OK candidate_runtime_sha256=$CANDIDATE_RUNTIME_SHA evaluator_code_sha256=$EVALUATOR_CODE_SHA"
    exit 0
fi

verify_first_condition_bundles >/dev/null || \
    fatal "first-condition bundles changed before contract publication"
"${PREPARE[@]}" >/dev/null || fatal "paired evaluation contract preparation failed"
verify_first_condition_bundles >/dev/null || \
    fatal "first-condition bundles changed before container launch"
"${DOCKER_RUN[@]}" || fatal "paired evaluation container failed"
verify_first_condition_bundles >/dev/null || \
    fatal "first-condition bundles changed during paired evaluation"
verify_anchor_pool
verify_evaluation_source
# Re-read every outcome, renderer attestation, and raw metrics digest on the
# host and deterministically re-score it.  Immutable publication makes this an
# exact retry when the container already summarized the run; a fabricated or
# partial result cannot make the hook return success.
"$CONTROL_PYTHON" -m round_dagger.evaluation summarize \
    --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
    --results-dir "$RESULTS_DIR" --output "$EVALUATOR_RESULT" >/dev/null || \
    fatal "paired evaluation evidence validation failed"
require_regular "$EVALUATOR_RESULT" "orchestrator evaluator result"
"$CONTROL_PYTHON" -m round_dagger.evaluation verify-result \
    --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
    --results-dir "$RESULTS_DIR" --result "$EVALUATOR_RESULT" >/dev/null || \
    fatal "paired evaluation deep result verification failed"

# Validate the hook's exact output lineage and its digest-bound canonical bundle
# before returning success to the coordinator.
"$CONTROL_PYTHON" - "$EVALUATOR_RESULT" "$CAMPAIGN_ID" "$MACRO_ROUND" \
    "$CANDIDATE_MANIFEST_SHA" "$EVALUATOR_CONFIG_SHA" "$RESULTS_DIR" \
    "$CAMPAIGN_ROOT" <<'PY'
import sys
from pathlib import Path

from round_dagger.artifacts import load_strict_json, sha256_file
from round_dagger.evaluation import EVALUATOR_RESULT_SCHEMA

path = Path(sys.argv[1])
document = load_strict_json(path)
expected_keys = {
    "schema", "campaign_id", "macro_round", "candidate_manifest_sha256",
    "evaluator_config_sha256", "evidence_bundle_path",
    "evidence_bundle_sha256", "passed", "metrics",
}
if not isinstance(document, dict) or set(document) != expected_keys:
    raise SystemExit("evaluator result fields are not exact")
expected = {
    "schema": EVALUATOR_RESULT_SCHEMA,
    "campaign_id": sys.argv[2],
    "macro_round": int(sys.argv[3]),
    "candidate_manifest_sha256": sys.argv[4],
    "evaluator_config_sha256": sys.argv[5],
}
if any(document.get(key) != value for key, value in expected.items()):
    raise SystemExit("evaluator result lineage mismatch")
bundle = Path(sys.argv[6]) / "evaluation_bundle.json"
if bundle.is_symlink() or not bundle.is_file():
    raise SystemExit("canonical evaluation evidence bundle is missing or unsafe")
if document["evidence_bundle_path"] != bundle.relative_to(Path(sys.argv[7])).as_posix():
    raise SystemExit("evaluation evidence bundle path is non-canonical")
if sha256_file(bundle) != document["evidence_bundle_sha256"]:
    raise SystemExit("evaluation evidence bundle SHA mismatch")
PY
echo "PRODUCTION_PAIRED_EVALUATION_COMPLETE result=$EVALUATOR_RESULT"
