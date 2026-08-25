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

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
export PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
ENDPOINT_DIR=${ROUND_DAGGER_EVALUATION_ENDPOINT_DIR:?missing ROUND_DAGGER_EVALUATION_ENDPOINT_DIR}
RENDERER_RUNTIME_SHA=${ROUND_DAGGER_RENDERER_RUNTIME_SHA256:?missing ROUND_DAGGER_RENDERER_RUNTIME_SHA256}
EXPECTED_RENDERERS=${ROUND_DAGGER_EXPECTED_RENDERERS:?missing ROUND_DAGGER_EXPECTED_RENDERERS}
GR00T_SOURCE_ROOT=${GR00T_ROOT:?missing GR00T_ROOT}
CONTAINER_PYTHON=${GR00T_PYTHON:?missing GR00T_PYTHON}
DOCKER=${ROUND_DAGGER_DOCKER_BIN:?missing ROUND_DAGGER_DOCKER_BIN}
IMAGE=${ROUND_DAGGER_RTX_IMAGE:?missing ROUND_DAGGER_RTX_IMAGE}
EXPECTED_IMAGE_ID_SHA=${ROUND_DAGGER_RTX_IMAGE_ID_SHA256:?missing ROUND_DAGGER_RTX_IMAGE_ID_SHA256}
CONTROL_PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:-python3}
DRY_RUN=${DRY_RUN:-0}
RECOVERY_ID=${ROUND_DAGGER_EVALUATION_RECOVERY_ID:-}

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
require_sha "$RENDERER_RUNTIME_SHA" ROUND_DAGGER_RENDERER_RUNTIME_SHA256
require_sha "$EXPECTED_IMAGE_ID_SHA" ROUND_DAGGER_RTX_IMAGE_ID_SHA256
require_directory "$CAMPAIGN_ROOT" "campaign root"
require_directory "$GR00T_SOURCE_ROOT" "GR00T root"
require_directory "$ENDPOINT_DIR" "renderer endpoint directory"
require_regular "$PARENT_CHECKPOINT" "parent checkpoint"
require_regular "$CANDIDATE_MANIFEST" "candidate manifest"
require_regular "$EVALUATION_CONFIG" "evaluation config"
require_regular "$SCENARIO_CASES" "scenario cases"
require_regular "$THRESHOLDS" "evaluation thresholds"
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

# All files consumed inside Docker must live below one of the two explicit
# read/write mounts.  This prevents a host-only path from silently becoming a
# different file inside the actor image.
for path in "$REPOSITORY_ROOT" "$CAMPAIGN_ROOT" "$GR00T_SOURCE_ROOT" \
    "$EVALUATION_CONFIG" "$SCENARIO_CASES" "$THRESHOLDS" "$ENDPOINT_DIR"; do
    case "$path" in
        /mnt/data1|/mnt/data1/*|/mnt/data2|/mnt/data2/*) ;;
        *) fatal "container input is outside /mnt/data1 and /mnt/data2: $path" ;;
    esac
done
for mount_root in /mnt/data1 /mnt/data2; do
    require_directory "$mount_root" "Docker mount root"
done

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
    "$PARENT_CHECKPOINT" "$PARENT_RUNTIME_SHA" "$SCENARIO_CASES" "$THRESHOLDS" <<'PY'
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
cases = _validate_cases(load_strict_json(sys.argv[7]))
if cases != load_strict_json(sys.argv[7]):
    raise SystemExit("scenario cases are not canonically ordered")
_validate_thresholds(load_strict_json(sys.argv[8]))
print(candidate["runtime_policy_sha256"])
PY
) || fatal "campaign/candidate/evaluation preflight failed"
require_sha "$CANDIDATE_RUNTIME_SHA" "candidate runtime policy identity"

EVALUATOR_DIGEST_JSON=$($CONTROL_PYTHON -m round_dagger.evaluation evaluator-digest \
    --repository-root "$REPOSITORY_ROOT" --gr00t-root "$GR00T_SOURCE_ROOT") || \
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
    --thresholds "$THRESHOLDS"
    --output "$CONTRACT")

RUNNER="$REPOSITORY_ROOT/scripts/round_dagger/run_paired_evaluation.sh"
require_regular "$RUNNER" "paired evaluation launcher"
CONTAINER_NAME="rde-${CAMPAIGN_ID}-m${MACRO_ROUND}"
ENV_ARGS=()
while IFS= read -r name; do
    case "$name" in
        ROUND_DAGGER_*|IMAGE3F_*|IMAGE_DAGGER_*|ROBORENDER_*|GR00T_ROOT|GR00T_PYTHON|PYTHONPATH)
            ENV_ARGS+=(--env "$name")
            ;;
    esac
done < <(compgen -e | LC_ALL=C sort)
DOCKER_RUN=("$DOCKER" run --rm
    --name "$CONTAINER_NAME"
    --gpus '"device=0"'
    --ipc=private
    --shm-size=32g
    --network=host
    --workdir "$REPOSITORY_ROOT"
    --mount type=bind,src=/mnt/data1,dst=/mnt/data1
    --mount type=bind,src=/mnt/data2,dst=/mnt/data2
    "${ENV_ARGS[@]}"
    "$IMAGE"
    /usr/bin/env bash "$RUNNER"
    --root "$CAMPAIGN_ROOT"
    --contract "$CONTRACT"
    --results-dir "$RESULTS_DIR"
    --gr00t-root "$GR00T_SOURCE_ROOT"
    --evaluation-config "$EVALUATION_CONFIG"
    --endpoint-dir "$ENDPOINT_DIR"
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
    print_command "${PREPARE[@]}"
    print_command "${DOCKER_RUN[@]}"
    echo "PRODUCTION_PAIRED_EVALUATION_DRY_RUN_OK candidate_runtime_sha256=$CANDIDATE_RUNTIME_SHA evaluator_code_sha256=$EVALUATOR_CODE_SHA"
    exit 0
fi

"${PREPARE[@]}" >/dev/null || fatal "paired evaluation contract preparation failed"
"${DOCKER_RUN[@]}" || fatal "paired evaluation container failed"
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
