#!/usr/bin/env bash
# Sequential parent/candidate fixed-seed evaluation on one local RTX actor host.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: run_paired_evaluation.sh --root DIR --contract FILE --results-dir DIR
       --gr00t-root DIR --evaluation-config FILE --endpoint-dir DIR
       --teacher-checkpoint FILE --teacher-sha256 SHA256
       --sealed-teacher-checkpoint FILE
       --evaluation-source-inventory-sha256 SHA256
       --anchor-pool-dir DIR --anchor-count N --anchor-index N
       --anchor-inventory-sha256 SHA256
       --motion-split-manifest FILE --motion-split-manifest-sha256 SHA256
       --heldout-plan FILE --heldout-plan-sha256 SHA256 --scratch-root DIR
       [--require-teacher-readonly-mount]
       [--orchestrator-result FILE] [--python PYTHON] [--dry-run]

The contract must already be prepared by `python -m round_dagger.evaluation
prepare`. Dry-run verifies all immutable local inputs and prints the exact
parent/candidate commands without creating results or launching Isaac.
When --orchestrator-result is omitted, ROUND_DAGGER_EVALUATOR_RESULT must be
set by MacroRoundOrchestrator.  The result is evidence-only; this launcher
never publishes a policy evaluation or changes coordinator state.
EOF
}

ROOT=
CONTRACT=
RESULTS_DIR=
GR00T_ROOT=
EVALUATION_CONFIG=
ENDPOINT_DIR=
TEACHER_CHECKPOINT=
TEACHER_SHA=
SEALED_TEACHER=
EVALUATION_SOURCE_INVENTORY_SHA=
ANCHOR_POOL=
ANCHOR_COUNT=
ANCHOR_INDEX=
ANCHOR_INVENTORY_SHA=
MOTION_SPLIT=
MOTION_SPLIT_SHA=
HELDOUT_PLAN=
HELDOUT_PLAN_SHA=
SCRATCH_ROOT=
REQUIRE_TEACHER_READONLY_MOUNT=0
ORCHESTRATOR_RESULT=${ROUND_DAGGER_EVALUATOR_RESULT:-}
PYTHON_BIN=python3
DRY_RUN=0
while (($#)); do
    case "$1" in
        --root) ROOT=${2:?}; shift 2 ;;
        --contract) CONTRACT=${2:?}; shift 2 ;;
        --results-dir) RESULTS_DIR=${2:?}; shift 2 ;;
        --gr00t-root) GR00T_ROOT=${2:?}; shift 2 ;;
        --evaluation-config) EVALUATION_CONFIG=${2:?}; shift 2 ;;
        --endpoint-dir) ENDPOINT_DIR=${2:?}; shift 2 ;;
        --teacher-checkpoint) TEACHER_CHECKPOINT=${2:?}; shift 2 ;;
        --teacher-sha256) TEACHER_SHA=${2:?}; shift 2 ;;
        --sealed-teacher-checkpoint) SEALED_TEACHER=${2:?}; shift 2 ;;
        --evaluation-source-inventory-sha256)
            EVALUATION_SOURCE_INVENTORY_SHA=${2:?}; shift 2 ;;
        --anchor-pool-dir) ANCHOR_POOL=${2:?}; shift 2 ;;
        --anchor-count) ANCHOR_COUNT=${2:?}; shift 2 ;;
        --anchor-index) ANCHOR_INDEX=${2:?}; shift 2 ;;
        --anchor-inventory-sha256) ANCHOR_INVENTORY_SHA=${2:?}; shift 2 ;;
        --motion-split-manifest) MOTION_SPLIT=${2:?}; shift 2 ;;
        --motion-split-manifest-sha256) MOTION_SPLIT_SHA=${2:?}; shift 2 ;;
        --heldout-plan) HELDOUT_PLAN=${2:?}; shift 2 ;;
        --heldout-plan-sha256) HELDOUT_PLAN_SHA=${2:?}; shift 2 ;;
        --scratch-root) SCRATCH_ROOT=${2:?}; shift 2 ;;
        --require-teacher-readonly-mount)
            REQUIRE_TEACHER_READONLY_MOUNT=1
            shift
            ;;
        --orchestrator-result) ORCHESTRATOR_RESULT=${2:?}; shift 2 ;;
        --python) PYTHON_BIN=${2:?}; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

fatal() { echo "FATAL: $*" >&2; exit 2; }
for value in ROOT CONTRACT RESULTS_DIR GR00T_ROOT EVALUATION_CONFIG ENDPOINT_DIR \
    TEACHER_CHECKPOINT TEACHER_SHA SEALED_TEACHER EVALUATION_SOURCE_INVENTORY_SHA \
    ANCHOR_POOL ANCHOR_COUNT ANCHOR_INDEX ANCHOR_INVENTORY_SHA SCRATCH_ROOT \
    ORCHESTRATOR_RESULT; do
    [[ -n ${!value} ]] || fatal "missing --${value,,}"
done
[[ -d $ROOT && ! -L $ROOT ]] || fatal "unsafe/missing campaign root"
[[ -f $CONTRACT && ! -L $CONTRACT ]] || fatal "unsafe/missing evaluation contract"
[[ -f $EVALUATION_CONFIG && ! -L $EVALUATION_CONFIG ]] || fatal "unsafe/missing evaluation config"
[[ $TEACHER_CHECKPOINT == /* && -f $TEACHER_CHECKPOINT && \
   ! -L $TEACHER_CHECKPOINT ]] || \
    fatal "teacher checkpoint must be an absolute, non-symlink regular file"
[[ $TEACHER_SHA =~ ^[0-9a-f]{64}$ ]] || \
    fatal "teacher checkpoint SHA must be a lowercase SHA-256"
[[ $SEALED_TEACHER == /* && -f $SEALED_TEACHER && ! -L $SEALED_TEACHER ]] || \
    fatal "sealed teacher checkpoint must be an absolute, non-symlink regular file"
[[ $SEALED_TEACHER != "$TEACHER_CHECKPOINT" ]] || \
    fatal "sealed teacher source must differ from the configured target"
EVALUATOR="$GR00T_ROOT/gr00t/rl/eval_agent_trl.py"
[[ -f $EVALUATOR && ! -L $EVALUATOR ]] || fatal "unsafe/missing GR00T evaluator"

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ -n ${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:-} ]]; then
    [[ $ROUND_DAGGER_EVALUATION_SOURCE_ROOT == "$REPOSITORY_ROOT" ]] || \
        fatal "paired evaluator is not executing from the sealed evaluation source"
    [[ $GR00T_ROOT == "$REPOSITORY_ROOT/GR00T-VisualSim2Real" ]] || \
        fatal "GR00T evaluator root is not inside the sealed evaluation source"
fi
[[ $EVALUATION_SOURCE_INVENTORY_SHA =~ ^[0-9a-f]{64}$ ]] || \
    fatal "evaluation source inventory SHA must be lowercase SHA-256"
[[ $ANCHOR_POOL == /* && -d $ANCHOR_POOL && ! -L $ANCHOR_POOL ]] || \
    fatal "anchor pool must be an absolute, non-symlink directory"
[[ $ANCHOR_COUNT =~ ^[1-9][0-9]*$ ]] || fatal "anchor count must be positive"
[[ $ANCHOR_INDEX =~ ^(0|[1-9][0-9]*)$ ]] || \
    fatal "anchor index must be non-negative"
((10#$ANCHOR_INDEX < 10#$ANCHOR_COUNT)) || \
    fatal "anchor index must be smaller than anchor count"
[[ $ANCHOR_INVENTORY_SHA =~ ^[0-9a-f]{64}$ ]] || \
    fatal "anchor inventory SHA must be lowercase SHA-256"
MOTION_SPLIT_INPUTS=0
if [[ -n $MOTION_SPLIT || -n $MOTION_SPLIT_SHA || -n $HELDOUT_PLAN || \
      -n $HELDOUT_PLAN_SHA || -n ${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:-} ]]; then
    MOTION_SPLIT_INPUTS=1
    for value in MOTION_SPLIT MOTION_SPLIT_SHA HELDOUT_PLAN HELDOUT_PLAN_SHA; do
        [[ -n ${!value} ]] || fatal "production motion split inputs are incomplete"
    done
    [[ $MOTION_SPLIT == /* && -f $MOTION_SPLIT && ! -L $MOTION_SPLIT ]] || \
        fatal "motion split manifest must be an absolute, non-symlink regular file"
    [[ $HELDOUT_PLAN == /* && -f $HELDOUT_PLAN && ! -L $HELDOUT_PLAN ]] || \
        fatal "held-out plan must be an absolute, non-symlink regular file"
    [[ $MOTION_SPLIT_SHA =~ ^[0-9a-f]{64}$ ]] || \
        fatal "motion split manifest SHA must be lowercase SHA-256"
    [[ $HELDOUT_PLAN_SHA =~ ^[0-9a-f]{64}$ ]] || \
        fatal "held-out plan SHA must be lowercase SHA-256"
    [[ $(sha256sum "$MOTION_SPLIT" | awk '{print $1}') == "$MOTION_SPLIT_SHA" ]] || \
        fatal "motion split manifest SHA mismatch"
    [[ $(sha256sum "$HELDOUT_PLAN" | awk '{print $1}') == "$HELDOUT_PLAN_SHA" ]] || \
        fatal "held-out plan SHA mismatch"
fi
# Isaac's container entrypoint contributes extension and Kit package roots to
# PYTHONPATH.  Keep those paths after adding the sealed control/evaluator
# sources; replacing them here makes an otherwise valid Isaac runtime fail at
# import time.
export PYTHONPATH="$REPOSITORY_ROOT:$GR00T_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ $SCRATCH_ROOT == /* && -d $SCRATCH_ROOT && ! -L $SCRATCH_ROOT ]] || \
    fatal "scratch root must be an absolute, non-symlink directory"
if [[ -n ${ROUND_DAGGER_EVALUATION_SOURCE_ROOT:-} ]]; then
    "$PYTHON_BIN" - "$SCRATCH_ROOT" <<'PY' || \
        fatal "production evaluator scratch root is not an exact hardened tmpfs"
import re
import sys

target = sys.argv[1]
found = []
with open("/proc/self/mountinfo", encoding="utf-8") as handle:
    for raw in handle:
        left, right = raw.rstrip("\n").split(" - ", 1)
        fields = left.split()
        mountpoint = re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            fields[4],
        )
        if mountpoint == target:
            found.append((set(fields[5].split(",")), right.split()[0]))
if len(found) != 1:
    raise SystemExit(1)
options, filesystem = found[0]
if filesystem != "tmpfs" or not {"rw", "nosuid", "nodev", "noexec"} <= options:
    raise SystemExit(1)
PY
fi

verify_anchor_pool() {
    "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/round_dagger/verify_anchor_pool.py" \
        --root "$ANCHOR_POOL" --expected-count "$ANCHOR_COUNT" \
        --expected-sha256 "$ANCHOR_INVENTORY_SHA" \
        >/dev/null || fatal "evaluation anchor pool verification failed"
}
verify_anchor_pool

"$PYTHON_BIN" -m round_dagger.evaluation verify-contract \
    --root "$ROOT" --contract "$CONTRACT" >/dev/null || fatal "contract verification failed"

# The production hook already validates the immutable scenario artifact.  At
# this inner layer, bind the plan to the scenario list embedded in the signed
# evaluation contract without publishing any additional durable file.
if ((MOTION_SPLIT_INPUTS)); then
    "$PYTHON_BIN" - "$CONTRACT" "$MOTION_SPLIT" "$MOTION_SPLIT_SHA" \
        "$HELDOUT_PLAN" "$HELDOUT_PLAN_SHA" <<'PY' || \
        fatal "evaluation contract differs from the held-out seed-block plan"
import sys
from pathlib import Path
from round_dagger.artifacts import load_strict_json
from round_dagger.motion_split_contract import load_verified_contract, validate_scenarios

contract = load_strict_json(Path(sys.argv[1]))
plan = load_verified_contract(
    manifest_path=Path(sys.argv[2]), manifest_sha256=sys.argv[3],
    plan_path=Path(sys.argv[4]), plan_sha256=sys.argv[5],
)
validate_scenarios(plan, contract["scenario_cases"])
PY
fi

readarray -t CONTRACT_FIELDS < <("$PYTHON_BIN" - "$CONTRACT" "$ROOT" <<'PY'
import json, sys
from round_dagger.state import load_campaign

d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["evaluation_config_sha256"])
print(d["evaluator_code_sha256"])
print(d["expected_renderer_count"])
print(d["num_envs_per_seed"])
print(load_campaign(sys.argv[2])["teacher_sha256"])
print(d["anchor_pool"]["path"])
print(d["anchor_pool"]["anchor_count"])
print(d["anchor_pool"]["anchor_index"])
print(d["anchor_pool"]["inventory_sha256"])
PY
)
[[ ${#CONTRACT_FIELDS[@]} -eq 9 ]] || fatal "cannot read evaluation contract"
EXPECTED_CONFIG_SHA=${CONTRACT_FIELDS[0]}
EXPECTED_EVALUATOR_SHA=${CONTRACT_FIELDS[1]}
EXPECTED_RENDERERS=${CONTRACT_FIELDS[2]}
NUM_ENVS=${CONTRACT_FIELDS[3]}
EXPECTED_TEACHER_SHA=${CONTRACT_FIELDS[4]}
EXPECTED_ANCHOR_POOL=${CONTRACT_FIELDS[5]}
EXPECTED_ANCHOR_COUNT=${CONTRACT_FIELDS[6]}
EXPECTED_ANCHOR_INDEX=${CONTRACT_FIELDS[7]}
EXPECTED_ANCHOR_INVENTORY_SHA=${CONTRACT_FIELDS[8]}
[[ $ANCHOR_POOL == "$EXPECTED_ANCHOR_POOL" && \
   $ANCHOR_COUNT == "$EXPECTED_ANCHOR_COUNT" && \
   $ANCHOR_INDEX == "$EXPECTED_ANCHOR_INDEX" && \
   $ANCHOR_INVENTORY_SHA == "$EXPECTED_ANCHOR_INVENTORY_SHA" ]] || \
    fatal "anchor inputs differ from the immutable evaluation contract"
[[ $TEACHER_SHA == "$EXPECTED_TEACHER_SHA" ]] || \
    fatal "teacher checkpoint SHA differs from campaign"
[[ $(sha256sum "$EVALUATION_CONFIG" | awk '{print $1}') == "$EXPECTED_CONFIG_SHA" ]] || \
    fatal "evaluation config SHA mismatch"
if [[ -n ${ROUND_DAGGER_EVALUATOR_CONFIG_SHA256:-} ]]; then
    [[ $ROUND_DAGGER_EVALUATOR_CONFIG_SHA256 == "$EXPECTED_CONFIG_SHA" ]] || \
        fatal "orchestrator evaluator config SHA differs from paired contract"
fi
ACTUAL_EVALUATOR_SHA=$("$PYTHON_BIN" - "$REPOSITORY_ROOT" "$GR00T_ROOT" \
    "$EVALUATION_SOURCE_INVENTORY_SHA" <<'PY'
import sys
from round_dagger.evaluation import evaluator_source_bundle_sha256

print(evaluator_source_bundle_sha256(sys.argv[1], sys.argv[2], sys.argv[3]))
PY
)
[[ $ACTUAL_EVALUATOR_SHA == "$EXPECTED_EVALUATOR_SHA" ]] || \
    fatal "evaluator source-bundle SHA mismatch"

CONFIGURED_TEACHER=$("$PYTHON_BIN" - "$EVALUATION_CONFIG" <<'PY'
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
print(teacher)
videogen = document.get("algo", {}).get("config", {}).get("videogen", {})
anchor = videogen.get("anchor_pool_dir")
index = videogen.get("anchor_index")
if not isinstance(anchor, str) or not Path(anchor).is_absolute():
    raise SystemExit("evaluation config anchor_pool_dir must be absolute")
if isinstance(index, bool) or not isinstance(index, int) or index < 0:
    raise SystemExit("evaluation config anchor_index must be non-negative")
print(anchor)
print(index)
PY
) || fatal "cannot read teacher path from the pinned evaluation config"
mapfile -t CONFIGURED_INPUTS <<<"$CONFIGURED_TEACHER"
[[ ${#CONFIGURED_INPUTS[@]} -eq 3 ]] || fatal "pinned config input parse is incomplete"
[[ ${CONFIGURED_INPUTS[0]} == "$TEACHER_CHECKPOINT" ]] || \
    fatal "pinned evaluation config teacher path mismatch"
[[ ${CONFIGURED_INPUTS[1]} == "$ANCHOR_POOL" ]] || \
    fatal "pinned evaluation config anchor pool path mismatch"
[[ ${CONFIGURED_INPUTS[2]} == "$ANCHOR_INDEX" ]] || \
    fatal "pinned evaluation config anchor index mismatch"

verify_teacher() {
    [[ -f $TEACHER_CHECKPOINT && ! -L $TEACHER_CHECKPOINT ]] || \
        fatal "teacher checkpoint became unsafe or unavailable"
    [[ -f $SEALED_TEACHER && ! -L $SEALED_TEACHER ]] || \
        fatal "sealed teacher checkpoint became unsafe or unavailable"
    [[ $(stat -c %a "$TEACHER_CHECKPOINT") == 444 ]] || \
        fatal "teacher checkpoint mode must be exactly 0444"
    [[ $(stat -c %a "$SEALED_TEACHER") == 444 ]] || \
        fatal "sealed teacher checkpoint mode must be exactly 0444"
    [[ $(stat -c %h "$SEALED_TEACHER") == 1 ]] || \
        fatal "sealed teacher checkpoint must have exactly one hard link"
    [[ $(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}') == "$TEACHER_SHA" ]] || \
        fatal "teacher checkpoint SHA mismatch"
    [[ $(sha256sum "$SEALED_TEACHER" | awk '{print $1}') == "$TEACHER_SHA" ]] || \
        fatal "sealed teacher checkpoint SHA mismatch"
    if ((REQUIRE_TEACHER_READONLY_MOUNT)); then
        [[ $(stat -c '%d:%i' "$TEACHER_CHECKPOINT") == \
           "$(stat -c '%d:%i' "$SEALED_TEACHER")" ]] || \
            fatal "teacher target and sealed source mounts differ"
        "$PYTHON_BIN" - "$TEACHER_CHECKPOINT" "$SEALED_TEACHER" <<'PY' || \
            fatal "teacher checkpoints are not exact read-only mounts"
import re
import sys


def decode_mountinfo(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


expected = set(sys.argv[1:])
found = {path: [] for path in expected}
with open("/proc/self/mountinfo", "r", encoding="utf-8") as handle:
    for raw in handle:
        left = raw.rstrip("\n").split(" - ", 1)[0].split()
        if len(left) < 6:
            continue
        mountpoint = decode_mountinfo(left[4])
        if mountpoint in found:
            found[mountpoint].append(set(left[5].split(",")))
if any(len(entries) != 1 or "ro" not in entries[0] for entries in found.values()):
    raise SystemExit(1)
PY
    fi
}

# Check the exact file presented to the process before any evaluation setup.
# The same check runs immediately before each policy-role evaluator invocation,
# closing the window for a mutable broad bind to replace the pinned teacher.
verify_teacher
[[ $EXPECTED_RENDERERS =~ ^[1-9][0-9]*$ ]] || fatal "invalid renderer count"
[[ $NUM_ENVS =~ ^[1-9][0-9]*$ ]] || fatal "invalid environment count"
[[ -d $ENDPOINT_DIR && ! -L $ENDPOINT_DIR ]] || fatal "unsafe/missing endpoint directory"
mapfile -t ENDPOINT_FILES < <(find "$ENDPOINT_DIR" -maxdepth 1 -type f -name 'renderer_*.txt' -print | sort)
[[ ${#ENDPOINT_FILES[@]} -eq $EXPECTED_RENDERERS ]] || \
    fatal "endpoint inventory count differs from contract"
ENDPOINT_LIST="[$(IFS=,; echo "${ENDPOINT_FILES[*]}")]"

print_command() {
    printf 'DRY_RUN'
    printf ' %q' "$@"
    printf '\n'
}

next_attempt() {
    local base=$1 path name digits value canonical max=0
    if [[ -e $base ]]; then
        [[ -d $base && ! -L $base ]] || fatal "unsafe attempt directory $base"
        while IFS= read -r -d '' path; do
            name=${path##*/}
            [[ $name =~ ^attempt_([0-9]{6,})$ ]] || \
                fatal "ambiguous evaluation attempt entry $path"
            digits=${BASH_REMATCH[1]}
            value=$((10#$digits))
            ((value >= 1)) || fatal "attempt number must be positive: $path"
            canonical=$(printf 'attempt_%06d' "$value")
            [[ $name == "$canonical" ]] || \
                fatal "non-canonical evaluation attempt entry $path"
            ((value > max)) && max=$value
        done < <(find "$base" -mindepth 1 -maxdepth 1 -print0 | LC_ALL=C sort -z)
    fi
    printf '%d\n' "$((max + 1))"
}

if ((!DRY_RUN)); then
    if [[ -e $RESULTS_DIR ]]; then
        [[ -d $RESULTS_DIR && ! -L $RESULTS_DIR ]] || \
            fatal "unsafe evaluation results directory"
    else
        mkdir -p "$RESULTS_DIR"
    fi
fi

# Staged checkpoints are execution inputs, not durable evidence.  Only one
# role/seed attempt may occupy the bounded tmpfs at a time.  Clean the active
# namespace after the canonical outcome has been verified, and also from the
# EXIT trap on every failure/signal path.  Attempts and evaluator evidence live
# under RESULTS_DIR and are never removed here.
ACTIVE_STAGE_ROLE_DIR=
cleanup_stage_dir() {
    local target=$1
    [[ -n $target && $target == "$SCRATCH_ROOT"/* ]] || \
        fatal "refusing to clean an unsafe evaluation scratch path"
    if [[ ! -e $target ]]; then
        return 0
    fi
    [[ -d $target && ! -L $target ]] || \
        fatal "evaluation scratch attempt became unsafe"
    find "$target" -xdev -depth -delete || \
        fatal "cannot clean evaluation scratch attempt"
    [[ ! -e $target ]] || fatal "evaluation scratch attempt cleanup was incomplete"
}
cleanup_active_stage_on_exit() {
    local rc=$?
    trap - EXIT
    if [[ -n $ACTIVE_STAGE_ROLE_DIR ]]; then
        cleanup_stage_dir "$ACTIVE_STAGE_ROLE_DIR"
    fi
    exit "$rc"
}
trap cleanup_active_stage_on_exit EXIT

# The pinned production evaluation config predates the physical-safety signal.
# Supply every required leaf explicitly so Hydra can compose that legacy config
# while PhysicalSafetyThresholds.from_config remains strict and fail-closed.
# These values are the approved production thresholds in the GR00T environment
# contract; do not infer them from task-termination switches.
PHYSICAL_SAFETY_OVERRIDES=(
    ++env.config.safety_thresholds.contact_force_n=10.0
    ++env.config.safety_thresholds.gravity_xy=0.5
    ++env.config.safety_thresholds.min_base_height_m=0.3
    ++env.config.safety_thresholds.dof_velocity_limit_fraction=0.98
    ++env.config.safety_thresholds.torque_limit_fraction=0.98
    ++env.config.safety_thresholds.min_object_height_m=0.6
    ++env.config.safety_thresholds.right_arm_velocity_rad_s=3.14
)

while IFS=$'\t' read -r ROLE SEED CHECKPOINT_REL CHECKPOINT_SHA RUNTIME_POLICY_SHA; do
    [[ $ROLE == parent || $ROLE == candidate ]] || fatal "invalid role from contract"
    [[ $SEED =~ ^[0-9]+$ ]] || fatal "invalid seed from contract"
    MOTION_OVERRIDES=()
    if ((MOTION_SPLIT_INPUTS)); then
        mapfile -t MOTION_BLOCK < <(
            "$PYTHON_BIN" -m round_dagger.motion_split_contract block \
                --manifest "$MOTION_SPLIT" --manifest-sha256 "$MOTION_SPLIT_SHA" \
                --plan "$HELDOUT_PLAN" --plan-sha256 "$HELDOUT_PLAN_SHA" \
                --seed "$SEED"
        ) || fatal "cannot resolve immutable motion block for seed $SEED"
        [[ ${#MOTION_BLOCK[@]} -eq 2 && ${MOTION_BLOCK[0]} =~ ^(0|[1-9][0-9]*)$ && \
           ${MOTION_BLOCK[1]} == 32 ]] || \
            fatal "held-out motion block for seed $SEED is malformed"
        MOTION_OFFSET=${MOTION_BLOCK[0]}
        MOTION_LIMIT=${MOTION_BLOCK[1]}
        MOTION_OVERRIDES=(
            ++env.config.reset_from_dataset.motion_split_manifest="$MOTION_SPLIT"
            ++env.config.reset_from_dataset.motion_split_manifest_sha256="$MOTION_SPLIT_SHA"
            ++env.config.reset_from_dataset.motion_split_role=eval
            ++env.config.reset_from_dataset.motion_split_verify_content=true
            ++env.config.reset_from_dataset.motion_split_offset="$MOTION_OFFSET"
            ++env.config.reset_from_dataset.motion_split_limit="$MOTION_LIMIT"
            ++env.config.reset_from_dataset.motion_assignment_offset=0
        )
    fi
    CHECKPOINT="$ROOT/$CHECKPOINT_REL"
    [[ -f $CHECKPOINT && ! -L $CHECKPOINT ]] || fatal "unsafe/missing $ROLE checkpoint"
    [[ $(sha256sum "$CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA" ]] || \
        fatal "$ROLE checkpoint SHA mismatch"
    CANONICAL_DIR="$RESULTS_DIR/$ROLE/seed_$SEED"
    CANONICAL_OUTCOME="$CANONICAL_DIR/outcome.json"
    ATTEMPT_BASE="$RESULTS_DIR/attempts/$ROLE/seed_$SEED"
    ATTEMPT=1
    if ((!DRY_RUN)); then
        if [[ -e $CANONICAL_OUTCOME ]]; then
            [[ -f $CANONICAL_OUTCOME && ! -L $CANONICAL_OUTCOME ]] || \
                fatal "unsafe canonical outcome $CANONICAL_OUTCOME"
            "$PYTHON_BIN" -m round_dagger.evaluation verify-outcome \
                --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" \
                --seed "$SEED" --outcome "$CANONICAL_OUTCOME" >/dev/null || \
                fatal "canonical $ROLE seed $SEED outcome is invalid or ambiguous"
            echo "PAIRED_EVALUATION_CASE_REUSED role=$ROLE seed=$SEED"
            continue
        fi
        # Preserve a pre-resume launcher partial case as an immutable failed
        # attempt.  A canonical directory containing an invalid outcome is
        # rejected above; only an unattested partial directory is migrated.
        if [[ -e $CANONICAL_DIR ]]; then
            [[ -d $CANONICAL_DIR && ! -L $CANONICAL_DIR ]] || \
                fatal "ambiguous partial case state $CANONICAL_DIR"
            mkdir -p "$ATTEMPT_BASE"
            LEGACY_ATTEMPT_NUMBER=$(next_attempt "$ATTEMPT_BASE")
            LEGACY_ATTEMPT=$(printf '%s/attempt_%06d' \
                "$ATTEMPT_BASE" "$LEGACY_ATTEMPT_NUMBER")
            [[ ! -e $LEGACY_ATTEMPT ]] || fatal "attempt allocation collision"
            mv -- "$CANONICAL_DIR" "$LEGACY_ATTEMPT"
            echo "PAIRED_EVALUATION_PARTIAL_PRESERVED role=$ROLE seed=$SEED attempt=$LEGACY_ATTEMPT_NUMBER"
        fi

        # A process can die after attest wrote a complete immutable attempt but
        # before publish-outcome linked it into the canonical case directory.
        # Recover exactly one such attempt without launching Isaac again.  Any
        # attested-but-invalid or multiply valid state is ambiguous and fails
        # closed; partial attempts with no outcome remain preserved diagnostics.
        VALID_ATTEMPT_OUTCOMES=()
        VALID_ATTEMPT_NUMBERS=()
        if [[ -e $ATTEMPT_BASE ]]; then
            [[ -d $ATTEMPT_BASE && ! -L $ATTEMPT_BASE ]] || \
                fatal "unsafe attempt directory $ATTEMPT_BASE"
            while IFS= read -r -d '' ATTEMPT_PATH; do
                ATTEMPT_NAME=${ATTEMPT_PATH##*/}
                [[ $ATTEMPT_NAME =~ ^attempt_([0-9]{6,})$ ]] || \
                    fatal "ambiguous evaluation attempt entry $ATTEMPT_PATH"
                ATTEMPT_DIGITS=${BASH_REMATCH[1]}
                ATTEMPT_NUMBER=$((10#$ATTEMPT_DIGITS))
                ((ATTEMPT_NUMBER >= 1)) || \
                    fatal "attempt number must be positive: $ATTEMPT_PATH"
                CANONICAL_ATTEMPT_NAME=$(printf 'attempt_%06d' "$ATTEMPT_NUMBER")
                [[ $ATTEMPT_NAME == "$CANONICAL_ATTEMPT_NAME" ]] || \
                    fatal "non-canonical evaluation attempt entry $ATTEMPT_PATH"
                [[ -d $ATTEMPT_PATH && ! -L $ATTEMPT_PATH ]] || \
                    fatal "unsafe evaluation attempt $ATTEMPT_PATH"
                ATTEMPT_OUTCOME="$ATTEMPT_PATH/outcome.json"
                if [[ -e $ATTEMPT_OUTCOME ]]; then
                    [[ -f $ATTEMPT_OUTCOME && ! -L $ATTEMPT_OUTCOME ]] || \
                        fatal "unsafe attempt outcome $ATTEMPT_OUTCOME"
                    "$PYTHON_BIN" -m round_dagger.evaluation verify-outcome \
                        --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" \
                        --seed "$SEED" --outcome "$ATTEMPT_OUTCOME" \
                        --expected-attempt "$ATTEMPT_NUMBER" >/dev/null || \
                        fatal "invalid or tampered attested attempt $ATTEMPT_OUTCOME"
                    VALID_ATTEMPT_OUTCOMES+=("$ATTEMPT_OUTCOME")
                    VALID_ATTEMPT_NUMBERS+=("$ATTEMPT_NUMBER")
                fi
            done < <(find "$ATTEMPT_BASE" -mindepth 1 -maxdepth 1 -print0 | \
                LC_ALL=C sort -z)
        fi
        ((${#VALID_ATTEMPT_OUTCOMES[@]} <= 1)) || \
            fatal "multiple valid unpublished outcomes make $ROLE seed $SEED ambiguous"
        if ((${#VALID_ATTEMPT_OUTCOMES[@]} == 1)); then
            "$PYTHON_BIN" -m round_dagger.evaluation publish-outcome \
                --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" \
                --seed "$SEED" --source "${VALID_ATTEMPT_OUTCOMES[0]}" \
                --output "$CANONICAL_OUTCOME" \
                --expected-attempt "${VALID_ATTEMPT_NUMBERS[0]}" >/dev/null || \
                fatal "cannot recover attested $ROLE seed $SEED attempt"
            "$PYTHON_BIN" -m round_dagger.evaluation verify-outcome \
                --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" \
                --seed "$SEED" --outcome "$CANONICAL_OUTCOME" >/dev/null || \
                fatal "recovered canonical outcome failed validation"
            echo "PAIRED_EVALUATION_CASE_RECOVERED role=$ROLE seed=$SEED attempt=${VALID_ATTEMPT_NUMBERS[0]}"
            continue
        fi
        ATTEMPT=$(next_attempt "$ATTEMPT_BASE")
    fi
    ATTEMPT_DIR=$(printf '%s/attempt_%06d' "$ATTEMPT_BASE" "$ATTEMPT")
    IMAGE3F_EVAL_RUN_ID=$(
        "$PYTHON_BIN" -m round_dagger.evaluation run-id \
            --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" \
            --seed "$SEED" --attempt "$ATTEMPT"
    ) || fatal "cannot derive Image3F evaluation run ID"
    RESULT_DIR="$ATTEMPT_DIR"
    # The evaluator may materialize loader-derived sibling checkpoints (for
    # example model_step_000000.pt).  Keep every writable loader namespace
    # inside this one immutable attempt so no seed or policy role can consume
    # another case's uncontracted byproduct.
    STAGE_ROLE_DIR="$SCRATCH_ROOT/$ROLE/seed_$SEED/attempt_$(printf '%06d' "$ATTEMPT")"
    STAGED="$STAGE_ROLE_DIR/checkpoint.pt"
    STAGE_PROVENANCE="$STAGE_ROLE_DIR/provenance.json"
    STAGE_CONFIG="$STAGE_ROLE_DIR/config.yaml"
    BEFORE="$RESULT_DIR/renderer_before.json"
    AFTER="$RESULT_DIR/renderer_after.json"
    METRICS="$RESULT_DIR/metrics.json"
    OUTCOME="$RESULT_DIR/outcome.json"

    SANITIZE=("$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/eval/fixed_seed_gate.py" sanitize-checkpoint
        --checkpoint "$CHECKPOINT" --expected-sha256 "$CHECKPOINT_SHA"
        --output "$STAGED" --provenance "$STAGE_PROVENANCE")
    CONFIG_COPY=(install -m 0444 "$EVALUATION_CONFIG" "$STAGE_CONFIG")
    VERIFY_STAGE=("$PYTHON_BIN" -m round_dagger.evaluation_inputs
        --checkpoint-source "$CHECKPOINT"
        --expected-checkpoint-sha256 "$CHECKPOINT_SHA"
        --staged-checkpoint "$STAGED"
        --provenance "$STAGE_PROVENANCE"
        --evaluation-config "$EVALUATION_CONFIG"
        --staged-config "$STAGE_CONFIG"
        --expected-config-sha256 "$EXPECTED_CONFIG_SHA"
        --teacher-checkpoint "$TEACHER_CHECKPOINT")
    PROBE_BEFORE=("$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/scaling/image3f_probe.py"
        --endpoint-dir "$ENDPOINT_DIR" --expected "$EXPECTED_RENDERERS"
        --required-reference-lag 256 --require-reference-history
        --expected-output-views head_color,hand_left,hand_right
        --expected-output-layout vertical --wait-timeout-s 120 --connect-timeout-s 10
        --write-report "$BEFORE" --json)
    ANCHOR_SOURCE=${IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE:-pool}
    [[ $ANCHOR_SOURCE == pool || $ANCHOR_SOURCE == simulator_rgb ]] || \
        fatal "IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE must be pool or simulator_rgb"
    EVAL=(env IMAGE_DAGGER_PAIRED_EVAL=1
        IMAGE_DAGGER_EVAL_RELABEL=1
        IMAGE3F_RUN_ID="$IMAGE3F_EVAL_RUN_ID"
        IMAGE_DAGGER_PAIRED_EVAL_CONTRACT="$CONTRACT"
        IMAGE_DAGGER_EXPECT_RUNTIME_POLICY_SHA256="$RUNTIME_POLICY_SHA"
        "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/eval/eval_agent_metrics_adapter.py"
        --evaluator "$EVALUATOR" --metrics-output "$METRICS" --
        ++checkpoint="$STAGED" ++num_envs="$NUM_ENVS" ++headless=true ++seed="$SEED" ++eval_output_dir="$RESULT_DIR"
        ++algo.config.eval.eval_num_envs_episodes=true
        ++algo.config.eval.num_eval_episodes="$NUM_ENVS"
        ++algo.config.eval.save_videos=false ++algo.config.eval.save_trajectories=false
        ++algo.config.init_at_random_ep_len=false
        ++algo.config.videogen.raytraced_mix_prob=0.0
        ++algo.config.videogen.image3f_endpoint_files="$ENDPOINT_LIST"
        '++algo.config.videogen.view_names=[head_color,hand_left,hand_right]'
        ++algo.config.videogen.reference_lag=256
        ++algo.config.videogen.primary_view=head_color
        ++algo.config.videogen.seed="$SEED"
        "${MOTION_OVERRIDES[@]}"
        ++algo.config.videogen.anchor_pool_dir="$ANCHOR_POOL"
        ++algo.config.videogen.anchor_index="$ANCHOR_INDEX"
        ++algo.config.videogen.anchor_source="$ANCHOR_SOURCE"
        '++simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]'
        ++env.config.image_dagger_policy_view_count=3
        "${PHYSICAL_SAFETY_OVERRIDES[@]}"
        ++algo.config.eval.training_enabled=false
        ++simulator.config.render_results=false ++use_wandb=false
        hydra.run.dir="$RESULT_DIR/hydra" ++output_dir="$RESULT_DIR/output")
    PROBE_AFTER=("${PROBE_BEFORE[@]}")
    PROBE_AFTER[$((${#PROBE_AFTER[@]} - 2))]="$AFTER"
    ATTEST=("$PYTHON_BIN" -m round_dagger.evaluation attest
        --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" --seed "$SEED"
        --metrics "$METRICS" --renderer-before "$BEFORE" --renderer-after "$AFTER"
        --image3f-run-id "$IMAGE3F_EVAL_RUN_ID" --output "$OUTCOME")
    PUBLISH=("$PYTHON_BIN" -m round_dagger.evaluation publish-outcome
        --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" --seed "$SEED"
        --source "$OUTCOME" --output "$CANONICAL_OUTCOME"
        --expected-attempt "$ATTEMPT")
    VERIFY=("$PYTHON_BIN" -m round_dagger.evaluation verify-outcome
        --root "$ROOT" --contract "$CONTRACT" --role "$ROLE" --seed "$SEED"
        --outcome "$CANONICAL_OUTCOME")

    if ((DRY_RUN)); then
        verify_teacher
        print_command "${SANITIZE[@]}"
        print_command "${CONFIG_COPY[@]}"
        print_command "${VERIFY_STAGE[@]}"
        print_command "${PROBE_BEFORE[@]}"
        print_command "${VERIFY_STAGE[@]}"
        print_command env \
            EXPECTED_TRAIN_CHECKPOINT_SHA256='<sha256-of-sanitized-checkpoint>' \
            "${EVAL[@]}"
        print_command "${VERIFY_STAGE[@]}"
        print_command "${PROBE_AFTER[@]}"
        print_command "${ATTEST[@]}"
        print_command "${PUBLISH[@]}"
        print_command "${VERIFY[@]}"
    else
        mkdir -p "$ATTEMPT_BASE"
        mkdir "$RESULT_DIR" || fatal "attempt allocation collision at $RESULT_DIR"
        mkdir -p "$(dirname "$STAGE_ROLE_DIR")"
        ACTIVE_STAGE_ROLE_DIR=$STAGE_ROLE_DIR
        mkdir "$STAGE_ROLE_DIR" || fatal "attempt input allocation collision"
        "${SANITIZE[@]}"
        "${CONFIG_COPY[@]}"
        "${VERIFY_STAGE[@]}" >/dev/null || \
            fatal "new staged evaluation inputs failed verification"
        "${PROBE_BEFORE[@]}" >/dev/null
        VERIFIED_STAGE=$("${VERIFY_STAGE[@]}") || \
            fatal "staged evaluation inputs changed before evaluator launch"
        STAGED_SHA=$(printf '%s' "$VERIFIED_STAGE" | "$PYTHON_BIN" -c '
import json,re,sys
d=json.load(sys.stdin)
v=d.get("staged_checkpoint_sha256")
if d.get("status") != "PASS" or not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None:
    raise SystemExit(2)
print(v)
') || fatal "verified staged checkpoint report is malformed"
        [[ $STAGED_SHA =~ ^[0-9a-f]{64}$ ]] || \
            fatal "verified staged checkpoint SHA-256 is malformed"
        verify_anchor_pool
        verify_teacher
        EXPECTED_TRAIN_CHECKPOINT_SHA256="$STAGED_SHA" "${EVAL[@]}"
        verify_teacher
        verify_anchor_pool
        "${VERIFY_STAGE[@]}" >/dev/null || \
            fatal "staged evaluation inputs changed during evaluator run"
        "${PROBE_AFTER[@]}" >/dev/null
        "${ATTEST[@]}" >/dev/null
        "${PUBLISH[@]}" >/dev/null
        "${VERIFY[@]}" >/dev/null
        cleanup_stage_dir "$STAGE_ROLE_DIR"
        ACTIVE_STAGE_ROLE_DIR=
        echo "PAIRED_EVALUATION_CASE_COMPLETE role=$ROLE seed=$SEED attempt=$ATTEMPT run_id=$IMAGE3F_EVAL_RUN_ID"
    fi
done < <("$PYTHON_BIN" -m round_dagger.evaluation list-cases --root "$ROOT" --contract "$CONTRACT")

verify_anchor_pool

SUMMARY=("$PYTHON_BIN" -m round_dagger.evaluation summarize
    --root "$ROOT" --contract "$CONTRACT" --results-dir "$RESULTS_DIR"
    --output "$ORCHESTRATOR_RESULT")
if ((DRY_RUN)); then
    print_command "${SUMMARY[@]}"
    echo "PAIRED_EVALUATION_DRY_RUN_OK reference_lag=256 views=3 training=false"
else
    "${SUMMARY[@]}"
    echo "PAIRED_EVALUATION_COMPLETE evidence_only=true"
fi
