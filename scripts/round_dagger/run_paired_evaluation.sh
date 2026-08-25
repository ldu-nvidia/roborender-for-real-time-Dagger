#!/usr/bin/env bash
# Sequential parent/candidate fixed-seed evaluation on one local RTX actor host.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: run_paired_evaluation.sh --root DIR --contract FILE --results-dir DIR
       --gr00t-root DIR --evaluation-config FILE --endpoint-dir DIR
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
        --orchestrator-result) ORCHESTRATOR_RESULT=${2:?}; shift 2 ;;
        --python) PYTHON_BIN=${2:?}; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

fatal() { echo "FATAL: $*" >&2; exit 2; }
for value in ROOT CONTRACT RESULTS_DIR GR00T_ROOT EVALUATION_CONFIG ENDPOINT_DIR ORCHESTRATOR_RESULT; do
    [[ -n ${!value} ]] || fatal "missing --${value,,}"
done
[[ -d $ROOT && ! -L $ROOT ]] || fatal "unsafe/missing campaign root"
[[ -f $CONTRACT && ! -L $CONTRACT ]] || fatal "unsafe/missing evaluation contract"
[[ -f $EVALUATION_CONFIG && ! -L $EVALUATION_CONFIG ]] || fatal "unsafe/missing evaluation config"
EVALUATOR="$GR00T_ROOT/gr00t/rl/eval_agent_trl.py"
[[ -f $EVALUATOR && ! -L $EVALUATOR ]] || fatal "unsafe/missing GR00T evaluator"

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
export PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m round_dagger.evaluation verify-contract \
    --root "$ROOT" --contract "$CONTRACT" >/dev/null || fatal "contract verification failed"

readarray -t CONTRACT_FIELDS < <("$PYTHON_BIN" - "$CONTRACT" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["evaluation_config_sha256"])
print(d["evaluator_code_sha256"])
print(d["expected_renderer_count"])
print(d["num_envs_per_seed"])
PY
)
[[ ${#CONTRACT_FIELDS[@]} -eq 4 ]] || fatal "cannot read evaluation contract"
EXPECTED_CONFIG_SHA=${CONTRACT_FIELDS[0]}
EXPECTED_EVALUATOR_SHA=${CONTRACT_FIELDS[1]}
EXPECTED_RENDERERS=${CONTRACT_FIELDS[2]}
NUM_ENVS=${CONTRACT_FIELDS[3]}
[[ $(sha256sum "$EVALUATION_CONFIG" | awk '{print $1}') == "$EXPECTED_CONFIG_SHA" ]] || \
    fatal "evaluation config SHA mismatch"
if [[ -n ${ROUND_DAGGER_EVALUATOR_CONFIG_SHA256:-} ]]; then
    [[ $ROUND_DAGGER_EVALUATOR_CONFIG_SHA256 == "$EXPECTED_CONFIG_SHA" ]] || \
        fatal "orchestrator evaluator config SHA differs from paired contract"
fi
ACTUAL_EVALUATOR_SHA=$("$PYTHON_BIN" - "$REPOSITORY_ROOT" "$GR00T_ROOT" <<'PY'
import sys
from round_dagger.evaluation import evaluator_source_bundle_sha256

print(evaluator_source_bundle_sha256(sys.argv[1], sys.argv[2]))
PY
)
[[ $ACTUAL_EVALUATOR_SHA == "$EXPECTED_EVALUATOR_SHA" ]] || \
    fatal "evaluator source-bundle SHA mismatch"
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

while IFS=$'\t' read -r ROLE SEED CHECKPOINT_REL CHECKPOINT_SHA RUNTIME_POLICY_SHA; do
    [[ $ROLE == parent || $ROLE == candidate ]] || fatal "invalid role from contract"
    [[ $SEED =~ ^[0-9]+$ ]] || fatal "invalid seed from contract"
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
    STAGED="$RESULTS_DIR/stage/$ROLE/checkpoint.pt"
    BEFORE="$RESULT_DIR/renderer_before.json"
    AFTER="$RESULT_DIR/renderer_after.json"
    METRICS="$RESULT_DIR/metrics.json"
    OUTCOME="$RESULT_DIR/outcome.json"

    SANITIZE=("$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/eval/fixed_seed_gate.py" sanitize-checkpoint
        --checkpoint "$CHECKPOINT" --expected-sha256 "$CHECKPOINT_SHA"
        --output "$STAGED" --provenance "$RESULTS_DIR/stage/$ROLE/provenance.json")
    CONFIG_COPY=(install -m 0444 "$EVALUATION_CONFIG" "$RESULTS_DIR/stage/$ROLE/config.yaml")
    VERIFY_STAGE=("$PYTHON_BIN" - "$CHECKPOINT" "$CHECKPOINT_SHA" "$STAGED"
        "$RESULTS_DIR/stage/$ROLE/provenance.json"
        "$RESULTS_DIR/stage/$ROLE/config.yaml" "$EXPECTED_CONFIG_SHA")
    PROBE_BEFORE=("$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/scaling/image3f_probe.py"
        --endpoint-dir "$ENDPOINT_DIR" --expected "$EXPECTED_RENDERERS"
        --required-reference-lag 256 --require-reference-history
        --expected-output-views head_color,hand_left,hand_right
        --expected-output-layout vertical --wait-timeout-s 120 --connect-timeout-s 10
        --write-report "$BEFORE" --json)
    EVAL=(env IMAGE_DAGGER_PAIRED_EVAL=1
        IMAGE_DAGGER_EVAL_RELABEL=1
        IMAGE3F_RUN_ID="$IMAGE3F_EVAL_RUN_ID"
        IMAGE_DAGGER_PAIRED_EVAL_CONTRACT="$CONTRACT"
        IMAGE_DAGGER_EXPECT_RUNTIME_POLICY_SHA256="$RUNTIME_POLICY_SHA"
        "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/eval/eval_agent_metrics_adapter.py"
        --evaluator "$EVALUATOR" --metrics-output "$METRICS" --
        +checkpoint="$STAGED" +num_envs="$NUM_ENVS" +headless=true +seed="$SEED" +eval_output_dir="$RESULT_DIR"
        algo.config.eval.eval_num_envs_episodes=true
        algo.config.eval.num_eval_episodes="$NUM_ENVS"
        algo.config.eval.save_videos=false algo.config.eval.save_trajectories=false
        algo.config.init_at_random_ep_len=false
        algo.config.videogen.raytraced_mix_prob=0.0
        algo.config.videogen.image3f_endpoint_files="$ENDPOINT_LIST"
        'algo.config.videogen.view_names=[head_color,hand_left,hand_right]'
        ++algo.config.videogen.reference_lag=256
        ++algo.config.videogen.primary_view=head_color
        ++algo.config.videogen.seed="$SEED"
        'simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]'
        ++env.config.image_dagger_policy_view_count=3
        ++algo.config.eval.training_enabled=false
        +simulator.config.render_results=false +use_wandb=false
        hydra.run.dir="$RESULT_DIR/hydra" +output_dir="$RESULT_DIR/output")
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
        print_command "${SANITIZE[@]}"
        print_command "${CONFIG_COPY[@]}"
        print_command "${PROBE_BEFORE[@]}"
        print_command "${EVAL[@]}"
        print_command "${PROBE_AFTER[@]}"
        print_command "${ATTEST[@]}"
        print_command "${PUBLISH[@]}"
        print_command "${VERIFY[@]}"
    else
        mkdir -p "$ATTEMPT_BASE"
        mkdir "$RESULT_DIR" || fatal "attempt allocation collision at $RESULT_DIR"
        # Stage once.  Retries validate all staged bytes and their source
        # provenance rather than overwriting them.
        STAGE_ROLE_DIR="$RESULTS_DIR/stage/$ROLE"
        STAGE_PROVENANCE="$STAGE_ROLE_DIR/provenance.json"
        STAGE_CONFIG="$STAGE_ROLE_DIR/config.yaml"
        if [[ -e $STAGE_ROLE_DIR ]]; then
            [[ -d $STAGE_ROLE_DIR && ! -L $STAGE_ROLE_DIR ]] || \
                fatal "unsafe staged role directory"
            [[ -z $(find "$STAGE_ROLE_DIR" -maxdepth 1 -type l -print -quit) ]] || \
                fatal "staged role directory contains a symlink"
        fi
        if [[ -e $STAGED && -e $STAGE_PROVENANCE && -e $STAGE_CONFIG ]]; then
            "${VERIFY_STAGE[@]}" <<'PY' || fatal "staged evaluation inputs changed"
import hashlib, json, os, pathlib, sys

source, source_sha, staged, provenance, config, config_sha = map(pathlib.Path, sys.argv[1:])
source_sha = str(source_sha)
config_sha = str(config_sha)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
for path in (staged, provenance, config):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe/missing staged input: {path}")
row = json.loads(provenance.read_text(encoding="utf-8"))
if row.get("source") != str(source.resolve()) or row.get("source_sha256") != source_sha:
    raise SystemExit("staged checkpoint source provenance mismatch")
if row.get("output") != str(staged) or row.get("output_sha256") != digest(staged):
    raise SystemExit("staged checkpoint output provenance mismatch")
if row.get("environment_state_restored") is not False:
    raise SystemExit("staged checkpoint restored environment state")
if digest(source) != source_sha or digest(config) != config_sha:
    raise SystemExit("staged source/config digest mismatch")
PY
        else
            if [[ -e $STAGED || -e $STAGE_PROVENANCE || -e $STAGE_CONFIG ]]; then
                STAGE_PARTIAL_BASE="$RESULTS_DIR/stage_partial/$ROLE"
                STAGE_PARTIAL_ATTEMPT=$(next_attempt "$STAGE_PARTIAL_BASE")
                STAGE_PARTIAL=$(printf '%s/attempt_%06d' \
                    "$STAGE_PARTIAL_BASE" "$STAGE_PARTIAL_ATTEMPT")
                mkdir -p "$STAGE_PARTIAL_BASE"
                [[ ! -e $STAGE_PARTIAL ]] || fatal "stage partial allocation collision"
                mv -- "$STAGE_ROLE_DIR" "$STAGE_PARTIAL"
                echo "PAIRED_EVALUATION_STAGE_PARTIAL_PRESERVED role=$ROLE attempt=$STAGE_PARTIAL_ATTEMPT"
            fi
            mkdir -p "$STAGE_ROLE_DIR"
            "${SANITIZE[@]}"
            "${CONFIG_COPY[@]}"
        fi
        "${PROBE_BEFORE[@]}" >/dev/null
        "${EVAL[@]}"
        "${PROBE_AFTER[@]}" >/dev/null
        "${ATTEST[@]}" >/dev/null
        "${PUBLISH[@]}" >/dev/null
        "${VERIFY[@]}" >/dev/null
        echo "PAIRED_EVALUATION_CASE_COMPLETE role=$ROLE seed=$SEED attempt=$ATTEMPT run_id=$IMAGE3F_EVAL_RUN_ID"
    fi
done < <("$PYTHON_BIN" -m round_dagger.evaluation list-cases --root "$ROOT" --contract "$CONTRACT")

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
