#!/usr/bin/env bash
# Recover one held TRAINED round without rerunning macro-round telemetry.
set -Eeuo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

[[ ${ROUND_DAGGER_RECOVERY_OPERATOR_ACK:-} == legacy-orchestrator-stopped ]] || \
    fatal "set ROUND_DAGGER_RECOVERY_OPERATOR_ACK=legacy-orchestrator-stopped only after the old coordinator has exited"

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
export PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CAMPAIGN_ROOT=${ROUND_DAGGER_CAMPAIGN_ROOT:?missing ROUND_DAGGER_CAMPAIGN_ROOT}
GR00T_SOURCE_ROOT=${GR00T_ROOT:?missing GR00T_ROOT}
HOLD_ATTESTATION=${ROUND_DAGGER_PROMOTION_HOLD_ATTESTATION:?missing ROUND_DAGGER_PROMOTION_HOLD_ATTESTATION}
HOLD_ATTESTATION_SHA=${ROUND_DAGGER_PROMOTION_HOLD_ATTESTATION_SHA256:?missing ROUND_DAGGER_PROMOTION_HOLD_ATTESTATION_SHA256}
SCENARIO_CASES=${ROUND_DAGGER_EVALUATION_SCENARIOS:?missing ROUND_DAGGER_EVALUATION_SCENARIOS}
THRESHOLDS=${ROUND_DAGGER_EVALUATION_THRESHOLDS:?missing ROUND_DAGGER_EVALUATION_THRESHOLDS}
EVALUATION_CONFIG=${ROUND_DAGGER_EVALUATION_CONFIG:?missing ROUND_DAGGER_EVALUATION_CONFIG}
RENDERER_RUNTIME_SHA=${ROUND_DAGGER_RENDERER_RUNTIME_SHA256:?missing ROUND_DAGGER_RENDERER_RUNTIME_SHA256}
EXPECTED_RENDERERS=${ROUND_DAGGER_EXPECTED_RENDERERS:?missing ROUND_DAGGER_EXPECTED_RENDERERS}
LEARNER_RECOVERY_RECEIPT=${ROUND_DAGGER_LEARNER_RECOVERY_RECEIPT:?missing ROUND_DAGGER_LEARNER_RECOVERY_RECEIPT}
LEARNER_RECOVERY_RECEIPT_SHA=${ROUND_DAGGER_LEARNER_RECOVERY_RECEIPT_SHA256:?missing ROUND_DAGGER_LEARNER_RECOVERY_RECEIPT_SHA256}
CONTROL_PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:-python3}
HOOK="$REPOSITORY_ROOT/scripts/round_dagger/run_production_paired_evaluation_hook.sh"
[[ -x $HOOK && ! -L $HOOK ]] || fatal "hardened evaluator hook is missing or unsafe"

if [[ $CONTROL_PYTHON == */* ]]; then
    [[ $CONTROL_PYTHON == /* && -x $CONTROL_PYTHON ]] || \
        fatal "control Python is missing or unsafe"
else
    CONTROL_PYTHON=$(command -v "$CONTROL_PYTHON") || \
        fatal "control Python is unavailable"
fi

PREPARED=$(
    "$CONTROL_PYTHON" -m round_dagger.evaluation_recovery prepare \
        --root "$CAMPAIGN_ROOT" \
        --repository-root "$REPOSITORY_ROOT" \
        --gr00t-root "$GR00T_SOURCE_ROOT" \
        --hold-attestation "$HOLD_ATTESTATION" \
        --expected-hold-attestation-sha256 "$HOLD_ATTESTATION_SHA" \
        --scenario-cases "$SCENARIO_CASES" \
        --thresholds "$THRESHOLDS" \
        --evaluation-config "$EVALUATION_CONFIG" \
        --renderer-runtime-sha256 "$RENDERER_RUNTIME_SHA" \
        --expected-renderer-count "$EXPECTED_RENDERERS" \
        --learner-recovery-receipt "$LEARNER_RECOVERY_RECEIPT" \
        --expected-learner-recovery-receipt-sha256 "$LEARNER_RECOVERY_RECEIPT_SHA"
) || fatal "hardened recovery preparation failed"
mapfile -t RECOVERY_FIELDS < <(
    printf '%s' "$PREPARED" | "$CONTROL_PYTHON" -c \
        'import json,sys; d=json.load(sys.stdin); print(d["recovery_id"]); print(d["plan_path"]); print(d["plan_sha256"]); print(d["evaluator_result_path"])'
)
[[ ${#RECOVERY_FIELDS[@]} -eq 4 ]] || fatal "cannot parse hardened recovery plan"
export ROUND_DAGGER_EVALUATION_RECOVERY_ID=${RECOVERY_FIELDS[0]}
PLAN="$CAMPAIGN_ROOT/${RECOVERY_FIELDS[1]}"
PLAN_SHA=${RECOVERY_FIELDS[2]}
export ROUND_DAGGER_EVALUATOR_RESULT="$CAMPAIGN_ROOT/${RECOVERY_FIELDS[3]}"

bash "$HOOK" || fatal "hardened paired evaluation failed"
"$CONTROL_PYTHON" -m round_dagger.evaluation_recovery finalize \
    --root "$CAMPAIGN_ROOT" \
    --plan "$PLAN" \
    --expected-plan-sha256 "$PLAN_SHA" \
    --repository-root "$REPOSITORY_ROOT" \
    --gr00t-root "$GR00T_SOURCE_ROOT" || \
    fatal "hardened recovery verification/promotion failed"
