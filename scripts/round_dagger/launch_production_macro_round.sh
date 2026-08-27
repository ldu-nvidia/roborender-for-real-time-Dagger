#!/usr/bin/env bash
# Validate the immutable 256x512 production deployment, then run/resume exactly
# one strict macro-round.  This launcher never initializes a campaign and never
# carries a W&B credential in argv, a JSON config, or its printed dry-run plan.
set -Eeuo pipefail

fatal() { echo "production macro-round launcher failed: $*" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
VALIDATOR="$ROOT/scripts/round_dagger/production_launch.py"
ORCHESTRATOR="$ROOT/scripts/round_dagger/orchestrate_macro_round.py"
CAMPAIGN_CONTRACT="$ROOT/configs/round_dagger/production_campaign_256x512.json"
LEARNER_CONTRACT="$ROOT/configs/round_dagger/production_learner_8xh100_1200.json"
PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:-/mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python}
DRY_RUN=${DRY_RUN:-0}
ORCHESTRATION_CONFIG=
SEALED_SOURCE_ROOT=
SEALED_SOURCE_INVENTORY_SHA256=
EVALUATION_SEALED_SOURCE_ROOT=
EVALUATION_SEALED_SOURCE_INVENTORY_SHA256=

usage() {
    cat >&2 <<'EOF'
usage: launch_production_macro_round.sh \
  --orchestration-config ABSOLUTE_CONFIG_JSON \
  --sealed-source-root ABSOLUTE_OCI_SOURCE_ROOT \
  --evaluation-sealed-source-root ABSOLUTE_RTX_SOURCE_ROOT \
  --evaluation-sealed-source-inventory-sha256 LOWERCASE_SHA256 \
  --sealed-source-inventory-sha256 LOWERCASE_SHA256 [--dry-run]
EOF
    exit 2
}

while (($#)); do
    case "$1" in
        --orchestration-config)
            (($# >= 2)) || usage
            ORCHESTRATION_CONFIG=$2
            shift 2
            ;;
        --sealed-source-root)
            (($# >= 2)) || usage
            SEALED_SOURCE_ROOT=$2
            shift 2
            ;;
        --sealed-source-inventory-sha256)
            (($# >= 2)) || usage
            SEALED_SOURCE_INVENTORY_SHA256=$2
            shift 2
            ;;
        --evaluation-sealed-source-root)
            (($# >= 2)) || usage
            EVALUATION_SEALED_SOURCE_ROOT=$2
            shift 2
            ;;
        --evaluation-sealed-source-inventory-sha256)
            (($# >= 2)) || usage
            EVALUATION_SEALED_SOURCE_INVENTORY_SHA256=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help) usage ;;
        *) fatal "unknown argument: $1" ;;
    esac
done

[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || fatal "DRY_RUN must be exactly 0 or 1"
[[ $ORCHESTRATION_CONFIG == /* ]] || fatal "--orchestration-config must be absolute"
[[ $SEALED_SOURCE_ROOT == /* ]] || fatal "--sealed-source-root must be absolute"
[[ $EVALUATION_SEALED_SOURCE_ROOT == /* ]] || \
    fatal "--evaluation-sealed-source-root must be absolute"
[[ -d $EVALUATION_SEALED_SOURCE_ROOT && ! -L $EVALUATION_SEALED_SOURCE_ROOT ]] || \
    fatal "--evaluation-sealed-source-root is missing or unsafe"
[[ $EVALUATION_SEALED_SOURCE_INVENTORY_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
    fatal "--evaluation-sealed-source-inventory-sha256 must be lowercase SHA-256"
[[ $SEALED_SOURCE_INVENTORY_SHA256 =~ ^[0-9a-f]{64}$ ]] || \
    fatal "--sealed-source-inventory-sha256 must be lowercase SHA-256"
[[ -f $ORCHESTRATION_CONFIG && ! -L $ORCHESTRATION_CONFIG ]] || \
    fatal "strict orchestration config is missing or unsafe"
for path in "$VALIDATOR" "$ORCHESTRATOR" "$CAMPAIGN_CONTRACT" "$LEARNER_CONTRACT"; do
    [[ -f $path && ! -L $path ]] || fatal "required production source artifact is missing or unsafe: $path"
done
PYTHON_REAL=$(realpath -e -- "$PYTHON") || fatal "cannot resolve validated control Python"
[[ -f $PYTHON_REAL && ! -L $PYTHON_REAL && -x $PYTHON_REAL ]] || \
    fatal "validated control Python is unavailable"
# Preserve the venv entry point for execution.  Calling its resolved base
# interpreter path would discard the venv's sys.prefix and site-packages.

# Local actor telemetry and the OCI learner both authenticate from protected
# netrc state.  Do not allow a caller's ambient key/token to enter child
# process environments or a process listing.
unset WANDB_API_KEY WANDB_ACCESS_TOKEN WANDB_IDENTITY_TOKEN WANDB_OIDC_TOKEN

ENV_HANDOFF=$(mktemp "${TMPDIR:-/tmp}/roborender-production-env.XXXXXX") || \
    fatal "cannot create private environment handoff"
chmod 0600 "$ENV_HANDOFF"
cleanup() { rm -f -- "$ENV_HANDOFF"; }
trap cleanup EXIT HUP INT TERM

PLAN=$(
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" "$VALIDATOR" \
        --orchestration-config "$ORCHESTRATION_CONFIG" \
        --campaign-contract "$CAMPAIGN_CONTRACT" \
        --learner-contract "$LEARNER_CONTRACT" \
        --sealed-source-root "$SEALED_SOURCE_ROOT" \
        --evaluation-sealed-source-root "$EVALUATION_SEALED_SOURCE_ROOT" \
        --evaluation-sealed-source-inventory-sha256 \
            "$EVALUATION_SEALED_SOURCE_INVENTORY_SHA256" \
        --sealed-source-inventory-sha256 "$SEALED_SOURCE_INVENTORY_SHA256" \
        --env-output "$ENV_HANDOFF"
) || fatal "fail-closed preflight rejected the launch"

ENVIRONMENT_COUNT=0
while IFS= read -r -d '' name; do
    IFS= read -r -d '' value || fatal "validator emitted a truncated environment handoff"
    [[ $name =~ ^[A-Z][A-Z0-9_]*$ ]] || fatal "validator emitted an unsafe environment name"
    export "$name=$value"
    ENVIRONMENT_COUNT=$((ENVIRONMENT_COUNT + 1))
done <"$ENV_HANDOFF"
((ENVIRONMENT_COUNT > 0)) || fatal "validator emitted an empty environment handoff"
[[ ${ROUND_DAGGER_LAUNCH_ORCHESTRATION_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || \
    fatal "validator omitted the orchestration config identity"
[[ $(sha256sum -- "$ORCHESTRATION_CONFIG" | awk '{print $1}') == \
   "$ROUND_DAGGER_LAUNCH_ORCHESTRATION_SHA256" ]] || \
    fatal "orchestration config changed after preflight"

if [[ $DRY_RUN == 1 ]]; then
    printf '%s\n' "$PLAN"
    echo "PRODUCTION_MACRO_ROUND_DRY_RUN_OK no_campaign_mutation=1 no_submission=1"
    exit 0
fi

# The validator has already bound this exact config to the durable campaign,
# active policy, source inventory, learner contract, and static deployment.
# Replace the launcher so signals and exit status belong to the coordinator.
trap - EXIT HUP INT TERM
rm -f -- "$ENV_HANDOFF"
exec "$PYTHON" "$ORCHESTRATOR" --config "$ORCHESTRATION_CONFIG"
