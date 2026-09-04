#!/usr/bin/env bash
# Canonical production entry point: RTX cohorts + reserved Lepton 8xH200 learner.
#
# All campaign, source-seal, and backend identities are explicit arguments to
# the fail-closed five-round driver. The legacy packed OCI workflow remains at
# submit_oci_packed.sh and is never selected as a fallback from this wrapper.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
DRIVER="$ROOT/scripts/round_dagger/run_five_round_production_campaign.py"
PYTHON=${ROUND_DAGGER_CONTROL_PYTHON:-/mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python}

for argument in "$@"; do
    case "$argument" in
        --learner-backend|--learner-backend=*)
            echo "FATAL: canonical production fixes learner backend to lepton_job_v1" >&2
            exit 2
            ;;
    esac
done

[[ -f $DRIVER && ! -L $DRIVER ]] || {
    echo "FATAL: canonical round-based production driver is unavailable" >&2
    exit 2
}
PYTHON_REAL=$(realpath -e -- "$PYTHON") || {
    echo "FATAL: control Python cannot be resolved" >&2
    exit 2
}
[[ -f $PYTHON_REAL && -x $PYTHON_REAL ]] || {
    echo "FATAL: control Python is not executable" >&2
    exit 2
}

exec "$PYTHON" "$DRIVER" "$@" --learner-backend lepton_job_v1
