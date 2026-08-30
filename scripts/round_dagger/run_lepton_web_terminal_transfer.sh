#!/usr/bin/env bash
# Invoke the authenticated web-terminal transport with the pinned Lepton SDK.
set -Eeuo pipefail

fatal() { echo "Lepton transfer wrapper failed: $*" >&2; exit 2; }

source_root=${ROUND_DAGGER_TRUSTED_HOOK_SOURCE_ROOT:-}
[[ $source_root == /* && -d $source_root ]] || fatal "trusted source root is missing"
transfer="$source_root/scripts/round_dagger/lepton_web_terminal_transfer.py"
[[ -f $transfer && ! -L $transfer ]] || fatal "sealed transfer program is unsafe"

python=${ROUND_DAGGER_LEPTON_SDK_PYTHON:-}
[[ $python == /* && -x $python ]] || fatal "explicit Lepton SDK Python is missing"
python_real=$(realpath -e -- "$python") || fatal "Lepton SDK Python cannot be resolved"
[[ -f $python_real && -x $python_real ]] || fatal "resolved Lepton SDK Python is unsafe"

exec "$python" "$transfer" "$@"
