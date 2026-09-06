#!/usr/bin/env bash
# Compatibility name retained to prevent accidental execution of the archived
# 512-step custom collector. Production dispatches to the Raven-exact collector.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
exec "$ROOT/bc_collection/run_raven_bc_collection_8gpu.sh" "$@"
