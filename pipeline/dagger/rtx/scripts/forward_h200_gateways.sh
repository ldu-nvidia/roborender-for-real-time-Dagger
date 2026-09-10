#!/usr/bin/env bash
# Foreground-only, loopback-only forwards for eight sharded data gateways.
# Each gateway owns an independent SSH connection so one stalled transport does
# not head-of-line block the other seven.  This does not inspect, restart, or
# stop the independent 18610-18614 tunnel.
set -Eeuo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly KEY=${DAGGER_H200_SSH_KEY:-/home/ldu/.ssh/id_ed25519_oci}
readonly PROXY_PYTHON=${DAGGER_H200_PROXY_PYTHON:-/home/ldu/.local/share/pipx/venvs/leptonai/bin/python}
readonly PROXY_MODULE_DIR=${DAGGER_H200_PROXY_MODULE_DIR:-/home/ldu/.ssh}
readonly PROXY_TARGET=${DAGGER_H200_PROXY_TARGET:-ldu-cosmos-h200-dev-2}
readonly PROXY_PORT=${DAGGER_H200_PROXY_PORT:-2222}
readonly DESTINATION=${DAGGER_H200_SSH_DESTINATION:-root@cosmos-h200.lepton}
readonly LOCK=${DAGGER_RTX_GATEWAY_LOCK:-$ROOT/runtime/h200-forward-gateways.lock}
readonly GATEWAY_PORTS_CSV=${DAGGER_H200_GATEWAY_PORTS:-18620,18621,18622,18623,18624,18625,18626,18627}

IFS=',' read -r -a gateway_ports <<<"$GATEWAY_PORTS_CSV"
[[ ${#gateway_ports[@]} -eq 8 ]] || {
  echo "DAGGER_H200_GATEWAY_PORTS must contain exactly eight ports" >&2
  exit 2
}
declare -A seen_gateway_ports=()
for port in "${gateway_ports[@]}"; do
  [[ $port =~ ^[1-9][0-9]{3,4}$ ]] || {
    echo "gateway ports must be decimal integers in [1024,65535]" >&2
    exit 2
  }
  (( 10#$port >= 1024 && 10#$port <= 65535 )) || {
    echo "gateway ports must be decimal integers in [1024,65535]" >&2
    exit 2
  }
  [[ -z ${seen_gateway_ports[$port]:-} ]] || {
    echo "gateway ports must be unique" >&2
    exit 2
  }
  seen_gateway_ports[$port]=1
done

[[ -r "$KEY" ]] || { echo "missing configured SSH identity: $KEY" >&2; exit 2; }
[[ -x "$PROXY_PYTHON" ]] || { echo "missing configured Lepton proxy Python" >&2; exit 2; }
install -d -m 0700 "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "the dedicated gateway forward is already active" >&2; exit 3; }

common_args=(
  -NT
  -i "$KEY"
  -o IdentitiesOnly=yes
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
  -o TCPKeepAlive=yes
  -o ControlMaster=no
  -o ControlPath=none
  -o "ProxyCommand=$PROXY_PYTHON -c 'import sys; sys.path.insert(0,\"$PROXY_MODULE_DIR\"); import lepton_ssh_proxy as p; exec(p.remote_python_source())' $PROXY_TARGET $PROXY_PORT"
)
pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

for port in "${gateway_ports[@]}"; do
  ssh "${common_args[@]}" \
    -L "127.0.0.1:${port}:127.0.0.1:${port}" \
    "$DESTINATION" &
  pids+=("$!")
done

# Any tunnel exit invalidates the eight-way transport.  Tear down only the
# seven sibling PIDs created above and fail closed.
set +e
wait -n "${pids[@]}"
status=$?
set -e
cleanup
exit "$status"
