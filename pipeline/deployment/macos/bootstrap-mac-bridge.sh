#!/usr/bin/env bash
# Private bootstrap for install_bridge.py's supervised LaunchAgent.
# Only scoped public-key authorization and loopback forwards; no GPU workload.
# Each attempt uses private ControlPath sockets and never reuses a user's master.
set -Eeuo pipefail

stage=startup
trap 'rc=$?; printf "Bridge stopped during: %s (exit %s).\n" "$stage" "$rc" >&2; exit "$rc"' ERR
announce() { stage=$1; printf '\n[%s]\n' "$stage" >&2; }

if [[ ${1:-} == --help ]]; then
  printf '%s\n' 'Run on your Mac: bash bootstrap-mac-bridge.sh' \
    'Requires already-trusted SSH aliases cosmos-h200, rtx-6k-alpha, rtx-6k-beta.' \
    'Adds one restricted public key on alpha/beta; pins host keys on CPU pod;' \
    'opens CPU localhost:22021 -> alpha:22 and localhost:22022 -> beta:22.' \
    'Ctrl-C closes the tunnels. The scoped authorization is retained for cleanup.'
  exit 0
fi
[[ $# == 0 ]] || { printf 'Unknown argument\n' >&2; exit 2; }
[[ $(uname -s) == Darwin ]] || { printf 'Run this bootstrap on your Mac.\n' >&2; exit 2; }

readonly CONTROL=cosmos-h200
readonly ALPHA=rtx-6k-alpha
readonly BETA=rtx-6k-beta
readonly EXPECTED_BETA=465d6b0-lcedt
readonly EXPECTED_CONTROL=leo-cosmos-h200-access-8b47858b5-hzzsf
readonly PINNER=/workspace/project/roborender/runtime/migrations/alpha-to-beta-20261008/pin_bridge_hostkeys.py
# Keep the configured proxy, identity and pinned host keys, but never reuse a
# possibly stale user master. %C separates the three hosts inside our own
# mode-0700 temporary directory. The Python owner removes only these masters.
readonly bridge_control_path="${ROBORENDER_BRIDGE_CONTROL_PATH:?Run the bridge installer supervisor}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes
  -o "ControlPath=$bridge_control_path" -o ControlMaster=auto -o ControlPersist=60
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3)

printf 'Checking existing authenticated routes; host-key verification remains enabled.\n'
announce '1/7: SSH rtx-6k-alpha'
alpha_hostname=$(ssh "${ssh_opts[@]}" "$ALPHA" hostname)
printf 'OK: %s\n' "$alpha_hostname"
announce '2/7: SSH rtx-6k-beta'
beta_hostname=$(ssh "${ssh_opts[@]}" "$BETA" hostname)
[[ "$beta_hostname" == "$EXPECTED_BETA" ]] || {
  printf 'Unexpected beta hostname: %s (expected %s)\n' "$beta_hostname" "$EXPECTED_BETA" >&2
  exit 1
}
printf 'OK: %s\n' "$beta_hostname"
announce '3/7: SSH cosmos-h200 and shared migration files'
control_hostname=$(ssh "${ssh_opts[@]}" "$CONTROL" \
  'test -f /workspace/project/roborender/runtime/migrations/alpha-to-beta-20261008/pin_bridge_hostkeys.py && hostname')
[[ "$control_hostname" == "$EXPECTED_CONTROL" ]] || {
  printf 'CPU access pod changed: %s (expected %s); verify before forwarding.\n' "$control_hostname" "$EXPECTED_CONTROL" >&2
  exit 1
}
printf 'OK: %s\n' "$control_hostname"
printf 'Source alpha: %s; destination beta: %s\n' "$alpha_hostname" "$beta_hostname"

# Derive Mac-reachable LAN destinations from the user's SSH config. Require the
# exact known migration IPs so no unrelated host is authorized or forwarded.
announce '4/7: Verify configured destinations and unused bridge ports'
alpha_address=$(ssh -G "$ALPHA" 2>/dev/null | awk '$1=="hostname" {print $2}')
beta_address=$(ssh -G "$BETA" 2>/dev/null | awk '$1=="hostname" {print $2}')
alpha_port=$(ssh -G "$ALPHA" 2>/dev/null | awk '$1=="port" {print $2}')
beta_port=$(ssh -G "$BETA" 2>/dev/null | awk '$1=="port" {print $2}')
[[ "$alpha_address" == 10.110.20.44 && "$alpha_port" == 22 ]] || {
  printf 'Alpha route changed: %s:%s; ask the coordinator to verify it.\n' "$alpha_address" "$alpha_port" >&2
  exit 1
}
[[ "$beta_address" == 10.110.21.136 && "$beta_port" == 22 ]] || {
  printf 'Beta wired route changed: %s:%s; ask the coordinator to verify it.\n' "$beta_address" "$beta_port" >&2
  exit 1
}
listeners=$(ssh "${ssh_opts[@]}" "$CONTROL" \
  "ss -H -lnt '( sport = :22021 or sport = :22022 )'")
[[ -z "$listeners" ]] || {
  printf 'Bridge ports already have listeners; leaving them untouched.\n%s\n' "$listeners" >&2
  exit 1
}

# Read only PUBLIC host keys over the already-authenticated Mac SSH sessions.
announce '5/7: Pin authenticated public host keys'
alpha_hostkey=$(ssh "${ssh_opts[@]}" "$ALPHA" 'cat /etc/ssh/ssh_host_ed25519_key.pub')
beta_hostkey=$(ssh "${ssh_opts[@]}" "$BETA" 'cat /etc/ssh/ssh_host_ed25519_key.pub')
printf '%s\n%s\n' "$alpha_hostkey" "$beta_hostkey" |
  ssh "${ssh_opts[@]}" "$CONTROL" python3 "$PINNER"

announce '6/7: Authorize the existing restricted migration key'
for target in "$ALPHA" "$BETA"; do
  stage="6/7: Restricted key on $target"
  printf 'Authorizing dedicated restricted migration key on %s\n' "$target"
  ssh "${ssh_opts[@]}" "$target" 'bash -se' <<'REMOTE'
set -euo pipefail
[[ $(id -un) == ldu ]] || { printf 'Expected user ldu\n' >&2; exit 1; }
key_blob=AAAAC3NzaC1lZDI1NTE5AAAAIPL7mCJdoVQOYTY3SHHY6BMSDLZtg+yjZ+xnKqDLzdIr
line="restrict ssh-ed25519 $key_blob roborender-alpha-beta-migration-20261008"
ssh_dir="$HOME/.ssh"
authorized="$ssh_dir/authorized_keys"
[[ ! -L "$authorized" ]] || { printf 'Refusing authorized_keys symlink\n' >&2; exit 1; }
umask 077
mkdir -p "$ssh_dir"
if [[ -e "$authorized" ]] && grep -Fqx -- "$line" "$authorized"; then
  printf 'Scoped key already installed.\n'
elif [[ -e "$authorized" ]] && grep -Fq -- "$key_blob" "$authorized"; then
  printf 'Same key has unexpected options; refusing to alter existing entry.\n' >&2
  exit 1
else
  # Leading newline also handles an existing authorized_keys without final LF.
  printf '\n%s\n' "$line" >> "$authorized"
  printf 'Scoped key added; existing keys preserved.\n'
fi
REMOTE
done

announce '7/7: Open loopback-only reverse forwards'
alpha_forward="127.0.0.1:22021:${alpha_address}:${alpha_port}"
beta_forward="127.0.0.1:22022:${beta_address}:${beta_port}"
active_forwards=()
cleanup_forwards() {
  local forward
  [[ ${#active_forwards[@]} -gt 0 ]] || return 0
  for forward in "${active_forwards[@]}"; do
    ssh "${ssh_opts[@]}" -O cancel -R "$forward" "$CONTROL" >/dev/null 2>&1 || true
  done
}
trap cleanup_forwards EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for forward in "$alpha_forward" "$beta_forward"; do
  ssh "${ssh_opts[@]}" -o ExitOnForwardFailure=yes -O forward -R "$forward" "$CONTROL"
  active_forwards+=("$forward")
done
printf '\nBridge forwards installed on the private SSH connection; awaiting end-to-end health check.\n'
# Local master liveness is only an early failure signal. The Python supervisor
# independently runs authenticated round trips through both remote ports.
while sleep 20; do
  stage='monitor private SSH connection'
  ssh "${ssh_opts[@]}" -O check "$CONTROL" >/dev/null 2>&1
done
