#!/usr/bin/env bash
# Workload-only pod: never starts Codex or reserves GPUs for an agent.
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v sshd >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends openssh-server ca-certificates rsync curl tmux
fi
install -d -m 0700 /root/.ssh
install -d -m 0755 /run/sshd
: "${ROBORENDER_SMOKE_PUBLIC_KEYS:?public SSH keys required}"
printf '%s\n' "$ROBORENDER_SMOKE_PUBLIC_KEYS" > /root/.ssh/authorized_keys
chmod 0600 /root/.ssh/authorized_keys
unset ROBORENDER_SMOKE_PUBLIC_KEYS
ssh-keygen -A
/usr/sbin/sshd -t
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
# Publish only the public host key through shared storage, so management can
# pin this exact workload without disabling SSH host authentication.
[[ ${ROBORENDER_SMOKE_ID:-} =~ ^leo-roborender-beta-smoke-[a-z0-9-]+$ ]] || exit 2
SMOKE_EVIDENCE=/workspace/project/roborender/runtime/migrations/alpha-to-beta-20261008/h200/$ROBORENDER_SMOKE_ID
install -d -m 0700 "$SMOKE_EVIDENCE"
install -m 0444 /etc/ssh/ssh_host_ed25519_key.pub "$SMOKE_EVIDENCE/ssh_host_ed25519_key.pub"
exec /usr/sbin/sshd -D -e -p 2222 \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o PermitRootLogin=prohibit-password -o AllowTcpForwarding=yes \
  -o GatewayPorts=no -o X11Forwarding=no -o ClientAliveInterval=30 \
  -o ClientAliveCountMax=3
