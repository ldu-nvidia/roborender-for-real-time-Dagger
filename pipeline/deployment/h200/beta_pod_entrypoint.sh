#!/usr/bin/env bash
# Workload-only beta pod. No agent, automatic training, or unapproved GPU shape.
set -Eeuo pipefail
: "${ROBORENDER_BETA_RUN_ID:?}" "${ROBORENDER_BETA_ROUND_ID:?}" "${ROBORENDER_BETA_PUBLIC_KEYS:?}"
[[ $ROBORENDER_BETA_RUN_ID =~ ^dagger-(beta|leo)-[a-z0-9-]+$ ]] || exit 2
[[ $ROBORENDER_BETA_ROUND_ID =~ ^[1-4]$ ]] || exit 2
export DEBIAN_FRONTEND=noninteractive
if ! command -v sshd >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends openssh-server ca-certificates rsync curl
fi
install -d -m 0700 /root/.ssh
install -d -m 0755 /run/sshd
printf '%s\n' "$ROBORENDER_BETA_PUBLIC_KEYS" > /root/.ssh/authorized_keys
chmod 0600 /root/.ssh/authorized_keys
unset ROBORENDER_BETA_PUBLIC_KEYS
ssh-keygen -A
/usr/sbin/sshd -t
EVIDENCE=/workspace/project/roborender/runtime/beta-pods/$ROBORENDER_BETA_RUN_ID/round-$ROBORENDER_BETA_ROUND_ID
if [[ $ROBORENDER_BETA_RUN_ID == dagger-leo-* ]]; then
  EVIDENCE=/workspace/project/roborender/runtime/leo-pods/$ROBORENDER_BETA_RUN_ID/round-$ROBORENDER_BETA_ROUND_ID
fi
if [[ -n ${ROBORENDER_INFRASTRUCTURE_ATTEMPT:-} ]]; then
  [[ $ROBORENDER_INFRASTRUCTURE_ATTEMPT =~ ^[a-f0-9]{32}$ ]] || exit 2
  EVIDENCE=$EVIDENCE/recovery-$ROBORENDER_INFRASTRUCTURE_ATTEMPT
fi
install -d -m 0700 "$EVIDENCE"
install -m 0444 /etc/ssh/ssh_host_ed25519_key.pub "$EVIDENCE/host-key.pub"
exec /usr/sbin/sshd -D -e -p 2222 \
  -o "SetEnv=ROBORENDER_BETA_RUN_ID=$ROBORENDER_BETA_RUN_ID ROBORENDER_BETA_ROUND_ID=$ROBORENDER_BETA_ROUND_ID" \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o PermitRootLogin=prohibit-password -o AllowTcpForwarding=yes \
  -o GatewayPorts=no -o X11Forwarding=no -o ClientAliveInterval=30 \
  -o ClientAliveCountMax=3
