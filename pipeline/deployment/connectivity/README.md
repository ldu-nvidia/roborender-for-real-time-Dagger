# Alpha-free Beta / Cosmos connectivity

Authentication and connectivity are different gates. `lep login` on Beta saves
its approved local credential; it does **not** open an SSH route. These scripts
install a Beta-owned user service without copying that credential anywhere.

```text
CPU access pod -> localhost:22032 -> Beta's reverse SSH -> Beta sshd:22
                                      |
Beta -> authenticated TLS cloud shell -> CPU access pod sshd:2222

Beta -> local cloud_proxy.py -> reserved H200 workload sshd:2222
```

No Alpha or Mac is in these runtime paths. The Mac is needed only for one-time
bootstrap if the CPU has no working route to Beta. The management service itself
does not allocate GPUs or start/restart Isaac, a learner or production.

## Components

- `cloud_proxy.py`: uses Beta's approved workspace login and verified TLS to
  reach only the CPU access pod or named RoboRender campaign pods. SSH separately
  pins the destination host key. Tokens stay inside the SDK process.
- `cpu_endpoint.py`: enrolls a dedicated restricted public tunnel key and permits
  only localhost reverse port 22032 plus a forced round-trip health command.
  Existing SSH keys/configs are preserved. The CPU's existing management key
  has a historical migration filename; its use does not contact Alpha.
- `beta_access.py`: user-systemd supervision; 30-second authenticated health
  checks, bounded reconnect, owned-process-only cleanup and a private status
  file. Readiness requires CPU -> Beta `/bin/hostname`, not merely a live PID.
  Health commands reuse the service's private authenticated SSH master; they
  cannot silently create a new cloud connection. A dead tunnel is detected
  without waiting for the next health interval.
- `build_bundle.py`: reproducible source-only zipapp with checked member hashes.
  No credentials, private keys, checkpoints, images or runtime data are bundled.
- `verify_live.py`: authenticated host/service checks and byte-exact binary
  round trips. Its optional reconnect probe interrupts only the exact owned
  SSH child, verifies a replacement, and leaves the supervisor running.

## Bootstrap

Use the already authenticated compatible SDK environment on Beta. Download the
generated `roborender-beta-connect.pyz`, copy it directly from the Mac to Beta,
and run it there:

```bash
scp "$HOME/Downloads/roborender-beta-connect.pyz" rtx-6k-beta:/home/ldu/roborender-beta-connect.pyz
ssh -t rtx-6k-beta '/home/ldu/.local/share/roborender-cloud/venv-compatible/bin/python /home/ldu/roborender-beta-connect.pyz'
```

The installer preserves existing files and refuses conflicting managed files.
It pins both SSH host keys from the authenticated migration. Host-key changes
require explicit reconciliation; it never disables host checking. It attempts
normal-user `loginctl enable-linger ldu` without password prompts. A result of
`persistent_after_logout=false` requires operator help before claiming autonomy.
Cloud login expiry, machine/network outages, and inaccessible user-systemd are
reported failures, not an unconditional always-connected guarantee.

CPU verification after installation:

```bash
ssh -F /workspace/.ssh/roborender_beta_access_config roborender-beta-direct /bin/hostname
python3 pipeline/deployment/connectivity/cpu_endpoint.py health
```

Beta verification:

```bash
systemctl --user status roborender-beta-access.service
cat /home/ldu/.local/share/roborender-beta-access/status.json
```

Stop only this service with `systemctl --user stop roborender-beta-access.service`.
Old Mac migration ports 22021/22022 are untouched. Do not retire a working old
route until this new route and the actual workload endpoints pass live tests.

## Test boundary

`test_connectivity.py` is CPU-only regression coverage of framing, SSH restrictions,
pins, idempotent enrollment, package hashes, and health semantics. It is **not**
evidence of a live cloud connection, reconnect recovery, GPU service handoff,
native DAgger step/update or closed-loop success evaluation. Record those gates
separately after deployment. Never replay uncertain native simulator steps when
a management connection reconnects.

## Live verification, 2026-10-12 23:18 UTC

The r2 service is installed on Beta and passed authenticated CPU/Beta return
traffic without Alpha or the Mac bridge. It is enabled with user linger active.
Three byte-exact round trips passed (4 KiB, 1 MiB, 1 MiB in each direction).
Deliberately terminating its verified SSH child recovered automatically in
**25.57 seconds**, with the same supervisor and a new SSH child. This is one
controlled reconnect test, not an availability or sustained-throughput guarantee.

The earlier r1 reconnect exceeded the 180-second test deadline before recovering;
that failure is not relabeled as a pass. The r2 change reuses the authenticated
master for health checks, detects child exit promptly, and bounds reconnect
backoff. Deployment preserved a rollback copy and did not change credentials.

Evidence on the CPU/shared volume:
`/workspace/project/roborender/runtime/connectivity-validation-20261012/alpha-free-live-r2.json`.
The source CPU gate passed **654 tests**, including 25 connectivity tests.
Read-only W&B authentication also passed; no new online training run was created.

**Still separate:** no H200 was allocated and no native training or success
evaluation ran during this connection test. New-route genuine workload endpoints
and a bounded GPU pipeline check must pass before this evidence can be used for
production admission. Existing checkpoints, datasets and earlier native test
evidence are unchanged.
