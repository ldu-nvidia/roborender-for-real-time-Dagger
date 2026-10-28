# Persistent Mac management bridge

This wraps the versioned, host-key-verified migration bootstrap in a
per-user macOS LaunchAgent. It uses the existing `cosmos-h200`, `rtx-6k-alpha`
and `rtx-6k-beta` aliases. No private keys are copied, no SSH settings are edited,
and occupied ports or unrelated SSH masters are never taken over.

On the Mac, once:

```bash
scp -o ControlMaster=no -o ControlPath=none cosmos-h200:/workspace/project/roborender/pipeline/deployment/macos/install_bridge.py ~/Downloads/roborender-install-bridge.py
python3 ~/Downloads/roborender-install-bridge.py --keep-awake-on-ac
```

After installation, Ctrl-C the previous foreground bridge. The new agent will
acquire the released ports on its next attempt (up to about 30 seconds). Closing
Terminal no longer stops the bridge. It starts again at login and retries after
disconnects. The optional flag prevents idle system sleep on AC power; it does
not prevent lid-close sleep or keep a battery-powered Mac awake.

Version 2 uses fresh private SSH masters, with 15-second server keepalives,
instead of attaching forwarding to a possibly stale user ControlMaster.
Every 30 seconds it executes authenticated CPU-to-alpha/beta hostname round
trips through both forwarded ports. A local `Master running` message or
launchd's `state=running` is **not** readiness. Two failed end-to-end probes
cause only the bridge's own connections to be replaced. Post-authentication
hangs have a 25-second probe deadline and a 180-second bootstrap deadline.
The supervisor never stops an unrelated user master or takes occupied ports.

Inspect `bridge-status.json` in the application directory for `state=ready`
and the latest verified hostnames; healthy log heartbeats are emitted every
five minutes, with immediate error/reconnect messages. Installation remains
one command pair above, including upgrades from version 1. The `scp` bypasses
a stale shared master without closing it. A Mac-side upgrade is required;
changing the server copy alone cannot replace an installed LaunchAgent.

Upgrades with unchanged launch arguments use `kickstart` on the verified owned
registration; they do not race an unconditional bootout/bootstrap pair. When
launch arguments change, the installer waits for confirmed removal and retries
transient bootstrap failures within a bounded budget. It never unloads an
entire user domain or requests root. A sanitized `activation.json` report
(without inherited environment values) is saved under the private log directory
and sent over the existing authenticated CPU route to the ignored migration
diagnostics directory, so failures can be investigated remotely.

Installed files are under `~/Library/Application Support/RoboRenderBridge`;
private logs are under `~/Library/Logs/RoboRenderBridge`.

```bash
python3 "$HOME/Library/Application Support/RoboRenderBridge/install_bridge.py" status
python3 "$HOME/Library/Application Support/RoboRenderBridge/install_bridge.py" stop
```

`stop` unloads only this job and preserves its files and all SSH keys. Re-run
the installer to restart it. An expired login/MFA session may still require the
user to authenticate. The current route also depends on alpha's authenticated
Lepton proxy; this is not yet an alpha-independent beta service credential.

The CPU tests validate generated launchd configuration, source integrity,
owned-session cleanup, real-health gating, post-auth timeouts and reconnect
when a process is alive but the forwarding path is dead. Native macOS
installation and reconnect must be verified
on the user's Mac; do not label those tested based on Linux unit tests.
