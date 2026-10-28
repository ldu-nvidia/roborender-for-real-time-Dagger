#!/usr/bin/env python3
"""Install the existing verified Mac bridge as a user-owned launchd agent.

No sudo, new credentials, SSH-config changes, or listener takeover. The user
must stop the old foreground bridge before launchd can acquire its two ports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

LABEL = "com.roborender.migration-bridge"
BOOTSTRAP_REMOTE = (
    "cosmos-h200:/workspace/project/roborender/pipeline/deployment/"
    "macos/bootstrap-mac-bridge.sh"
)
BOOTSTRAP_SHA = "e80a1b91d656392b5d50a7284c443ca7541138669a0a6209b72af0e7751ba354"
EXPECTED_CONTROL = "leo-cosmos-h200-access-8b47858b5-hzzsf"
EXPECTED_TARGETS = {"alpha": "8f68286-lcedt", "beta": "465d6b0-lcedt"}
HEALTH_INTERVAL = 30
HEALTH_SCRIPT = r'''
import json, socket, subprocess
result = {"control_hostname": socket.gethostname(), "targets": {}}
for name in ("alpha", "beta"):
    try:
        proc = subprocess.run([
            "ssh", "-F", "/workspace/.ssh/roborender_migration_config",
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=6", "-o", "ConnectionAttempts=1",
            "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "roborender-migration-" + name, "/bin/hostname"
        ], text=True, capture_output=True, timeout=8)
        result["targets"][name] = {"returncode": proc.returncode,
                                    "hostname": proc.stdout.strip()[:160]}
    except subprocess.TimeoutExpired:
        result["targets"][name] = {"error": "session_timeout"}
print(json.dumps(result, sort_keys=True))
'''


def bounded_command(command: list[str], timeout: float, *, input_data: str | None = None):
    """Bound the whole SSH/proxy process group, including post-auth hangs."""
    child = subprocess.Popen(command, stdin=subprocess.PIPE if input_data is not None else None,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.communicate()
        raise TimeoutError("SSH session exceeded its deadline") from None
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def probe_bridge(control_path: str, *, runner=bounded_command) -> dict:
    """Verify CPU -> forwarded RTX -> CPU traffic, not local master liveness."""
    command = ["/usr/bin/ssh", "-T", "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
               "-o", "ControlMaster=no", "-o", f"ControlPath={control_path}",
               "cosmos-h200", "python3 -c " + shlex.quote(HEALTH_SCRIPT)]
    try:
        result = runner(command, timeout=25)
        if result.returncode:
            return {"ok": False, "reason": "control_session_failed", "returncode": result.returncode}
        payload = json.loads(result.stdout)
        valid = (payload.get("control_hostname") == EXPECTED_CONTROL
                 and payload.get("targets") == {
                     name: {"returncode": 0, "hostname": host}
                     for name, host in EXPECTED_TARGETS.items()})
        return {"ok": valid, "reason": "round_trip_verified" if valid else "round_trip_failed",
                "targets": payload.get("targets", {})}
    except TimeoutError:
        return {"ok": False, "reason": "control_session_timeout"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"ok": False, "reason": "invalid_health_response"}


class HealthGate:
    """Two failed probes restart an owned session; stdout is never readiness."""
    def __init__(self):
        self.ready = False
        self.failures = 0

    def observe(self, result: dict) -> bool:
        self.ready = result.get("ok") is True
        self.failures = 0 if self.ready else self.failures + 1
        return self.failures >= 2


def report_status(app: Path, state: str, *, emit_log: bool = True, **details) -> None:
    record = {"schema": "roborender.bridge.health.v2", "state": state,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), **details}
    publish(app / "bridge-status.json", json.dumps(record, sort_keys=True).encode())
    if emit_log:
        print(json.dumps(record, sort_keys=True), flush=True)


def verify_bootstrap(data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != BOOTSTRAP_SHA:
        raise RuntimeError("Bridge source changed; refuse installation until reviewed.")


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink directory: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise RuntimeError(f"Expected an owner-only directory: {path}")


def publish(path: Path, data: bytes, mode: int = 0o600) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink target: {path}")
    fd, name = tempfile.mkstemp(prefix=".bridge-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def launch_spec(app: Path, logs: Path, python: str, keep_awake_on_ac: bool) -> dict:
    args = [python, str(app / "install_bridge.py"), "run"]
    if keep_awake_on_ac:
        # -s prevents system sleep only on AC power, not lid-close sleep.
        args = ["/usr/bin/caffeinate", "-s", *args]
    return {
        "Label": LABEL, "ProgramArguments": args,
        "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 30,
        "ExitTimeOut": 45, "ProcessType": "Background", "Umask": 0o077,
        "WorkingDirectory": str(app),
        "EnvironmentVariables": {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
            "PYTHONUNBUFFERED": "1",
        },
        "StandardOutPath": str(logs / "bridge.log"),
        "StandardErrorPath": str(logs / "bridge-error.log"),
    }


def verify_owned_plist(path: Path, app: Path) -> None:
    if path.is_symlink():
        raise RuntimeError("Refusing symlink LaunchAgent.")
    if path.exists():
        value = plistlib.loads(path.read_bytes())
        if (value.get("Label") != LABEL
                or str(app / "install_bridge.py") not in value.get("ProgramArguments", ())):
            raise RuntimeError("An unrelated LaunchAgent occupies this name; leaving it untouched.")


def activate_agent(plist: Path, app: Path, logs: Path, *, reuse_registration: bool,
                   runner=bounded_command, sleep=time.sleep) -> None:
    """Reload only our job; never bootstrap while an old registration remains."""
    verify_owned_plist(plist, app)
    service = f"gui/{os.getuid()}/{LABEL}"
    domain = f"gui/{os.getuid()}"
    events = []
    activated = False

    def call(*args, timeout=10):
        try:
            result = runner(["/bin/launchctl", *args], timeout=timeout)
        except (TimeoutError, OSError) as exc:
            events.append({"arguments": args, "error": type(exc).__name__})
            return subprocess.CompletedProcess(args, 124, "", "command timed out or unavailable")
        # Never include launchctl's inherited environment in diagnostics.
        output = result.stdout or ""
        if args[0] == "print":
            output = "\n".join(line.strip() for line in output.splitlines()
                               if re.match(r"\s*(path|state|pid|runs|last exit code)\s*=", line))
        events.append({"arguments": args, "returncode": result.returncode,
                       "stdout": output[-4000:], "stderr": (result.stderr or "")[-2000:]})
        return result

    def loaded():
        result = call("print", service)
        if result.returncode == 0:
            paths = re.findall(r"^\s*path\s*=\s*(.+?)\s*$", result.stdout, flags=re.MULTILINE)
            if paths != [str(plist)]:
                raise RuntimeError("Loaded service has an unexpected plist path; refusing to replace it")
            return True
        # 113 is launchctl's absent-service result, not a general I/O failure.
        if result.returncode == 113:
            return False
        raise RuntimeError("Cannot determine whether the bridge is registered; no blind replacement")

    try:
        registered = loaded()
        if call("enable", service).returncode:
            raise RuntimeError("Could not enable the owned bridge LaunchAgent")
        if registered and reuse_registration:
            # Program/args are unchanged: restart the existing registration
            # directly, avoiding the bootout/bootstrap teardown race entirely.
            if call("kickstart", "-kp", service, timeout=50).returncode == 0:
                activated = True
                return
            registered = loaded()
        if registered:
            call("bootout", service, timeout=50)
            for _ in range(50):
                if not loaded():
                    break
                sleep(1)
            else:
                raise RuntimeError("Old bridge registration did not unload; new bootstrap was not attempted")
        # A removed job may leave transient launchd bookkeeping. Retry only
        # after checking whether the preceding bootstrap actually registered it.
        for attempt in range(5):
            result = call("bootstrap", domain, str(plist), timeout=20)
            if loaded():
                if result.returncode == 0 or call("kickstart", "-p", service, timeout=20).returncode == 0:
                    activated = True
                    return
            sleep(attempt + 1)
        raise RuntimeError("LaunchAgent activation failed after bounded retries; no root or global reset attempted")
    finally:
        publish(logs / "activation.json", json.dumps({"service": service, "activated": activated,
                                                     "events": events},
                                                     sort_keys=True, indent=2).encode())


def upload_activation_report(logs: Path, *, runner=bounded_command) -> bool:
    """Send only sanitized launchctl diagnostics to the existing CPU workspace."""
    report = logs / "activation.json"
    if not report.is_file() or report.stat().st_size > 131072:
        return False
    remote = r'''
import json, os, pathlib, sys, tempfile
payload = json.loads(sys.stdin.read(131073))
assert payload["service"].startswith("gui/")
assert payload["service"].endswith("/com.roborender.migration-bridge")
root = pathlib.Path("/workspace/project/roborender/runtime/migrations/alpha-to-beta-20261008")
assert root.is_dir()
fd, name = tempfile.mkstemp(prefix="mac-bridge-activation-", suffix=".json", dir=root)
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
print(pathlib.Path(name).name)
'''
    try:
        result = runner(["/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                         "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ConnectTimeout=10",
                         "cosmos-h200", "python3 -c " + shlex.quote(remote)],
                        timeout=20, input_data=report.read_text())
        return result.returncode == 0
    except (OSError, TimeoutError):
        return False


def run_worker(app: Path) -> int:
    # A short private path satisfies OpenSSH's Unix-socket length limit even
    # with macOS's long default TMPDIR. %C separates each host's own master.
    with tempfile.TemporaryDirectory(prefix="rr-bridge-", dir="/tmp") as sockets:
        control_path = str(Path(sockets) / "%C")
        try:
            return run_private_worker(app, control_path)
        finally:
            for host in ("cosmos-h200", "rtx-6k-alpha", "rtx-6k-beta"):
                try:
                    bounded_command(["/usr/bin/ssh", "-o", f"ControlPath={control_path}",
                                     "-o", "BatchMode=yes", "-O", "exit", host], timeout=3)
                except (OSError, TimeoutError):
                    pass  # Only these temporary sockets are eligible for exit.


def run_private_worker(app: Path, control_path: str) -> int:
    bootstrap = app / "bootstrap-mac-bridge.sh"
    verify_bootstrap(bootstrap.read_bytes())
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = dict(os.environ, ROBORENDER_BRIDGE_CONTROL_PATH=control_path)
    child = subprocess.Popen(["/bin/bash", str(bootstrap)], env=env, start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    started, forwards_installed = time.monotonic(), False
    next_probe, last_heartbeat = started, started
    health = HealthGate()
    report_status(app, "connecting")
    pending = b""
    try:
        while child.poll() is None and not stopped:
            # SSH ConnectTimeout does not bound a post-authentication session
            # hang. Restart only our child after a bounded bootstrap attempt.
            now = time.monotonic()
            if not health.ready and not forwards_installed and now - started > 180:
                report_status(app, "reconnecting", reason="bootstrap_timeout")
                return 1
            for key, _ in selector.select(timeout=1):
                data = os.read(key.fileobj.fileno(), 8192)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                pending = (pending + data)[-16384:]
                if b"Bridge forwards installed on the private SSH connection" in pending:
                    forwards_installed = True
            if forwards_installed and time.monotonic() >= next_probe and not stopped:
                result = probe_bridge(control_path)
                was_ready = health.ready
                restart = health.observe(result)
                now = time.monotonic()
                emit_log = restart or not result["ok"] or not was_ready or now - last_heartbeat >= 300
                report_status(app, "reconnecting" if restart else ("ready" if health.ready else "degraded"),
                              emit_log=emit_log, consecutive_failures=health.failures, **result)
                if emit_log:
                    last_heartbeat = now
                if restart:
                    return 1
                next_probe = now + HEALTH_INTERVAL
        if not stopped:
            report_status(app, "reconnecting", reason="bootstrap_exited", returncode=child.returncode)
        else:
            report_status(app, "stopped", reason="requested_shutdown")
        return child.returncode if child.returncode is not None else 0
    finally:
        selector.close()
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["install", "run", "status", "stop"], default="install")
    parser.add_argument("--keep-awake-on-ac", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin" or os.geteuid() == 0:
        raise RuntimeError("Run on your Mac as your normal user, without sudo.")
    app = Path.home() / "Library/Application Support/RoboRenderBridge"
    logs = Path.home() / "Library/Logs/RoboRenderBridge"
    plist = Path.home() / "Library/LaunchAgents" / (LABEL + ".plist")
    service = f"gui/{os.getuid()}/{LABEL}"
    if args.action == "run":
        return run_worker(app)
    if args.action == "status":
        return subprocess.run(["/bin/launchctl", "print", service]).returncode
    verify_owned_plist(plist, app)
    if args.action == "stop":
        if not plist.is_file():
            raise RuntimeError("No owned bridge LaunchAgent installed.")
        result = subprocess.run(["/bin/launchctl", "bootout", service])
        print("Stopped only the RoboRender bridge job. Files and SSH keys are unchanged.")
        return result.returncode
    private_directory(app)
    private_directory(logs)
    marker = app / "installation.json"
    if any(app.iterdir()) and not marker.is_file():
        raise RuntimeError("Installation directory is not owned by this installer.")
    if marker.exists() and json.loads(marker.read_text()).get("label") != LABEL:
        raise RuntimeError("Unexpected installation owner.")
    with tempfile.TemporaryDirectory(prefix="roborender-bridge-install-") as tmp:
        fetched = Path(tmp) / "bootstrap.sh"
        subprocess.run(["/usr/bin/scp", "-o", "ControlMaster=no", "-o", "ControlPath=none",
                        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                        "-o", "ConnectTimeout=15", BOOTSTRAP_REMOTE, str(fetched)], check=True, timeout=90)
        raw = fetched.read_bytes()
        verify_bootstrap(raw)
    old_spec = plistlib.loads(plist.read_bytes()) if plist.is_file() else None
    publish(marker, json.dumps({"label": LABEL, "version": 2, "bootstrap_sha256": BOOTSTRAP_SHA}).encode())
    publish(app / "bootstrap-mac-bridge.sh", raw, 0o700)
    source = Path(__file__).resolve()
    if source != app / "install_bridge.py":
        publish(app / "install_bridge.py", source.read_bytes(), 0o700)
    # Keep the interpreter path (including its stable Homebrew symlink), not
    # a version-specific resolved Cellar path.
    python = shutil.which("python3") or sys.executable
    if not os.path.isabs(python):
        raise RuntimeError("An absolute Python 3 interpreter is required.")
    plist.parent.mkdir(parents=True, exist_ok=True)
    new_spec = launch_spec(app, logs, python, args.keep_awake_on_ac)
    publish(plist, plistlib.dumps(new_spec))
    subprocess.run(["/usr/bin/plutil", "-lint", str(plist)], check=True)
    try:
        activate_agent(plist, app, logs, reuse_registration=(old_spec == new_spec))
    except RuntimeError as exc:
        print(f"{exc}. Diagnostic report: {logs / 'activation.json'}", file=sys.stderr)
        return 1
    finally:
        if upload_activation_report(logs):
            print("Sanitized activation report delivered to the CPU workspace for automatic diagnosis.")
    print("Installed the per-user RoboRender bridge. Closing this terminal will not stop it.")
    print("It uses private SSH masters and checks both forwarded RTX hosts every 30 seconds.")
    print("Two failed checks trigger a scoped reconnect; launchd retries are throttled to 30 seconds.")
    print("Stop the OLD foreground bridge with Ctrl-C; the agent will then acquire those ports.")
    print("No private keys were copied and no existing SSH master was stopped.")
    print("Keep the Mac awake and connected. Sleep, logout, network loss or expired MFA can interrupt it.")
    print(f"Logs: {logs}")
    print(f"Status: python3 '{app / 'install_bridge.py'}' status")
    print(f"Stop: python3 '{app / 'install_bridge.py'}' stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
