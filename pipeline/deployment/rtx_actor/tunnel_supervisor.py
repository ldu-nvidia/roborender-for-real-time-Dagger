"""Run-owned, identity-pinned SSH forwards with bounded automatic recovery.

Foreground only: the workload launcher owns this process and its lifetime.
No GPU use, SSH key installation, remote writes, or unscoped process cleanup.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

# Direct execution must load the same frozen source tree, not an older mutable
# checkout from the working directory.
SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from pipeline.dagger.services.recovery_policy import SSH_STARTUP_SECONDS, SSH_RECOVERY_SECONDS

SCHEMA = "roborender.ssh-supervisor.v1"
PRODUCTION_SCHEMA = "roborender.ssh-supervisor-retained-production.v1"
STARTUP_SECONDS = SSH_STARTUP_SECONDS
RECOVERY_SECONDS = SSH_RECOVERY_SECONDS
HEALTH_INTERVAL = 5.0
HEALTH_TIMEOUT = 5.0
UNRESPONSIVE_GRACE_SECONDS = 15.0


class IdentityMismatch(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_config(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if set(value) != {"schema", "run_id", "ssh_config", "known_hosts", "ssh_host", "lifetime_seconds", "forwards"}:
        raise ValueError("unexpected supervisor config keys")
    if value["schema"] not in {SCHEMA, PRODUCTION_SCHEMA} or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}", value["run_id"]):
        raise ValueError("invalid supervisor identity")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", value["ssh_host"]):
        raise ValueError("SSH host must be a configured alias, not options")
    maximum = 7 * 86400 if value['schema'] == PRODUCTION_SCHEMA else 86400
    if type(value["lifetime_seconds"]) is not int or not 30 <= value["lifetime_seconds"] <= maximum:
        raise ValueError("explicit bounded supervisor lifetime required")
    for name in ("ssh_config", "known_hosts"):
        p = Path(value[name])
        if not p.is_absolute() or not p.is_file() or p.is_symlink() or p.stat().st_mode & 0o022:
            raise ValueError(f"unsafe or missing {name}")
    rows = value["forwards"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("one to sixteen explicit forwards required")
    ports = set()
    for row in rows:
        if set(row) != {"local_port", "remote_port", "expected_ready"}:
            raise ValueError("unexpected forward keys")
        for key in ("local_port", "remote_port"):
            if type(row[key]) is not int or not 1024 <= row[key] <= 65535:
                raise ValueError("invalid explicit forward port")
        if row["local_port"] in ports:
            raise ValueError("duplicate local port")
        ports.add(row["local_port"])
        expected = row["expected_ready"]
        if (not isinstance(expected, dict) or not {"ok", "role", "pid"} <= set(expected)
                or not set(expected) <= {"ok", "role", "pid", "schema", "gateway_instance_id"}
                or expected["ok"] is not True or type(expected["pid"]) is not int or expected["pid"] <= 1
                or not isinstance(expected["role"], str) or not expected["role"]):
            raise ValueError("pin exact service role and PID before opening the forward")
        if any(not isinstance(v, str) or not v for k, v in expected.items() if k not in {"ok", "pid"}):
            raise ValueError("invalid readiness identity")
    return value, hashlib.sha256(raw).hexdigest()


def ssh_command(config, row):
    # Explicit command-line values override a stale or permissive SSH config.
    return ["ssh", "-F", config["ssh_config"], "-N", "-T",
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={config['known_hosts']}",
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ForwardAgent=no",
            "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ControlPersist=no",
            "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
            "-L", f"127.0.0.1:{row['local_port']}:127.0.0.1:{row['remote_port']}", config["ssh_host"]]


def probe(row):
    conn = http.client.HTTPConnection("127.0.0.1", row["local_port"], timeout=HEALTH_TIMEOUT)
    try:
        conn.request("GET", "/ready", headers={"Connection": "close"})
        response = conn.getresponse()
        raw = response.read((16 << 10) + 1)
        if response.status != 200:
            raise OSError(f"readiness HTTP {response.status}")
        if len(raw) > 16 << 10 or response.getheader("Content-Type", "").split(";", 1)[0].strip() != "application/json":
            raise IdentityMismatch("unexpected readiness response boundary")
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise IdentityMismatch("invalid readiness JSON") from exc
        if not isinstance(value, dict) or any(value.get(k) != v for k, v in row["expected_ready"].items()):
            raise IdentityMismatch("service identity changed; re-attestation required")
        return True, None
    except IdentityMismatch:
        raise
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return False, type(exc).__name__
    finally:
        conn.close()


class Supervisor:
    def __init__(self, config, digest, root, *, popen=subprocess.Popen, health=probe,
                 monotonic=time.monotonic, sleep=time.sleep):
        self.config, self.digest, self.root = config, digest, Path(root)
        self.popen, self.health, self.clock, self.sleep = popen, health, monotonic, sleep
        self.stop_requested = False
        self.entries = [{"spec": row, "process": None, "log": None, "healthy": False,
                         "starts": 0, "failures": 0, "ever_ready": False, "next_start": 0., "started": 0.,
                         "down_since": None, "error": None} for row in config["forwards"]]
        self.started = self.clock()
        self.events = None

    def event(self, name, row=None, **fields):
        record = {"event": name, "time_ns": time.time_ns(), "run_id": self.config["run_id"], **fields}
        if row is not None:
            record["local_port"] = row["spec"]["local_port"]
        self.events.write(json.dumps(record, sort_keys=True) + "\n")
        self.events.flush()

    def status(self, state):
        value = {"schema": SCHEMA, "run_id": self.config["run_id"], "config_sha256": self.digest,
                 "supervisor_pid": os.getpid(), "updated_at_ns": time.time_ns(), "state": state,
                 "forwards": [{"local_port": e["spec"]["local_port"], "expected_ready": e["spec"]["expected_ready"],
                               "healthy": e["healthy"], "ssh_pid": e["process"].pid if e["process"] else None,
                               "starts": e["starts"], "last_error_type": e["error"]} for e in self.entries]}
        part = self.root / "status.part"
        part.write_bytes(canonical(value))
        part.chmod(0o600)
        os.replace(part, self.root / "status.json")

    def launch(self, row):
        # Never take over somebody else's listener, including after a failure.
        with socket.socket() as check:
            # Match sshd/ssh's SO_REUSEADDR behavior: completed readiness
            # connections can leave this port in TIME_WAIT after SSH exits.
            # This does NOT enable SO_REUSEPORT or permit stealing a listener.
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", row["spec"]["local_port"]))
        row["log"] = (self.root / f"ssh-{row['spec']['local_port']}.log").open("ab", buffering=0)
        row["process"] = self.popen(ssh_command(self.config, row["spec"]), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=row["log"], start_new_session=True)
        row["starts"] += 1
        row["failures"], row["ever_ready"] = 0, False
        row["started"] = self.clock()
        if row["down_since"] is None:
            row["down_since"] = self.clock()
        self.event("ssh_started", row, ssh_pid=row["process"].pid, attempt=row["starts"])

    def terminate(self, row):
        process = row["process"]
        if process is not None:
            # Only the new session/process group created by this supervisor.
            if process.pid <= 1:
                raise RuntimeError("invalid owned SSH PID")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            row["process"] = None
        if row["log"] is not None:
            row["log"].close()
            row["log"] = None
        row["healthy"] = False

    def cycle(self, pool):
        for row in self.entries:
            p = row["process"]
            if p is not None and p.poll() is not None:
                self.event("ssh_exited", row, exit_code=p.returncode)
                self.terminate(row)
                row["error"] = "SSHExited"
                if row["down_since"] is None:
                    row["down_since"] = self.clock()
                row["next_start"] = self.clock() + min(5., .5 * row["starts"])
            # A retained campaign can remain safely idle through a longer
            # network outage. Keep reconnecting its owned transport within the
            # declared supervisor lifetime; never restart native work here.
            # IdentityMismatch remains terminal and old bounded probes retain
            # their original recovery budget.
            if (self.config['schema'] != PRODUCTION_SCHEMA and row["down_since"] is not None
                    and self.clock() - row["down_since"] > RECOVERY_SECONDS):
                raise RuntimeError(f"forward {row['spec']['local_port']} exceeded recovery budget")
            if row["process"] is None and self.clock() >= row["next_start"]:
                self.launch(row)
        futures = [(row, pool.submit(self.health, row["spec"])) for row in self.entries if row["process"]]
        for row, future in futures:
            ok, error = future.result()  # IdentityMismatch is terminal, never a retry.
            if ok:
                if not row["healthy"]:
                    self.event("forward_ready", row, recovery_seconds=self.clock() - row["down_since"]
                               if row["down_since"] is not None else 0.)
                row.update(healthy=True, failures=0, down_since=None, error=None, ever_ready=True)
                continue
            if row["healthy"]:
                self.event("forward_unhealthy", row, error_type=error)
            row["healthy"] = False
            row["error"] = error
            row["failures"] += 1
            if row["down_since"] is None:
                row["down_since"] = self.clock()
            if (row["failures"] >= 3
                    and self.clock() - row["down_since"] >= UNRESPONSIVE_GRACE_SECONDS
                    and (row["ever_ready"] or self.clock() - row["started"] >= STARTUP_SECONDS)):
                self.event("restarting_unresponsive_ssh", row, error_type=error)
                self.terminate(row)
                row["next_start"] = self.clock() + .5
        self.status("ready" if all(e["healthy"] for e in self.entries) else "recovering")

    def run(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        with (self.root / "owner.lock").open("x") as lock, (self.root / "events.jsonl").open("x") as self.events:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = "stopped"
            self.event("supervisor_started", supervisor_pid=os.getpid(), config_sha256=self.digest)
            try:
                with ThreadPoolExecutor(max_workers=len(self.entries)) as pool:
                    while not self.stop_requested:
                        if self.clock() - self.started >= self.config["lifetime_seconds"]:
                            state = "expired"
                            break
                        self.cycle(pool)
                        self.sleep(HEALTH_INTERVAL)
            except BaseException as exc:
                state = "failed"
                self.event("supervisor_failed", error_type=type(exc).__name__, message=str(exc)[:256])
                raise
            finally:
                for row in self.entries:
                    self.terminate(row)
                self.status(state)
                self.event("supervisor_stopped", state=state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config, digest = read_config(args.config)
    if not args.root.is_absolute():
        raise ValueError("an absolute, new run-owned status directory is required")
    supervisor = Supervisor(config, digest, args.root)
    def stop(_signum, _frame):
        supervisor.stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    supervisor.run()


if __name__ == "__main__":
    main()
