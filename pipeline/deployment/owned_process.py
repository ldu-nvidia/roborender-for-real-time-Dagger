"""Run-owned detached process, deadline and durable completion receipt.

SSH is not the workload lifetime. Repeating start never replays a lost native
task. Cleanup targets a recorded PID identity, never a global process name.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, value):
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as output:
        output.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def identity(pid):
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19], "cmdline_sha256": hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest()}
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def read_spec(path):
    value = json.loads(path.read_bytes())
    if set(value) != {"argv", "cwd", "ttl", "env", "owner"}:
        raise ValueError("unexpected process specification")
    if (not isinstance(value["argv"], list) or not value["argv"] or any(not isinstance(x, str) or not x for x in value["argv"])
            or not Path(value["argv"][0]).is_absolute() or not Path(value["cwd"]).is_absolute()
            or type(value["ttl"]) is not int or not 30 <= value["ttl"] <= 43200
            or not isinstance(value["owner"], str) or not value["owner"].startswith(("dagger-beta-", "dagger-leo-"))
            or not isinstance(value["env"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value["env"].items())
            or any(k in value["env"] for k in ("HOME", "CODEX_HOME", "LD_PRELOAD", "WANDB_API_KEY"))):
        raise ValueError("invalid scoped process specification")
    return value


def status(directory):
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_bytes())
    path = directory / "owner.json"
    if not path.exists():
        return {"state": "STARTING" if (directory / "spec.json").exists() else "ABSENT"}
    saved = json.loads(path.read_bytes())
    if identity(saved["pid"]) != saved["identity"]:
        return {"state": "UNCERTAIN", "owner": saved["owner"], "restart_allowed": False}
    return {"state": "RUNNING", "owner": saved["owner"], "pid": saved["pid"]}


def start(directory, spec_path):
    spec = read_spec(spec_path)
    if directory.exists():
        if json.loads((directory / "spec.json").read_bytes()) != spec:
            raise RuntimeError("existing process identity changed")
        return status(directory)
    directory.mkdir(parents=True)
    atomic(directory / "spec.json", spec)
    with (directory / "supervisor.log").open("xb") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "supervise", "--directory", str(directory)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"state": "STARTING", "owner": spec["owner"]}


def stop(directory):
    value = status(directory)
    if value["state"] == "RUNNING":
        saved = json.loads((directory / "owner.json").read_bytes())
        if identity(saved["pid"]) != saved["identity"]:
            raise RuntimeError("owned supervisor PID changed")
        os.kill(saved["pid"], signal.SIGTERM)
    elif value["state"] in {"UNCERTAIN", "STARTING"}:
        raise RuntimeError("workload process is uncertain; reconcile before cleanup")
    return value


def supervise(directory):
    spec = read_spec(directory / "spec.json")
    if (directory / "owner.json").exists():
        raise RuntimeError("never restart a native lifecycle from persisted process state")
    stopped = False
    def request_stop(sig, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    atomic(directory / "owner.json", {"pid": os.getpid(), "identity": identity(os.getpid()), "owner": spec["owner"]})
    child = None
    result = {"state": "FAILED", "owner": spec["owner"], "restart_allowed": False}
    try:
        with (directory / "console.log").open("xb") as log:
            child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=dict(os.environ, **spec["env"]),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + spec["ttl"]
            while child.poll() is None and not stopped and time.monotonic() < deadline:
                time.sleep(.5)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            result.update(state="COMPLETE" if child.returncode == 0 and not stopped else "STOPPED" if stopped else "FAILED",
                          exit_code=child.returncode)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=45)
        atomic(directory / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop", "supervise"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if "runtime" not in directory.parts or len(directory.parts) < 5:
        raise ValueError("owned process directories must be scoped under a runtime directory")
    if args.action == "supervise":
        supervise(directory)
        return
    if args.action == "start":
        if args.spec is None:
            parser.error("start requires --spec")
        result = start(directory, args.spec)
    else:
        result = status(directory) if args.action == "status" else stop(directory)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
