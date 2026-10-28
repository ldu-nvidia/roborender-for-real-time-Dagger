"""Durable, pinned host-operation boundary for the beta campaign.

Deployment bindings contain explicit argv (never shell strings). Each operation
must implement --request/--receipt and be idempotent for request.operation_id.
Native collection itself is never restarted by this boundary. A lost process
without a completed receipt requires reconciliation, not a second launch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical

PHASES = {"prepare", "collect", "release", "evaluate", "control_only", "abort"}


def round_runtime_path(root, round_id, attempts=None):
    """Separate infrastructure retries from immutable science/ledger identity."""
    if type(round_id) is not int or round_id not in range(1, 5):
        raise ValueError("invalid infrastructure round")
    attempts = attempts or {}
    if (not isinstance(attempts, dict) or any(k not in {"1", "2", "3", "4"}
            or not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{32}", v)
            for k, v in attempts.items())):
        raise ValueError("invalid infrastructure attempt namespace")
    path = Path(root) / f"round-{round_id}"
    return path / ("recovery-" + attempts[str(round_id)]) if str(round_id) in attempts else path


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process_identity(pid):
    root = Path("/proc") / str(pid)
    try:
        # comm may contain spaces/parentheses: split only after the final ')'.
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19],
                "cmdline_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest()}
    except (OSError, IndexError):
        return None


class CommandOperations:
    def __init__(self, binding, root):
        if binding.get("schema") != "roborender.beta-campaign-operations.v1" or set(binding.get("commands", {})) != PHASES:
            raise ValueError("all scoped deployment operations must be explicitly bound")
        for name, spec in binding["commands"].items():
            argv = spec.get("argv")
            if (not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv)
                    or not Path(argv[0]).is_absolute() or Path(argv[0]).name in {"sh", "bash", "zsh"}
                    or type(spec.get("timeout_seconds")) is not int
                    or not 1 <= spec["timeout_seconds"] <= 43200):
                raise ValueError("operations require bounded explicit executable argv: " + name)
            executable = Path(spec["entrypoint"])
            if (not executable.is_absolute() or str(executable) not in argv
                    or hashlib.sha256(executable.read_bytes()).hexdigest() != spec.get("entrypoint_sha256")):
                raise ValueError("deployment operation source is not pinned: " + name)
        self.binding, self.root = binding, Path(root).resolve()
        self.children = {}

    def start(self, phase, context):
        if phase not in PHASES:
            raise ValueError("unknown host operation")
        request = {"schema": "roborender.beta-campaign-operation.v1", "phase": phase, **context}
        digest = hashlib.sha256(canonical(request)).hexdigest()
        oid = f"r{context['round_id']}-{phase}-{digest[:16]}"
        request["operation_id"] = oid
        directory = self.root / oid
        if directory.exists():
            if json.loads((directory / "request.json").read_bytes()) != request:
                raise RuntimeError("operation identity changed")
            return oid
        directory.mkdir(parents=True)
        atomic(directory / "request.json", request)
        spec = self.binding["commands"][phase]
        if hashlib.sha256(Path(spec["entrypoint"]).read_bytes()).hexdigest() != spec["entrypoint_sha256"]:
            raise RuntimeError("deployment operation changed after admission")
        argv = spec["argv"] + ["--request", str(directory / "request.json"),
                               "--receipt", str(directory / "receipt.json")]
        atomic(directory / "intent.json", {"argv": argv, "started_at": time.time(),
                                           "deadline": time.time() + spec["timeout_seconds"]})
        with (directory / "console.log").open("xb") as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.children[oid] = child
        identity = process_identity(child.pid)
        atomic(directory / "process.json", {"pid": child.pid, "identity": identity})
        return oid

    def poll(self, oid):
        directory = self.root / oid
        request = json.loads((directory / "request.json").read_bytes())
        receipt = directory / "receipt.json"
        child = self.children.get(oid)
        exit_code = child.poll() if child is not None else None
        if receipt.exists():
            if receipt.stat().st_size > 4 * 1024 * 1024:
                raise RuntimeError("oversized host-operation receipt")
            value = json.loads(receipt.read_bytes())
            if (value.get("ok") is not True or value.get("operation_id") != oid
                    or value.get("run_id") != request["run_id"] or value.get("round_id") != request["round_id"]
                    or value.get("phase") != request["phase"] or exit_code not in (None, 0)):
                raise RuntimeError("host-operation receipt failed or identity differs")
            if child is not None and exit_code is None:
                return None  # completed evidence is not proof the owner exited
            process_path = directory / "process.json"
            if child is None and process_path.exists():
                saved = json.loads(process_path.read_bytes())
                actual = process_identity(saved["pid"])
                if actual is not None and actual == saved["identity"]:
                    return None
            return value
        intent = json.loads((directory / "intent.json").read_bytes())
        if time.time() > intent["deadline"]:
            raise TimeoutError("operation deadline; reconcile owned workload, never replay: " + oid)
        if child is not None:
            if exit_code is not None:
                raise RuntimeError("operation exited without a valid receipt: " + oid)
        else:
            path = directory / "process.json"
            saved = json.loads(path.read_bytes()) if path.exists() else {}
            if not saved.get("identity") or process_identity(saved.get("pid")) != saved["identity"]:
                raise RuntimeError("operation process uncertain; refusing automatic re-launch: " + oid)
        return None
