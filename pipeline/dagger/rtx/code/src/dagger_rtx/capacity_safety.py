"""Narrow ownership checks for concurrent certified capacity probes."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Callable


_GROUP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def validate_capacity_group_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not _GROUP_RE.fullmatch(value):
        raise RuntimeError("capacity group id must be a 1..96 character safe identifier")
    return value


def parse_collector_processes(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "pgrep -af" in stripped or "scripts/preflight.py" in stripped:
            continue
        pid_text, separator, command = stripped.partition(" ")
        if not separator or not pid_text.isdigit():
            raise RuntimeError(f"cannot parse active collector process row: {line!r}")
        rows.append({"pid": int(pid_text), "command": command})
    return rows


def validate_collector_ownership(
    processes: list[dict[str, Any]],
    *,
    capacity_group_id: str | None,
    allowed_pids: set[int],
) -> None:
    if not processes:
        return
    if capacity_group_id is None:
        raise RuntimeError("active Raven collector detected; preflight will not proceed")
    unmatched = [row for row in processes if row["pid"] not in allowed_pids]
    if unmatched:
        raise RuntimeError(
            "active Raven collector is not owned by the exact coordinated capacity group: "
            + json.dumps(unmatched, sort_keys=True)
        )


def attest_group_container_pids(
    group_id: str,
    *,
    selected_gpu: int,
    expected_image_id: str,
    run: Callable[[list[str]], str] | None = None,
    read_cmdline: Callable[[int], bytes] | None = None,
) -> tuple[set[int], list[dict[str, Any]]]:
    """Return host PIDs owned by active, certified, same-group containers.

    A group string alone grants nothing: every allowed PID must be inside a
    running container created by the certified launcher, carry exact labels,
    use the pinned image, and own a *different* physical GPU from this launch.
    """

    validate_capacity_group_id(group_id)
    execute = run or _run
    read_owner_cmdline = read_cmdline or _read_cmdline
    raw_ids = execute(
        [
            "docker",
            "ps",
            "--filter",
            f"label=roborender.capacity.group={group_id}",
            "--format",
            "{{.ID}}",
        ]
    )
    allowed: set[int] = set()
    attestations: list[dict[str, Any]] = []
    for container_id in filter(None, (line.strip() for line in raw_ids.splitlines())):
        raw = execute(
            [
                "docker",
                "inspect",
                container_id,
                "--format",
                "{{json .}}",
            ]
        )
        value = json.loads(raw)
        labels = value.get("Config", {}).get("Labels") or {}
        if value.get("Image") != expected_image_id:
            raise RuntimeError("same-group capacity container image identity mismatch")
        if value.get("State", {}).get("Status") != "running":
            raise RuntimeError("same-group capacity container is not running")
        if labels.get("roborender.capacity.group") != group_id:
            raise RuntimeError("capacity container group label mismatch")
        gpu_text = labels.get("roborender.capacity.physical-gpu", "")
        owner_text = labels.get("roborender.capacity.owner-pid", "")
        if gpu_text not in {"0", "1"} or not owner_text.isdigit():
            raise RuntimeError("capacity container ownership labels are malformed")
        physical_gpu = int(gpu_text)
        if physical_gpu == selected_gpu:
            raise RuntimeError("capacity group already owns the selected physical GPU")
        owner_pid = int(owner_text)
        try:
            cmdline = read_owner_cmdline(owner_pid).replace(b"\0", b" ")
        except OSError as exc:
            raise RuntimeError("capacity container launcher owner is no longer live") from exc
        if b"launch_native_raven.sh" not in cmdline:
            raise RuntimeError("capacity container owner is not the certified launcher")
        top = execute(["docker", "top", container_id, "-eo", "pid"])
        pids = {
            int(line.strip())
            for line in top.splitlines()[1:]
            if line.strip().isdigit()
        }
        if not pids:
            raise RuntimeError("capacity container has no attestable host processes")
        allowed.update(pids)
        attestations.append(
            {
                "container_id": container_id,
                "physical_gpu": physical_gpu,
                "owner_pid": owner_pid,
                "attested_process_count": len(pids),
            }
        )
    return allowed, attestations


def _read_cmdline(pid: int) -> bytes:
    with open(f"/proc/{pid}/cmdline", "rb") as stream:
        return stream.read()


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()
