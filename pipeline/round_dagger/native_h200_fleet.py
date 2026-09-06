"""Small fail-closed primitives shared by the native eight-H200 fleets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from round_dagger.artifacts import load_strict_json


GPU_COUNT = 8
GPU_NAME = "NVIDIA H200"
COMPUTE_CAPABILITY = "9.0"
CANONICAL_CAMPAIGN_ROOT = Path(
    "/workspace/project/roborender/data"
)
CUDA_DEVICE_SET = "0,1,2,3,4,5,6,7"
SAFE_EXECUTABLE_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


class NativeH200FleetError(RuntimeError):
    """A fleet input, local runtime, or output boundary is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NativeH200FleetError("cannot hash non-canonical data") from exc
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise NativeH200FleetError(f"{label} must be a lowercase SHA-256")
    return value


def require_environment(
    role: str, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Validate the no-argument launch-plan envelope for one fleet."""

    values = dict(os.environ if environment is None else environment)
    required = (
        "ROUND_DAGGER_FLEET_ROLE",
        "ROUND_DAGGER_CAMPAIGN_ROOT",
        "ROUND_DAGGER_GENERATION_ID",
        "ROUND_DAGGER_GENERATION_ROOT",
        "ROUND_DAGGER_OUTPUT_ROOT",
        "ROUND_DAGGER_PROTOCOL_ID",
        "ROUND_DAGGER_SPLIT_SHA256",
        "CUDA_VISIBLE_DEVICES",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise NativeH200FleetError(
            "missing launch-plan environment: " + ", ".join(missing)
        )
    if values["ROUND_DAGGER_FLEET_ROLE"] != role:
        raise NativeH200FleetError(f"fleet role must be exactly {role}")
    if values["CUDA_VISIBLE_DEVICES"] != CUDA_DEVICE_SET:
        raise NativeH200FleetError("fleet must receive physical CUDA devices 0..7")
    if values["ROUND_DAGGER_CAMPAIGN_ROOT"] != str(CANONICAL_CAMPAIGN_ROOT):
        raise NativeH200FleetError("fleet campaign root differs")
    require_sha256(values["ROUND_DAGGER_SPLIT_SHA256"], "group split")
    generation_root = safe_directory(
        Path(values["ROUND_DAGGER_GENERATION_ROOT"]),
        "generation root",
        root=CANONICAL_CAMPAIGN_ROOT,
    )
    output_root = safe_directory(
        Path(values["ROUND_DAGGER_OUTPUT_ROOT"]),
        "output root",
        root=CANONICAL_CAMPAIGN_ROOT,
    )
    try:
        generation_root.relative_to(CANONICAL_CAMPAIGN_ROOT)
        output_root.relative_to(CANONICAL_CAMPAIGN_ROOT)
    except ValueError as exc:  # pragma: no cover - safe_directory already catches it
        raise NativeH200FleetError("fleet output escaped its campaign") from exc
    return values


def _resolved(path: Path, label: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise NativeH200FleetError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise NativeH200FleetError(f"{label} is unavailable: {path}") from exc
    if resolved != path:
        raise NativeH200FleetError(f"{label} must be canonical and contain no symlinks")
    return resolved


def _under(path: Path, root: Path, label: str) -> None:
    root = root.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise NativeH200FleetError(f"{label} lies outside {root}") from exc


def safe_directory(path: Path, label: str, *, root: Path | None = None) -> Path:
    path = _resolved(path, label, must_exist=True)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeH200FleetError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise NativeH200FleetError(f"{label} must be a real directory")
    if root is not None:
        _under(path, root, label)
    return path


def safe_regular(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
    expected_sha256: str | None = None,
    read_only: bool = True,
) -> Path:
    path = _resolved(path, label, must_exist=True)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeH200FleetError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise NativeH200FleetError(f"{label} must be a regular non-symlink file")
    if read_only and stat.S_IMODE(info.st_mode) & 0o222:
        raise NativeH200FleetError(f"{label} must be immutable")
    if root is not None:
        _under(path, root, label)
    if expected_sha256 is not None:
        expected = require_sha256(expected_sha256, f"{label} SHA-256")
        if sha256_file(path) != expected:
            raise NativeH200FleetError(f"{label} SHA-256 differs")
    return path


def safe_python_launcher(
    path: Path,
    label: str,
    *,
    target_root: Path,
) -> Path:
    """Admit a Python launcher symlink only when its executable target is pinned."""

    if not path.is_absolute():
        raise NativeH200FleetError(f"{label} must be absolute")
    try:
        target = path.resolve(strict=True)
        target_root = target_root.resolve(strict=True)
        target.relative_to(target_root)
        info = os.stat(target)
    except (OSError, ValueError) as exc:
        raise NativeH200FleetError(f"{label} target is unavailable or escaped") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or not stat.S_IMODE(info.st_mode) & 0o111
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise NativeH200FleetError(f"{label} target is unsafe")
    return path


def strict_document(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
    expected_sha256: str | None = None,
    read_only: bool = True,
) -> dict[str, Any]:
    path = safe_regular(
        path,
        label,
        root=root,
        expected_sha256=expected_sha256,
        read_only=read_only,
    )
    try:
        document = load_strict_json(path)
    except Exception as exc:
        raise NativeH200FleetError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise NativeH200FleetError(f"{label} must contain a JSON object")
    return document


def query_h200_inventory() -> list[dict[str, Any]]:
    """Return the exact local, non-MIG, physical 8xH200 inventory."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            env={"PATH": SAFE_EXECUTABLE_PATH},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeH200FleetError("cannot query the local GPU inventory") from exc
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != GPU_COUNT:
        raise NativeH200FleetError("native fleet requires exactly eight physical GPUs")
    inventory: list[dict[str, Any]] = []
    for expected_index, line in enumerate(rows):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise NativeH200FleetError("GPU inventory row is malformed")
        try:
            row = {
                "index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2].upper(),
                "name": fields[3],
                "compute_capability": fields[4],
                "total_memory_mib": int(fields[5]),
                "mig_mode": fields[6],
            }
        except ValueError as exc:
            raise NativeH200FleetError("GPU inventory value is malformed") from exc
        if (
            row["index"] != expected_index
            or not row["uuid"].startswith("GPU-")
            or row["name"] != GPU_NAME
            or row["compute_capability"] != COMPUTE_CAPABILITY
            or row["total_memory_mib"] < 140_000
            or not str(row["mig_mode"]).lower().startswith("disabled")
        ):
            raise NativeH200FleetError(
                f"physical GPU {expected_index} is not an admitted H200"
            )
        inventory.append(row)
    return inventory


def sanitized_environment(
    additions: Mapping[str, str], *, cuda_devices: str = CUDA_DEVICE_SET
) -> dict[str, str]:
    """Construct a local-only child environment without inherited hooks."""

    result = {
        "PATH": SAFE_EXECUTABLE_PATH,
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in additions.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise NativeH200FleetError("child environment contains an invalid value")
        if "\x00" in key or "\x00" in value or "=" in key:
            raise NativeH200FleetError("child environment contains an unsafe value")
        result[key] = value
    return result


def run_checked(
    argv: Sequence[str | Path],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    timeout: float | None = None,
) -> None:
    command = [str(item) for item in argv]
    if not command or any(not item or "\x00" in item for item in command):
        raise NativeH200FleetError("child command is malformed")
    safe_directory(cwd, "child working directory")
    try:
        subprocess.run(
            command,
            check=True,
            cwd=cwd,
            env=dict(environment),
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeH200FleetError(f"fleet child failed: {command[0]}") from exc


__all__ = (
    "CANONICAL_CAMPAIGN_ROOT",
    "COMPUTE_CAPABILITY",
    "CUDA_DEVICE_SET",
    "GPU_COUNT",
    "GPU_NAME",
    "NativeH200FleetError",
    "SAFE_EXECUTABLE_PATH",
    "canonical_sha256",
    "query_h200_inventory",
    "require_environment",
    "require_sha256",
    "run_checked",
    "safe_directory",
    "safe_python_launcher",
    "safe_regular",
    "sanitized_environment",
    "sha256_file",
    "strict_document",
)
