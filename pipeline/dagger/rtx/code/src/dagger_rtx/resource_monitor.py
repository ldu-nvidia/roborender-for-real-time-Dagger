"""Host/physical-GPU sampler used only by isolated capacity probes."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .capacity_telemetry import AppendOnlyTelemetry
from .durable import atomic_write_json


_SIZE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?i?B)$", re.I)
_MULTIPLIERS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def parse_size_bytes(value: str) -> int:
    match = _SIZE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"unsupported size value: {value!r}")
    unit = match.group(2).upper()
    if unit not in _MULTIPLIERS:
        raise ValueError(f"unsupported size unit: {unit}")
    return int(float(match.group(1)) * _MULTIPLIERS[unit])


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.strip().split()
        if not fields or not fields[0].isdigit():
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        values[name] = int(fields[0]) * multiplier
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required.issubset(values):
        raise RuntimeError("/proc/meminfo lacks required capacity fields")
    return {
        "host_memory_total_bytes": values["MemTotal"],
        "host_memory_available_bytes": values["MemAvailable"],
        "host_swap_total_bytes": values["SwapTotal"],
        "host_swap_free_bytes": values["SwapFree"],
        "host_swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def query_gpu(physical_gpu: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(physical_gpu),
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.strip()
    fields = [part.strip() for part in raw.split(",")]
    if len(fields) != 9:
        raise RuntimeError(f"unexpected nvidia-smi field count: {raw!r}")

    def number(value: str) -> float | None:
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    return {
        "physical_gpu": int(fields[0]),
        "gpu_name": fields[1],
        "gpu_uuid": fields[2],
        "gpu_memory_total_mib": number(fields[3]),
        "gpu_memory_used_mib": number(fields[4]),
        "gpu_utilization_percent": number(fields[5]),
        "gpu_memory_utilization_percent": number(fields[6]),
        "gpu_power_watts": number(fields[7]),
        "gpu_temperature_c": number(fields[8]),
    }


def query_container(container_name: str) -> dict[str, Any]:
    raw = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            container_name,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if not raw:
        return {"container_running": False}
    value = json.loads(raw.splitlines()[-1])
    usage_text = str(value.get("MemUsage", "")).split("/", 1)[0].strip()
    cpu_text = str(value.get("CPUPerc", "")).rstrip("%")
    return {
        "container_running": True,
        "container_memory_usage_bytes": parse_size_bytes(usage_text),
        "container_cpu_percent": float(cpu_text),
        "container_pids": int(value.get("PIDs", 0)),
    }


def summarize_resource_samples(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = [row for row in rows if row.get("event") == "resource_sample"]

    def values(name: str) -> list[float]:
        return [
            float(row[name])
            for row in samples
            if isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
            and math.isfinite(float(row[name]))
        ]

    def extrema(name: str, operation) -> float | int | None:
        found = values(name)
        return operation(found) if found else None

    return {
        "schema": "roborender.dagger.rtx.capacity-resources-summary.v1",
        "sample_count": len(samples),
        "sample_error_count": sum(row.get("event") == "resource_sample_error" for row in rows),
        "gpu_memory_used_mib_max": extrema("gpu_memory_used_mib", max),
        "gpu_utilization_percent_max": extrema("gpu_utilization_percent", max),
        "gpu_memory_utilization_percent_max": extrema(
            "gpu_memory_utilization_percent", max
        ),
        "gpu_power_watts_max": extrema("gpu_power_watts", max),
        "gpu_temperature_c_max": extrema("gpu_temperature_c", max),
        "host_memory_available_bytes_min": extrema("host_memory_available_bytes", min),
        "host_swap_used_bytes_max": extrema("host_swap_used_bytes", max),
        "container_memory_usage_bytes_max": extrema(
            "container_memory_usage_bytes", max
        ),
        "container_cpu_percent_max": extrema("container_cpu_percent", max),
        "container_pids_max": extrema("container_pids", max),
    }


def sample_until_stopped(
    *,
    physical_gpu: int,
    container_name: str,
    journal_path: Path,
    summary_path: Path,
    should_stop,
    interval_s: float = 1.0,
) -> None:
    telemetry = AppendOnlyTelemetry(journal_path)
    while not should_stop():
        started = time.perf_counter_ns()
        try:
            row = {**read_meminfo(), **query_gpu(physical_gpu)}
            try:
                row.update(query_container(container_name))
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                row["container_running"] = False
            telemetry.record("resource_sample", **row)
        except BaseException as exc:
            telemetry.record(
                "resource_sample_error",
                error_type=type(exc).__name__,
                error=str(exc)[:512],
            )
        elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
        time.sleep(max(0.05, interval_s - elapsed_s))
    rows = telemetry.read()
    atomic_write_json(summary_path, summarize_resource_samples(rows))
