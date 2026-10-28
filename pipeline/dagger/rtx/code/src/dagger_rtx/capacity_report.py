"""Deterministic aggregation of isolated capacity telemetry."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .durable import atomic_write_json


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0,1]")
    position = quantile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def latency_summary_ns(values: Iterable[int | float]) -> dict[str, float | int | None]:
    found = [float(value) for value in values]
    return {
        "count": len(found),
        "min_ns": min(found) if found else None,
        "p50_ns": percentile(found, 0.50),
        "p95_ns": percentile(found, 0.95),
        "max_ns": max(found) if found else None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def build_capacity_report(
    *,
    native_result: dict[str, Any],
    telemetry_rows: list[dict[str, Any]],
    resource_summary: dict[str, Any],
    expected_steps: int | None = None,
) -> dict[str, Any]:
    if expected_steps is None:
        configured_steps = native_result.get("capacity_steps", 10)
        expected_steps = (
            configured_steps
            if isinstance(configured_steps, int) and not isinstance(configured_steps, bool)
            else 10
        )
    batch_size = native_result.get("num_envs")
    wire_protocol = native_result.get("wire_protocol", "v1")
    step_rows = [row for row in telemetry_rows if row.get("event") == "vector_step_complete"]
    http_rows = [
        row
        for row in telemetry_rows
        if row.get("event")
        in {"http_request", "http_batch_request_v2", "http_batch_request_v3"}
    ]
    phase_values: dict[str, list[int]] = defaultdict(list)
    for row in telemetry_rows:
        if row.get("event") == "vector_phase" and row.get("outcome") == "success":
            duration = row.get("duration_ns")
            if isinstance(duration, int):
                phase_values[str(row.get("phase"))].append(duration)
    http_values: dict[str, list[int]] = defaultdict(list)
    for row in http_rows:
        if row.get("outcome") == "success" and isinstance(row.get("duration_ns"), int):
            http_values[str(row.get("path"))].append(row["duration_ns"])
    initializations = [
        row["initialization_duration_ns"]
        for row in telemetry_rows
        if row.get("event") == "native_service_ready"
        and isinstance(row.get("initialization_duration_ns"), int)
    ]
    status_counts = Counter(
        str(row.get("http_status")) if row.get("http_status") is not None else "none"
        for row in http_rows
    )
    worker_counts = Counter()
    queue_depths = []
    for row in http_rows:
        headers = row.get("response_headers")
        if not isinstance(headers, dict):
            continue
        worker = headers.get("x-roborender-worker") or headers.get("x-worker-id")
        if worker:
            worker_counts[str(worker)] += 1
        depth = headers.get("x-queue-depth")
        if isinstance(depth, str) and depth.isdigit():
            queue_depths.append(int(depth))
    selected = Counter()
    observations = Counter()
    done_count = 0
    timeout_count = 0
    receipt_integrity = True
    for row in step_rows:
        selected.update(row.get("selected_source_counts", {}))
        observations.update(row.get("observation_source_counts", {}))
        done_count += int(row.get("done_count", 0))
        timeout_count += int(row.get("timeout_count", 0))
        for name in (
            "unique_prepare_receipts",
            "unique_capture_digests",
            "unique_selection_receipts",
            "unique_commit_receipts",
            "unique_remote_commit_receipts",
        ):
            receipt_integrity &= row.get(name) == batch_size
    error_rows = [row for row in http_rows if row.get("outcome") != "success"]
    retry_rows = [row for row in http_rows if int(row.get("attempt", 0)) > 1]
    successful_routes = Counter(
        str(row.get("path"))
        for row in http_rows
        if row.get("outcome") == "success"
    )
    if wire_protocol in {"v2", "v3"}:
        prefix = f"/{wire_protocol}"
        workers = native_result.get("rpc_workers", 8) if wire_protocol == "v3" else 1
        if not isinstance(workers, int) or isinstance(workers, bool) or workers not in {1, 2, 4, 8}:
            raise ValueError("invalid bounded gateway count in capacity evidence")
        expected_ready = workers
        expected_batches = expected_steps * workers
        anchor_route = f"{prefix}/anchors/register-batch"
        prepare_route = f"{prefix}/steps/prepare-batch"
        commit_route = f"{prefix}/steps/commit-batch"
        anchor_requests = successful_routes[anchor_route]
        http_successes_present = (
            successful_routes["/ready"] == expected_ready
            and successful_routes[prepare_route] == expected_batches
            and successful_routes[commit_route] == expected_batches
            and 0 <= anchor_requests <= expected_batches
            and set(successful_routes)
            <= {
                "/ready",
                anchor_route,
                prepare_route,
                commit_route,
            }
        )
    else:
        expected_http_successes = expected_steps * int(batch_size) * 2
        http_successes_present = sum(
            row.get("outcome") == "success" for row in http_rows
        ) == expected_http_successes
    transaction = native_result.get("transaction", {})
    acceptance = {
        "exact_vector_steps": len(step_rows) == expected_steps,
        "exact_physical_transition_rows": (
            transaction.get("physical_transition_rows") == expected_steps * batch_size
        ),
        "bounded_non_sealing_probe": (
            transaction.get("bounded_non_sealing_probe") is True
            and native_result.get("chunk_persistence") is False
        ),
        "all_http_successes_present": http_successes_present,
        "no_http_errors": not error_rows,
        "no_http_retries": not retry_rows,
        "unique_receipts_per_barrier": receipt_integrity,
        "resource_sampler_has_samples": resource_summary.get("sample_count", 0) > 0,
        "resource_sampler_has_no_errors": resource_summary.get("sample_error_count") == 0,
    }
    report = {
        "schema": "roborender.dagger.rtx.capacity-report.v1",
        "result": "PASS" if all(acceptance.values()) else "FAIL",
        "run": {
            "run_id": transaction.get("final", {}).get("step_keys", [{}])[0].get("run_id"),
            "physical_gpu": native_result.get("physical_gpu"),
            "batch_size": batch_size,
            "rpc_workers": native_result.get("rpc_workers"),
            "wire_protocol": wire_protocol,
            "vector_steps": len(step_rows),
            "physical_transition_rows": len(step_rows) * int(batch_size),
            "initialization_duration": latency_summary_ns(initializations),
        },
        "acceptance": acceptance,
        "latency": {
            "vector_phases": {
                name: latency_summary_ns(values)
                for name, values in sorted(phase_values.items())
            },
            "http_round_trip_by_route": {
                name: latency_summary_ns(values)
                for name, values in sorted(http_values.items())
            },
            "vector_step_total": latency_summary_ns(
                row.get("timings_ns", {}).get("vector_step_total")
                for row in step_rows
                if isinstance(row.get("timings_ns", {}).get("vector_step_total"), int)
            ),
        },
        "network": {
            "request_body_bytes": sum(int(row.get("request_body_bytes", 0)) for row in http_rows),
            "response_body_bytes": sum(int(row.get("response_body_bytes", 0)) for row in http_rows),
            "request_count": len(http_rows),
            "http_status_counts": dict(sorted(status_counts.items())),
            "error_count": len(error_rows),
            "retry_attempt_count": len(retry_rows),
        },
        "outcomes": {
            "teacher": selected["teacher"],
            "student": selected["student"],
            "generated_controls": observations["generated_controls"],
            "raytraced": observations["raytraced"],
            "done": done_count,
            "timeout": timeout_count,
        },
        "h200_observed_headers": {
            "worker_counts": dict(sorted(worker_counts.items())),
            "queue_depth_max": max(queue_depths) if queue_depths else None,
        },
        "resources": resource_summary,
        "measurement_boundary": {
            "h200_prepare_barrier": (
                "RTX measures one combined render+student+selection round trip; internal "
                "render/student split and H200 VRAM require the H200 pool telemetry."
            ),
            "body_bytes": "canonical JSON request/response bodies; TCP/IP and HTTP headers excluded",
            "v2_body_bytes": (
                "RDB2 binary anchor/prepare and canonical JSON commit bodies; TCP/IP and HTTP headers excluded"
                if wire_protocol == "v2"
                else None
            ),
            "v3_body_bytes": (
                "RDB3 zlib-1 anchor/prepare shards and canonical JSON commit shards; "
                "TCP/IP and HTTP headers excluded"
                if wire_protocol == "v3"
                else None
            ),
        },
    }
    return report


def build_from_paths(
    *,
    native_result_path: Path,
    telemetry_path: Path,
    resource_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    native = json.loads(native_result_path.read_text(encoding="utf-8"))
    resources = json.loads(resource_summary_path.read_text(encoding="utf-8"))
    report = build_capacity_report(
        native_result=native,
        telemetry_rows=read_jsonl(telemetry_path),
        resource_summary=resources,
    )
    atomic_write_json(output_path, report)
    return report
