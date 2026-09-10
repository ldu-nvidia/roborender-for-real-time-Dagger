#!/usr/bin/env python3
"""Continuously exercise eight read-only H200 /ready routes concurrently.

This is an isolated socket-concurrency regression aid.  It never calls a
state-changing route and never reads or emits an authentication credential.
One worker is permanently assigned to each gateway so every gateway is
pinged throughout concurrent Image3F prepare traffic.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT_V3 = "roborender.dagger.rtx.v3"
SCHEMA_V1 = "roborender.dagger.rtx.v1"
REQUIRED_FEATURES = (
    "minimal_compressed_prepare_batch",
    "content_addressed_anchor_cache",
    "source_specific_observation_payload",
    "learner_only_persistence",
    "shared_filesystem_chunk_publish",
    "batched_commit",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temp, path)


def validate_ready(value: Any, expected_pid: int) -> None:
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    if value.get("ok") is not True:
        raise ValueError("ok is not true")
    if value.get("role") != "render_student":
        raise ValueError("role mismatch")
    if value.get("schema") != SCHEMA_V1:
        raise ValueError("schema mismatch")
    if value.get("pid") != expected_pid:
        raise ValueError(f"PID mismatch: {value.get('pid')!r} != {expected_pid}")
    if CONTRACT_V3 not in value.get("supported_contract_versions", []):
        raise ValueError("v3 is not advertised")
    features = value.get("features", {})
    missing = [name for name in REQUIRED_FEATURES if features.get(name) is not True]
    if missing:
        raise ValueError(f"missing v3 features: {missing}")
    if value.get("fake") is True or value.get("fake_backend") is True:
        raise ValueError("fake backend advertised")


@dataclass
class GatewayStats:
    port: int
    expected_pid: int
    attempts: int = 0
    successes: int = 0
    errors: int = 0
    first_started_at_utc: str | None = None
    last_finished_at_utc: str | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float = 0.0
    latency_sum_ms: float = 0.0
    last_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, started: str, finished: str, elapsed_ms: float, error: str | None) -> None:
        with self.lock:
            self.attempts += 1
            self.first_started_at_utc = self.first_started_at_utc or started
            self.last_finished_at_utc = finished
            self.min_latency_ms = elapsed_ms if self.min_latency_ms is None else min(self.min_latency_ms, elapsed_ms)
            self.max_latency_ms = max(self.max_latency_ms, elapsed_ms)
            self.latency_sum_ms += elapsed_ms
            if error is None:
                self.successes += 1
            else:
                self.errors += 1
                self.last_error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            mean = self.latency_sum_ms / self.attempts if self.attempts else None
            return {
                "port": self.port,
                "expected_pid": self.expected_pid,
                "attempts": self.attempts,
                "successes": self.successes,
                "errors": self.errors,
                "first_started_at_utc": self.first_started_at_utc,
                "last_finished_at_utc": self.last_finished_at_utc,
                "latency_ms_mean": mean,
                "latency_ms_min": self.min_latency_ms,
                "latency_ms_max": self.max_latency_ms if self.attempts else None,
                "last_error": self.last_error,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", default="18620,18621,18622,18623,18624,18625,18626,18627")
    parser.add_argument("--expected-pids", required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    args.ports = [int(item) for item in args.ports.split(",")]
    args.expected_pids = [int(item) for item in args.expected_pids.split(",")]
    if len(args.ports) != 8 or len(set(args.ports)) != 8:
        parser.error("exactly eight distinct ports are required")
    if len(args.expected_pids) != 8 or len(set(args.expected_pids)) != 8:
        parser.error("exactly eight distinct expected PIDs are required")
    if args.interval_ms < 20:
        parser.error("interval must be at least 20 ms")
    if args.timeout_ms < 100:
        parser.error("timeout must be at least 100 ms")
    return args


def main() -> int:
    args = parse_args()
    args.journal.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.summary.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.stop_file.exists():
        raise SystemExit(f"refusing stale stop file: {args.stop_file}")

    started_at = utc_now()
    started_monotonic = time.monotonic()
    stop = threading.Event()
    stats = [GatewayStats(port, pid) for port, pid in zip(args.ports, args.expected_pids, strict=True)]
    journal_lock = threading.Lock()
    start_barrier = threading.Barrier(len(stats))

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with args.journal.open("a", encoding="utf-8", buffering=1) as journal:
        def worker(item: GatewayStats) -> None:
            url = f"http://127.0.0.1:{item.port}/ready"
            start_barrier.wait()
            while not stop.is_set() and not args.stop_file.exists():
                wall_started = utc_now()
                monotonic_started = time.monotonic()
                error: str | None = None
                try:
                    request = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(request, timeout=args.timeout_ms / 1000.0) as response:
                        payload = response.read()
                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}")
                    validate_ready(json.loads(payload), item.expected_pid)
                except Exception as exc:  # evidence is recorded; hammer remains continuous
                    error = f"{type(exc).__name__}: {exc}"
                elapsed_ms = (time.monotonic() - monotonic_started) * 1000.0
                wall_finished = utc_now()
                item.record(
                    started=wall_started,
                    finished=wall_finished,
                    elapsed_ms=elapsed_ms,
                    error=error,
                )
                event = {
                    "schema": "roborender.dagger.ready-hammer.v1",
                    "event": "ready_probe",
                    "port": item.port,
                    "expected_pid": item.expected_pid,
                    "started_at_utc": wall_started,
                    "finished_at_utc": wall_finished,
                    "elapsed_ms": elapsed_ms,
                    "status": "ok" if error is None else "error",
                    "error": error,
                }
                with journal_lock:
                    journal.write(canonical_json(event) + "\n")
                stop.wait(args.interval_ms / 1000.0)

        threads = [threading.Thread(target=worker, args=(item,), name=f"ready-{item.port}") for item in stats]
        for thread in threads:
            thread.start()
        while not stop.is_set() and not args.stop_file.exists():
            stop.wait(0.1)
        stop.set()
        for thread in threads:
            thread.join(timeout=args.timeout_ms / 1000.0 + 1.0)

    gateway_stats = [item.snapshot() for item in stats]
    total_attempts = sum(item["attempts"] for item in gateway_stats)
    total_successes = sum(item["successes"] for item in gateway_stats)
    total_errors = sum(item["errors"] for item in gateway_stats)
    complete = all(item["attempts"] > 0 for item in gateway_stats)
    summary = {
        "schema": "roborender.dagger.ready-hammer.v1",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_s": time.monotonic() - started_monotonic,
        "ports": args.ports,
        "expected_pids": args.expected_pids,
        "interval_ms": args.interval_ms,
        "timeout_ms": args.timeout_ms,
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "total_errors": total_errors,
        "all_gateways_sampled": complete,
        "pass": complete and total_errors == 0,
        "gateways": gateway_stats,
    }
    atomic_json(args.summary, summary)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
