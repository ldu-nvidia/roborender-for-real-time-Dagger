#!/usr/bin/env python3
"""Discover and actively health-check an arbitrary-size Image3F endpoint fleet."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("server closed connection")
        data.extend(chunk)
    return bytes(data)


def ping(path: Path, timeout_s: float) -> dict[str, object]:
    started = time.perf_counter()
    fields = path.read_text(encoding="utf-8").split()
    if len(fields) != 2:
        raise ValueError("endpoint file must contain exactly 'host port'")
    host, port_s = fields
    port = int(port_s)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        payload = pickle.dumps(
            {"cmd": "ping", "protocol_version": 2}, protocol=pickle.HIGHEST_PROTOCOL
        )
        sock.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", recv_exact(sock, 8))[0]
        response = pickle.loads(recv_exact(sock, size))  # noqa: S301 - trusted cluster
    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"unhealthy response: {response!r}")
    return {
        "file": str(path.resolve()),
        "host": host,
        "port": port,
        "rtt_ms": (time.perf_counter() - started) * 1000.0,
        "response": response,
    }


def probe(paths: list[Path], timeout_s: float) -> tuple[list[dict[str, object]], list[str]]:
    healthy: list[dict[str, object]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(64, max(1, len(paths)))) as pool:
        futures = {pool.submit(ping, path, timeout_s): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                healthy.append(future.result())
            except Exception as exc:  # report every endpoint; the caller decides whether to retry
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    healthy.sort(key=lambda item: item["file"])
    errors.sort()
    return healthy, errors


def capability_error(
    response: dict[str, object],
    *,
    minimum_protocol_version: int,
    required_reference_lag: int,
    require_reference_history: bool,
) -> str | None:
    """Return why a healthy endpoint is incompatible, or ``None``."""
    capabilities = response.get("capabilities", {})
    if not isinstance(capabilities, dict):
        return "capabilities must be a mapping"
    try:
        protocol_version = int(response.get("protocol_version", 0))
        max_lag = int(capabilities.get("reference_lag_max", 0))
    except (TypeError, ValueError) as exc:
        return f"malformed numeric capability: {exc}"
    if protocol_version < minimum_protocol_version:
        return f"protocol {protocol_version} < required {minimum_protocol_version}"
    history_required = require_reference_history or required_reference_lag > 0
    if history_required and not capabilities.get("lagged_reference_history"):
        return "lagged reference history is not supported"
    if history_required and max_lag < required_reference_lag:
        return f"maximum reference lag {max_lag} < required {required_reference_lag}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--wait-timeout-s", type=float, default=3600)
    parser.add_argument("--connect-timeout-s", type=float, default=10)
    parser.add_argument("--poll-s", type=float, default=10)
    parser.add_argument("--require-reference-history", action="store_true")
    parser.add_argument("--required-reference-lag", type=int, default=0)
    parser.add_argument("--minimum-protocol-version", type=int, default=2)
    parser.add_argument("--write-manifest", type=Path, default=None)
    parser.add_argument("--hydra-list", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.expected < 0:
        raise SystemExit("--expected must be non-negative")
    if args.required_reference_lag < 0:
        raise SystemExit("--required-reference-lag must be non-negative")
    if args.expected == 0:
        print("[]" if args.hydra_list else json.dumps({"healthy": [], "count": 0}))
        return 0

    deadline = time.monotonic() + args.wait_timeout_s
    last_report: dict[str, object] = {}
    while True:
        paths = sorted(args.endpoint_dir.glob("renderer_*.txt"))
        healthy, errors = probe(paths, args.connect_timeout_s) if paths else ([], [])
        unique = {(item["host"], item["port"]) for item in healthy}
        duplicate_count = len(healthy) - len(unique)
        capable = []
        for item in healthy:
            response = item["response"]
            incompatible = capability_error(
                response,
                minimum_protocol_version=args.minimum_protocol_version,
                required_reference_lag=args.required_reference_lag,
                require_reference_history=args.require_reference_history,
            )
            if incompatible is None:
                capable.append(item)
            else:
                errors.append(f"{item['file']}: incompatible endpoint: {incompatible}")
        last_report = {
            "endpoint_dir": str(args.endpoint_dir.resolve()),
            "expected": args.expected,
            "discovered": len(paths),
            "healthy_count": len(healthy),
            "capable_count": len(capable),
            "duplicate_count": duplicate_count,
            "healthy": healthy,
            "errors": errors,
        }
        if len(capable) >= args.expected and duplicate_count == 0:
            selected = capable[: args.expected]
            if args.write_manifest:
                args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.write_manifest.with_name(
                    f".{args.write_manifest.name}.{os.getpid()}.tmp"
                )
                tmp.write_text(
                    "".join(f"{item['file']}\n" for item in selected), encoding="utf-8"
                )
                os.replace(tmp, args.write_manifest)
            if args.hydra_list:
                print("[" + ",".join(item["file"] for item in selected) + "]")
            elif args.json:
                print(json.dumps(last_report, indent=2, sort_keys=True))
            else:
                print(
                    f"READY: {len(capable)} healthy/capable Image3F endpoints "
                    f"(required {args.expected})"
                )
                for item in selected:
                    print(
                        f"  {Path(item['file']).name}: {item['host']}:{item['port']} "
                        f"ping={item['rtt_ms']:.1f}ms"
                    )
            return 0
        if time.monotonic() >= deadline:
            print(json.dumps(last_report, indent=2, sort_keys=True))
            print(
                f"FATAL: only {len(capable)}/{args.expected} healthy endpoints became ready",
                flush=True,
            )
            return 2
        print(
            f"waiting for renderers: discovered={len(paths)} healthy={len(healthy)} "
            f"capable={len(capable)} required={args.expected}",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
