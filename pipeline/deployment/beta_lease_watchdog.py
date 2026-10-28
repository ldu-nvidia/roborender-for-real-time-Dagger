"""CPU-only orphan guard for one round's approved GPU allocation.

Never restarts simulation. A missing scheduler heartbeat triggers scoped
cleanup via the same pinned site binding; network failures remain visible and
are retried without pretending capacity was released. Successful cleanup stops
owned work and retains the allocation until an explicit user release request.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.beta_site import Site


def heartbeat_fresh(value, run_id, now_ns):
    return (isinstance(value, dict) and value.get("run_id") == run_id
            and type(value.get("updated_at_ns")) is int
            and -5_000_000_000 <= now_ns - value["updated_at_ns"] <= 300_000_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--site-sha256", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--resident-pool", type=Path)
    parser.add_argument("--resident-pool-sha256")
    args = parser.parse_args()
    if hashlib.sha256(args.site.read_bytes()).hexdigest() != args.site_sha256:
        raise RuntimeError("watchdog site identity differs")
    request = json.loads(args.request.read_bytes())
    if args.resident_pool is not None:
        if not args.resident_pool_sha256:
            raise ValueError("resident pool hash required")
        from pipeline.deployment.resident_campaign_site import ResidentSite
        site = ResidentSite(json.loads(args.site.read_bytes()), request, args.resident_pool, args.resident_pool_sha256)
    else:
        site = Site(json.loads(args.site.read_bytes()), request)
    evidence = site.round / "lease-watchdog.json"
    while not any((site.round / name).exists() for name in ("released.json", "retained.json")):
        try:
            heartbeat = json.loads((site.root / "heartbeat.json").read_bytes())
        except (OSError, ValueError):
            heartbeat = None
        if not heartbeat_fresh(heartbeat, site.run, time.time_ns()):
            try:
                receipt = site.abort()
                atomic(evidence, {"scheduler_lost": True, "native_restart_allowed": False, **receipt})
                return
            except Exception as exc:
                atomic(evidence, {"scheduler_lost": True, "native_restart_allowed": False,
                                  "cleanup_confirmed": False, "error_type": type(exc).__name__,
                                  "updated_at_ns": time.time_ns()})
        time.sleep(15)


if __name__ == "__main__":
    main()
