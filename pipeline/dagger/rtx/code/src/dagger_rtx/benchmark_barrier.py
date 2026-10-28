"""Optional cold-path start gate for two bounded benchmark actors only."""
from pathlib import Path
import json
import os
import re
import time


def wait_for_benchmark_peer(mode: str, actor_prefix: str, *, timeout_s: float = 360) -> None:
    directory = os.environ.get("DAGGER_BENCHMARK_BARRIER_DIR")
    if not directory:
        return
    if mode not in {"capture-only", "capacity-probe", "single-window", "multi-window-check", "beta-production"} or directory != "/benchmark-barrier":
        raise RuntimeError("benchmark start gate is restricted to bounded native tests")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", actor_prefix):
        raise RuntimeError("invalid benchmark actor identity")
    root = Path(directory)
    marker = root / (actor_prefix + ".ready.json")
    with marker.open("x") as stream:
        json.dump({"actor_prefix": actor_prefix, "pid": os.getpid(), "ready_time_ns": time.time_ns()}, stream)
    started = time.monotonic()
    while not (root / "GO").is_file():
        if (root / "ABORT").exists() or time.monotonic() - started > timeout_s:
            raise RuntimeError("bounded benchmark peer did not become ready")
        time.sleep(0.2)
