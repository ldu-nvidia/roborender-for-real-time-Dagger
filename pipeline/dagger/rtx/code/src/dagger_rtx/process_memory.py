"""Opt-in release of unused CPU heap at native lifecycle barriers, never per step."""
from __future__ import annotations

import ctypes
import gc
import json
import os
from pathlib import Path
import time


def compact_host_heap(phase):
    if os.environ.get("ROBORENDER_RTX_COMPACT_HEAP") != "1":
        return None
    if os.environ.get("MALLOC_ARENA_MAX") != "2":
        raise RuntimeError("compact heap mode requires MALLOC_ARENA_MAX=2 before process startup")
    def rss():
        return next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmRSS:"))
    before = rss()
    started = time.monotonic()
    gc.collect()
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    released = trim(0)
    report = {"phase": "cpu_heap_compaction", "boundary": phase,
              "rss_kib_before": before, "rss_kib_after": rss(),
              "allocator_released_pages": bool(released), "seconds": time.monotonic() - started}
    print(json.dumps(report), flush=True)
    return report
