from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = ACTOR_ROOT
sys.path.insert(0, str(ROOT))
from run_bc_pair_capacity import admission, safety, validate_spec, verify_requests


def spec():
    return {"schema": "roborender.bc-pair-capacity.v1", "pair_id": "bc-pair-leo-test",
        "source_root": "/source", "source_manifest": "/manifest", "source_sha256": "a" * 64,
        "runtime_root": "/runtime", "timeout_seconds": 1800,
        "actors": [{"physical_gpu": i, "session_file": f"/s{i}", "session_sha256": str(i + 1) * 64,
                    "requests_dir": f"/r{i}"} for i in range(2)]}


def test_cpu_import_does_not_initialize_native_or_torch():
    subprocess.run([sys.executable, "-c", f"import sys;sys.path.insert(0,{str(ROOT)!r});import run_bc_pair_capacity;assert 'torch' not in sys.modules"], check=True)


def test_fixed_two_gpu_scope_and_bounded_requests():
    assert validate_spec(spec()) == spec()
    verify_requests({"requests": [1]}, [{"motions": [1, 2], "chunks_per_env": [1, 1], "max_windows": 4}])
    for field, value in [("motions", list(range(4))), ("chunks_per_env", [2, 1]), ("max_windows", 5)]:
        row = {"motions": [1, 2], "chunks_per_env": [1, 1], "max_windows": 4}
        row[field] = value
        with pytest.raises(ValueError):
            verify_requests({"requests": [1]}, [row])


@pytest.mark.parametrize("mutate", [
    lambda v: v["actors"][1].update(physical_gpu=2),
    lambda v: v["actors"].reverse(),
    lambda v: v.update(timeout_seconds=9999),
    lambda v: v["actors"][1].update(session_sha256="1" * 64),
])
def test_unsafe_scope_rejected(mutate):
    value = spec()
    mutate(value)
    with pytest.raises(ValueError):
        validate_spec(value)


def test_exact_admission_floor_and_stricter_runtime_floor():
    admission({"MemAvailable": 52 * 1024**2})
    with pytest.raises(RuntimeError):
        admission({"MemAvailable": 52 * 1024**2 - 1})
    called = []
    gate = lambda *args: called.append(args) or {"ok": True}
    assert safety({}, {"MemAvailable": 8 * 1024**2}, [], gate) == {"ok": True}
    assert len(called) == 1
    with pytest.raises(RuntimeError):
        safety({}, {"MemAvailable": 8 * 1024**2 - 1}, [], gate)
    with pytest.raises(RuntimeError):
        safety({}, {"MemAvailable": 10 * 1024**2}, [{"events": {"oom": 1}}], gate)
    assert len(called) == 1
