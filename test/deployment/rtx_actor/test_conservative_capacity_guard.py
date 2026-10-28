from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
import pytest
from run_throughput_pair import conservative_capacity_safety

GIB = 1024**3


def fixtures():
    memory = {"MemAvailable": 20*1024**2, "SwapFree": 30*1024**2}
    sample = {"stat": {"anon": 12*GIB, "shmem": GIB, "kernel": GIB},
              "events": {"oom": 0, "oom_kill": 0}, "memory_swap.current": 0}
    return memory, sample, ["0, 20000, 97887, 50, 150", "1, 21000, 97887, 40, 140"]


def test_safe_pair_with_cache_pressure_not_mistaken_for_private_memory():
    memory, sample, gpus = fixtures()
    sample["stat"]["file"] = 8*GIB
    sample["events"]["max"] = 12
    conservative_capacity_safety(memory, memory, [sample, sample], gpus, 21)


@pytest.mark.parametrize("reason", ["host", "private", "swap", "oom", "gpu", "missing"])
def test_stop_before_exhaustion(reason):
    initial, sample, gpus = fixtures()
    current = deepcopy(initial)
    if reason == "host": current["MemAvailable"] = 11*1024**2
    if reason == "private": sample["stat"]["anon"] = 15*GIB
    if reason == "swap": sample["memory_swap.current"] = 1
    if reason == "oom": sample["events"]["oom"] = 1
    if reason == "gpu": gpus[0] = "0, 75000, 97887, 80, 250"
    if reason == "missing": sample.pop("stat")
    with pytest.raises(RuntimeError):
        conservative_capacity_safety(initial, current, [sample], gpus, 21)


def test_capacity_receipts_are_host_readable_not_root_private():
    """Container UID0 publication must be readable before host-side chown."""
    import ast
    from pathlib import Path
    tree = ast.parse((ACTOR_ROOT / 'isaac6_capacity_trainer.py').read_text())
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == 'atomic_write_json']
    assert len(writes) == 3
    assert all(any(k.arg == 'mode' and isinstance(k.value, ast.Constant)
                   and k.value.value == 0o644 for k in n.keywords) for n in writes)
