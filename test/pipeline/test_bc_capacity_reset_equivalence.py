"""CPU-only: full writer buffers are independent of done/reset emission timing."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

from pipeline.bc_training import production_protocol as checker


def fixture():
    execution = {"source_release_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64}
    rule = dict(minimum_full_flushes_per_lane=2, maximum_growth_fraction=.10, absolute_growth_slack_bytes=256 << 20)
    models = {key: "c" * 64 for key in ("teacher", "homie_walk", "homie_stand")}
    protocol = {"collector_execution": execution, "capacity_equivalence": rule, "model_sha256": models,
        "execution_topology": {"host_id": "leo-rtx-6k", "concurrent_collectors": 1,
            "cgroup_limit_bytes_per_collector": 43 << 30, "minimum_host_reserve_bytes": 8 << 30}}
    receipt = dict(schema="roborender.native-bc-capacity.v2", purpose="full_bc_capture_and_save", verdict="PASS",
        native_runtime_tested=True, collector_execution=execution, native_collector_sha256=checker.NATIVE_COLLECTOR_SHA256,
        model_sha256=models, num_envs=2, host_id="leo-rtx-6k", concurrent_collectors=1,
        simultaneous_collectors_observed=1, cgroup_limit_bytes=43 << 30, runtime_guard_enabled=True,
        active_guard_minimum_host_available_bytes=8 << 30, bounded_repeated_buffer_only=True,
        oom_events_delta=0, swap_bytes_peak=0, plateau_rule=rule, per_flush_memory=[])
    for index, counts in enumerate(([0, 1], [1, 2], [2, 2]), start=1):
        receipt["per_flush_memory"].append(dict(vector_steps=81 * index, emitted_chunks_per_lane=counts,
            memory_current_bytes=25 << 30, memory_peak_bytes=29 << 30, anon_bytes=20 << 30,
            shmem_bytes=0, kernel_bytes=1 << 30, file_bytes=4 << 30, host_available_bytes=20 << 30))
    return protocol, receipt


def test_legitimate_reset_skew_full_buffers_are_admitted():
    protocol, receipt = fixture()
    checker.validate_bounded_capacity(receipt, protocol)


@pytest.mark.parametrize("fault", ["decreasing", "incomplete_lane", "one_flush", "duplicate_steps", "partial_window", "memory_growth", "no_guard", "wrong_runtime"])
def test_core_identity_and_full_buffer_guards_remain(fault):
    protocol, receipt = fixture()
    receipt = deepcopy(receipt)
    rows = receipt["per_flush_memory"]
    if fault == "decreasing": rows[1]["emitted_chunks_per_lane"] = [1, 0]
    elif fault == "incomplete_lane": rows[-1]["emitted_chunks_per_lane"] = [2, 1]
    elif fault == "one_flush": receipt["per_flush_memory"] = rows[-1:]
    elif fault == "duplicate_steps": rows[-1]["vector_steps"] = rows[-2]["vector_steps"]
    elif fault == "partial_window": rows[-1]["vector_steps"] += 1
    elif fault == "memory_growth": rows[-1]["anon_bytes"] = 30 << 30
    elif fault == "no_guard": receipt["runtime_guard_enabled"] = False
    else: receipt["collector_execution"]["runtime_contract_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        checker.validate_bounded_capacity(receipt, protocol)
