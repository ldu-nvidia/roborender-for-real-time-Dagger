from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.boundary import H200_COMMIT_PATH, H200_PREPARE_PATH
from dagger_rtx.capacity_report import build_capacity_report, percentile
from dagger_rtx.capacity_safety import (
    attest_group_container_pids,
    parse_collector_processes,
    validate_capacity_group_id,
    validate_collector_ownership,
)
from dagger_rtx.capacity_telemetry import AppendOnlyTelemetry, RUN_TELEMETRY_SCHEMA
from dagger_rtx.contracts import StepKey
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.fake_h200 import FakeH200Server, FakeH200State
from dagger_rtx.http_transport import BoundedBearerJsonTransport
from dagger_rtx.resource_monitor import parse_size_bytes, summarize_resource_samples
from dagger_rtx.window_runner import BatchWindowRunner, TransportRetryPolicy


PROJECT = Path(__file__).resolve().parents[1]


def _token(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text("test-token\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _batch_stack(tmp_path, tiny_profile, *, num_envs=2, telemetry=None, transport=None):
    backend = DeterministicFakeBatchRaven(tiny_profile, num_envs=num_envs)
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "owner",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
        persist_chunks=False,
        telemetry=telemetry,
    )
    if transport is None:
        remote = FakeH200Server(shape_profile=tiny_profile)
        remote.__enter__()
        transport = BoundedBearerJsonTransport(
            remote.base_url, _token(tmp_path), telemetry=telemetry
        )
    else:
        remote = None
    orchestrator = BatchStepOrchestrator(
        owner,
        transport,
        tmp_path / "orchestrator",
        num_envs=num_envs,
        max_parallel_requests=min(8, num_envs),
        telemetry=telemetry,
    )
    runner = BatchWindowRunner(
        owner,
        orchestrator,
        run_id="capacity-test-run",
        round_id=1,
        actor_prefix="capacity-test",
        retry_policy=TransportRetryPolicy(
            max_attempts=2, initial_backoff_s=0, max_backoff_s=0
        ),
        sleep=lambda _: None,
    )
    return backend, owner, runner, remote


def test_bounded_capacity_probe_is_exactly_ten_steps_and_writes_no_chunk(
    tmp_path, tiny_profile
):
    telemetry = AppendOnlyTelemetry(tmp_path / "capacity.jsonl")
    backend, owner, runner, remote = _batch_stack(
        tmp_path, tiny_profile, telemetry=telemetry
    )
    try:
        result = runner.run_bounded_probe(0, steps=10)
        assert result["transition_batches"] == 10
        assert result["physical_transition_rows"] == 20
        assert result["bounded_non_sealing_probe"] is True
        assert backend.batch_step_calls == 10
        assert owner.status()["next_rollout_step"] == 10
        assert owner.status()["phase"] == "READY"
        assert owner.status()["persist_chunks"] is False
        assert not list((tmp_path / "staging").rglob("*.npz"))
        assert not list((tmp_path / "staging").rglob("*.json"))
        assert remote.state.prepare_effects == 20
        assert remote.state.commit_effects == 20
        step_rows = [
            row for row in telemetry.read() if row["event"] == "vector_step_complete"
        ]
        assert len(step_rows) == 10
        assert all(row["unique_remote_commit_receipts"] == 2 for row in step_rows)
    finally:
        remote.__exit__(None, None, None)


def test_capacity_probe_refuses_sealing_boundary(tmp_path, tiny_profile):
    backend, owner, runner, remote = _batch_stack(tmp_path, tiny_profile)
    try:
        with pytest.raises(Exception, match="bounded probe steps"):
            runner.run_bounded_probe(0, steps=81)
        assert backend.batch_step_calls == 0
        assert owner.status()["phase"] == "EMPTY"
    finally:
        remote.__exit__(None, None, None)


class _ConcurrentInMemoryH200:
    def __init__(self, profile):
        self.state = FakeH200State("unused", shape_profile=profile)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.completion_order: list[tuple[str, str]] = []

    def post(self, path, payload, *, max_response_bytes):
        lane = payload["step_key"]["actor_id"]
        lane_index = int(lane.rsplit("-", 1)[-1])
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            # Inverse delay forces completion order to differ from input order.
            time.sleep((8 - lane_index) * 0.001)
            with self.lock:
                response = (
                    self.state.prepare(payload)
                    if path == H200_PREPARE_PATH
                    else self.state.commit(payload)
                )
                self.completion_order.append((path, lane))
            return response
        finally:
            with self.lock:
                self.active -= 1


def test_eight_way_h200_fanout_is_concurrent_and_result_order_is_stable(
    tmp_path, tiny_profile
):
    transport = _ConcurrentInMemoryH200(tiny_profile)
    backend, owner, runner, _ = _batch_stack(
        tmp_path,
        tiny_profile,
        num_envs=8,
        transport=transport,
    )
    runner.begin_window(0)
    result = runner.run_step(0, 0)
    expected = [f"capacity-test-lane-{lane:03d}" for lane in range(8)]
    assert transport.max_active == 8
    assert [row["step_key"]["actor_id"] for row in result["selections"]] == expected
    assert [
        row["step_key"]["actor_id"] for row in result["remote_acknowledgements"]
    ] == expected
    assert backend.batch_step_calls == 1
    assert backend.selected_batches[0].shape == (8, 26)
    completed_prepares = [
        lane for path, lane in transport.completion_order if path == H200_PREPARE_PATH
    ]
    assert completed_prepares != expected


def test_http_telemetry_records_status_retry_attempt_and_body_bytes(
    tmp_path, tiny_profile
):
    telemetry = AppendOnlyTelemetry(tmp_path / "http.jsonl")
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        remote.state.fail_next_prepare = 1
        transport = BoundedBearerJsonTransport(
            remote.base_url, _token(tmp_path), telemetry=telemetry
        )
        backend, _, runner, _ = _batch_stack(
            tmp_path,
            tiny_profile,
            telemetry=telemetry,
            transport=transport,
        )
        runner.begin_window(0)
        result = runner.run_step(0, 0)
    assert result["phase"] == "COMPLETE"
    assert backend.batch_step_calls == 1
    http = [row for row in telemetry.read() if row["event"] == "http_request"]
    assert any(row["http_status"] == 503 and row["retryable"] for row in http)
    assert any(row["attempt"] == 2 and row["outcome"] == "success" for row in http)
    assert all(row["request_body_bytes"] > 0 for row in http)
    assert all(row["duration_ns"] > 0 for row in http)


def test_capacity_group_gate_is_exact_not_a_global_bypass():
    rows = parse_collector_processes("101 python run_raven_bc.py\n102 distill_trainer_videogen")
    with pytest.raises(RuntimeError, match="active Raven collector"):
        validate_collector_ownership(rows, capacity_group_id=None, allowed_pids=set())
    with pytest.raises(RuntimeError, match="not owned"):
        validate_collector_ownership(
            rows, capacity_group_id="capacity-a", allowed_pids={101}
        )
    validate_collector_ownership(
        rows, capacity_group_id="capacity-a", allowed_pids={101, 102}
    )
    with pytest.raises(RuntimeError, match="safe identifier"):
        validate_capacity_group_id("../../escape")


def test_capacity_group_container_attestation_requires_other_gpu_and_pinned_image():
    image = "sha256:" + "a" * 64

    def run(command):
        if command[:2] == ["docker", "ps"]:
            return "abc123"
        if command[:2] == ["docker", "inspect"]:
            return json.dumps(
                {
                    "Image": image,
                    "State": {"Status": "running"},
                    "Config": {
                        "Labels": {
                            "roborender.capacity.group": "capacity-a",
                            "roborender.capacity.physical-gpu": "0",
                            "roborender.capacity.owner-pid": "555",
                        }
                    },
                }
            )
        if command[:2] == ["docker", "top"]:
            return "PID\n1001\n1002"
        raise AssertionError(command)

    allowed, details = attest_group_container_pids(
        "capacity-a",
        selected_gpu=1,
        expected_image_id=image,
        run=run,
        read_cmdline=lambda _pid: b"bash\0/home/ldu/roborender/pipeline/dagger/rtx/scripts/launch_native_raven.sh",
    )
    assert allowed == {1001, 1002}
    assert details[0]["physical_gpu"] == 0
    with pytest.raises(RuntimeError, match="already owns"):
        attest_group_container_pids(
            "capacity-a",
            selected_gpu=0,
            expected_image_id=image,
            run=run,
            read_cmdline=lambda _pid: b"launch_native_raven.sh",
        )


def test_resource_and_capacity_report_aggregation():
    assert parse_size_bytes("1.5 GiB") == int(1.5 * 1024**3)
    resources = summarize_resource_samples(
        [
            {
                "event": "resource_sample",
                "gpu_memory_used_mib": 100,
                "gpu_utilization_percent": 20,
                "host_memory_available_bytes": 500,
                "host_swap_used_bytes": 0,
                "container_memory_usage_bytes": 300,
            },
            {
                "event": "resource_sample",
                "gpu_memory_used_mib": 200,
                "gpu_utilization_percent": 80,
                "host_memory_available_bytes": 400,
                "host_swap_used_bytes": 10,
                "container_memory_usage_bytes": 350,
            },
        ]
    )
    assert resources["gpu_memory_used_mib_max"] == 200
    assert resources["host_memory_available_bytes_min"] == 400
    assert percentile([1, 2, 3], 0.5) == 2


def test_v2_three_step_capacity_report_counts_batched_routes_and_ready():
    batch_size = 28
    step_rows = []
    for step in range(3):
        step_rows.append(
            {
                "event": "vector_step_complete",
                "timings_ns": {"vector_step_total": 100 + step},
                "selected_source_counts": {"teacher": 21, "student": 7},
                "observation_source_counts": {
                    "generated_controls": 14,
                    "raytraced": 14,
                },
                "done_count": 0,
                "timeout_count": 0,
                "unique_prepare_receipts": batch_size,
                "unique_capture_digests": batch_size,
                "unique_selection_receipts": batch_size,
                "unique_commit_receipts": batch_size,
                "unique_remote_commit_receipts": batch_size,
            }
        )
    routes = ["/ready", "/v2/anchors/register-batch"] + [
        route
        for _ in range(3)
        for route in ("/v2/steps/prepare-batch", "/v2/steps/commit-batch")
    ]
    http_rows = [
        {
            "event": "http_batch_request_v2",
            "path": route,
            "outcome": "success",
            "attempt": 1,
            "http_status": 200,
            "duration_ns": 10,
            "request_body_bytes": 100,
            "response_body_bytes": 10,
        }
        for route in routes
    ]
    report = build_capacity_report(
        native_result={
            "num_envs": batch_size,
            "capacity_steps": 3,
            "wire_protocol": "v2",
            "chunk_persistence": False,
            "physical_gpu": 1,
            "rpc_workers": 8,
            "transaction": {
                "physical_transition_rows": 84,
                "bounded_non_sealing_probe": True,
                "final": {"step_keys": [{"run_id": "v2-three-step"}]},
            },
        },
        telemetry_rows=[
            {
                "event": "native_service_ready",
                "initialization_duration_ns": 10,
            },
            *step_rows,
            *http_rows,
        ],
        resource_summary={
            "sample_count": 1,
            "sample_error_count": 0,
        },
    )
    assert report["result"] == "PASS"
    assert report["run"]["vector_steps"] == 3
    assert report["run"]["physical_transition_rows"] == 84
    assert report["network"]["request_count"] == 8
    assert report["run"]["wire_protocol"] == "v2"


def test_v3_three_step_capacity_report_counts_eight_sharded_gateways():
    batch_size = 28
    step_rows = [
        {
            "event": "vector_step_complete",
            "timings_ns": {"vector_step_total": 100 + step},
            "selected_source_counts": {"teacher": 21, "student": 7},
            "observation_source_counts": {"generated_controls": 14, "raytraced": 14},
            "done_count": 0,
            "timeout_count": 0,
            "unique_prepare_receipts": batch_size,
            "unique_capture_digests": batch_size,
            "unique_selection_receipts": batch_size,
            "unique_commit_receipts": batch_size,
            "unique_remote_commit_receipts": batch_size,
        }
        for step in range(3)
    ]
    routes = ["/ready"] * 8 + [
        route
        for _ in range(3)
        for route in (["/v3/steps/prepare-batch"] * 8 + ["/v3/steps/commit-batch"] * 8)
    ]
    http_rows = [
        {
            "event": "http_batch_request_v3",
            "path": route,
            "outcome": "success",
            "attempt": 1,
            "http_status": 200,
            "duration_ns": 10,
            "request_body_bytes": 100,
            "response_body_bytes": 10,
        }
        for route in routes
    ]
    report = build_capacity_report(
        native_result={
            "num_envs": batch_size,
            "capacity_steps": 3,
            "wire_protocol": "v3",
            "chunk_persistence": False,
            "physical_gpu": 1,
            "rpc_workers": 8,
            "transaction": {
                "physical_transition_rows": 84,
                "bounded_non_sealing_probe": True,
                "final": {"step_keys": [{"run_id": "v3-three-step"}]},
            },
        },
        telemetry_rows=[
            {"event": "native_service_ready", "initialization_duration_ns": 10},
            *step_rows,
            *http_rows,
        ],
        resource_summary={"sample_count": 1, "sample_error_count": 0},
    )
    assert report["result"] == "PASS"
    assert report["run"]["vector_steps"] == 3
    assert report["run"]["physical_transition_rows"] == 84
    assert report["network"]["request_count"] == 56
    assert report["run"]["wire_protocol"] == "v3"


def test_three_step_v3_capacity_probe_requires_exact_eight_pids_before_token_gate():
    base = [
        "bash",
        str(PROJECT / "scripts/launch_native_raven.sh"),
        "capacity-probe",
        "28",
        "v3-three-step-test",
        "1",
        "v3-three-step-test-g1",
        "0",
        "/definitely/missing/token",
        "I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS",
        "--gpu",
        "1",
        "--capacity-group-id",
        "v3-three-step-test",
        "--wire-protocol",
        "v3",
        "--probe-steps",
        "3",
        "--rpc-workers",
        "8",
    ]
    missing = subprocess.run(
        base, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert missing.returncode == 2
    assert "eight ordered PIDs" in missing.stderr

    complete = subprocess.run(
        [*base, "--expected-h200-pids", "1,2,3,4,5,6,7,8"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert complete.returncode == 2
    assert "missing H200 bearer-token file" in complete.stderr


@pytest.mark.parametrize(
    "arguments,needle",
    [
        (
            [
                "capacity-probe", "8", "run", "1", "actor", "0", "/missing", "bad",
                "--gpu", "2", "--capacity-group-id", "capacity-a",
            ],
            "I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS",
        ),
        (
            [
                "capacity-probe", "8", "run", "1", "actor", "0", "/missing",
                "I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS",
            ],
            "requires --capacity-group-id",
        ),
        (
            ["capture-only", "8", "run", "1", "actor", "0", "--rpc-workers", "4"],
            "valid only for single-window, capacity-probe, or production",
        ),
    ],
)
def test_launcher_rejects_unsafe_capacity_arguments_before_any_job(arguments, needle):
    result = subprocess.run(
        ["bash", str(PROJECT / "scripts/launch_native_raven.sh"), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert needle in result.stderr


def test_launcher_is_valid_bash():
    subprocess.run(
        ["bash", "-n", str(PROJECT / "scripts/launch_native_raven.sh")],
        check=True,
    )


def test_launcher_is_hard_bound_to_physical_gpu1_uuid_and_rejects_gpu0():
    launcher = PROJECT / "scripts/launch_native_raven.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "capture-only",
            "1",
            "gpu-binding-test",
            "1",
            "gpu-binding-test-g1",
            "0",
            "--gpu",
            "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "GPU0 must never be probed or used" in result.stderr

    source = launcher.read_text(encoding="utf-8")
    expected_uuid = "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9"
    assert f"EXPECTED_PHYSICAL_GPU_UUID={expected_uuid}" in source
    assert 'nvidia-smi -i "$GPU_UUID"' in source
    assert '--gpus "device=$GPU_UUID"' in source
    assert '--env CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert '--env DAGGER_PHYSICAL_GPU_UUID="$GPU_UUID"' in source
    assert '--gpus "device=$GPU"' not in source


def test_preflight_is_uuid_scoped_and_rejects_gpu0_before_host_queries():
    preflight = PROJECT / "scripts/preflight.py"
    result = subprocess.run(
        [sys.executable, str(preflight), "--gpu", "0"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    assert "GPU0 must never be probed or used" in result.stderr

    source = preflight.read_text(encoding="utf-8")
    assert (
        'EXPECTED_PHYSICAL_GPU_UUID = "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9"'
        in source
    )
    assert '"-i",\n            EXPECTED_PHYSICAL_GPU_UUID,' in source
    assert '"gpus": [selected]' in source
    assert '"gpus": gpus' not in source


def test_three_step_v2_capacity_probe_is_explicit_and_reaches_token_gate():
    result = subprocess.run(
        [
            "bash",
            str(PROJECT / "scripts/launch_native_raven.sh"),
            "capacity-probe",
            "28",
            "v2-three-step-test",
            "1",
            "v2-three-step-g1",
            "0",
            "/definitely/missing/token",
            "I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS",
            "--gpu",
            "1",
            "--capacity-group-id",
            "v2-three-step-test",
            "--wire-protocol",
            "v2",
            "--probe-steps",
            "3",
            "--expected-h200-pid",
            "1701913",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "missing H200 bearer-token file" in result.stderr


def test_three_step_probe_rejects_ten_step_acknowledgement():
    result = subprocess.run(
        [
            "bash",
            str(PROJECT / "scripts/launch_native_raven.sh"),
            "capacity-probe",
            "28",
            "v2-three-step-test",
            "1",
            "v2-three-step-g1",
            "0",
            "/definitely/missing/token",
            "I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS",
            "--capacity-group-id",
            "v2-three-step-test",
            "--probe-steps",
            "3",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS" in result.stderr


def test_launcher_rejects_invalid_expected_h200_pid_before_preflight():
    result = subprocess.run(
        [
            "bash",
            str(PROJECT / "scripts/launch_native_raven.sh"),
            "capacity-probe",
            "28",
            "v2-three-step-test",
            "1",
            "v2-three-step-g1",
            "0",
            "/definitely/missing/token",
            "I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS",
            "--capacity-group-id",
            "v2-three-step-test",
            "--probe-steps",
            "3",
            "--expected-h200-pid",
            "not-a-pid",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "--expected-h200-pid must be a positive integer" in result.stderr


def test_single_window_accepts_eight_way_rpc_fanout_before_native_preflight():
    result = subprocess.run(
        [
            "bash",
            str(PROJECT / "scripts/launch_native_raven.sh"),
            "single-window",
            "28",
            "smoke-test",
            "1",
            "smoke-test-g0",
            "0",
            "/definitely/missing/token",
            "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW",
            "--rpc-workers",
            "8",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "missing H200 bearer-token file" in result.stderr
    assert "valid only" not in result.stderr


def test_launcher_passes_generic_rpc_workers_and_samples_single_window():
    source = (PROJECT / "scripts/launch_native_raven.sh").read_text(encoding="utf-8")
    assert '--env DAGGER_H200_RPC_WORKERS="$native_rpc_workers"' in source
    assert '--env DAGGER_CAPACITY_RPC_WORKERS="$native_rpc_workers"' not in source
    assert '[[ $mode == capacity-probe || $mode == single-window || $mode == production ]]' in source
    assert 'resource_journal="$RUN/resource-samples.jsonl"' in source
    assert 'resource_summary="$RUN/resource-summary.json"' in source
    assert '--env DAGGER_WIRE_PROTOCOL="$wire_protocol"' in source
    assert '[[ $mode == capacity-probe ]] && native_steps=$probe_steps' in source


def test_native_service_uses_generic_rpc_workers_with_legacy_fallback_and_run_telemetry():
    source = (
        PROJECT / "code/src/dagger_rtx/native_service_trainer.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ.get("DAGGER_H200_RPC_WORKERS")' in source
    assert 'os.environ.get("DAGGER_CAPACITY_RPC_WORKERS")' in source
    assert 'mode in {"single-window", "capacity-probe"}' in source
    assert 'runtime_root / "run-telemetry.jsonl"' in source


def test_single_window_telemetry_uses_distinct_append_only_schema(tmp_path):
    telemetry = AppendOnlyTelemetry(
        tmp_path / "run-telemetry.jsonl", schema=RUN_TELEMETRY_SCHEMA
    )
    telemetry.record("native_service_ready", rpc_workers=8)
    telemetry.record("native_service_complete", transition_batches=81)
    rows = telemetry.read()
    assert [row["event"] for row in rows] == [
        "native_service_ready",
        "native_service_complete",
    ]
    assert {row["schema"] for row in rows} == {RUN_TELEMETRY_SCHEMA}
