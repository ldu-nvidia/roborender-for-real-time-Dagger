from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dagger_rtx.contracts import ContractError, StepKey, encode_array
from dagger_rtx.lean_boundary_v4 import CONTRACT_VERSION_V4
from dagger_rtx.pipelined_window_v4 import (
    PipelinedBatchStepOrchestratorV4,
    TransportTimingObserverV4,
    prewarm_readiness_v4,
)


class _Clock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class _Delegate:
    def __init__(self, clock: _Clock, cost_ns: int = 0) -> None:
        self.clock = clock
        self.cost_ns = cost_ns
        self.events: list[tuple[str, dict]] = []

    def record(self, event: str, **detail) -> None:
        self.events.append((event, detail))
        self.clock.value += self.cost_ns


class _ReadyClient:
    def __init__(self, *, change_generation_at: int | None = None) -> None:
        self.calls = 0
        self.change_generation_at = change_generation_at

    def verify_ready(self):
        self.calls += 1
        suffix = (
            "changed"
            if self.change_generation_at is not None
            and self.calls >= self.change_generation_at
            else "stable"
        )
        return {
            "gateway_count": 2,
            "gateways": [
                {"pid": 7000, "gateway_instance_id": f"gateway-0-{suffix}"},
                {"pid": 7001, "gateway_instance_id": f"gateway-1-{suffix}"},
            ],
        }


class _OneLaneOwner:
    num_envs = 1

    def prepare_batch_lean_v4(self, requests):
        return [
            {
                "step_key": requests[0]["step_key"],
                "prepare_receipt": "1" * 64,
                "capture_digest": "2" * 64,
                "metadata": {"beta": requests[0]["beta"]},
                "arrays": {},
            }
        ]

    def commit_batch(self, requests):
        return [
            {
                "step_key": requests[0]["step_key"],
                "commit_receipt": "5" * 64,
                "result": {"done": False, "timeout": False},
                "sealed": False,
            }
        ]


class _TimedOneLaneH200:
    protocol_version = CONTRACT_VERSION_V4
    remote_request_count = 1

    def __init__(self, observer: TransportTimingObserverV4) -> None:
        self.observer = observer

    def prepare_batch(self, local_prepares, idempotency_keys):
        self.observer.record(
            "http_batch_request_v4",
            duration_ns=1,
            request_body_bytes=123,
            response_body_bytes=45,
            path="/v4/steps/prepare-batch",
        )
        row = local_prepares[0]
        return [
            {
                "step_key": row["step_key"],
                "prepare_receipt": row["prepare_receipt"],
                "capture_digest": row["capture_digest"],
                "selection_id": "3" * 64,
                "selection_receipt": "4" * 64,
                "selected_source": "teacher",
                "beta": 0.75,
                "selected_action": encode_array(np.zeros(26, dtype=np.float32)),
                "idempotency_key": idempotency_keys[0],
            }
        ]


def test_readiness_prewarm_is_get_only_and_generation_pinned():
    client = _ReadyClient()
    clock = _Clock(100)

    report = prewarm_readiness_v4(
        client,
        probes=3,
        expected_gateway_pids=(7000, 7001),
        expected_gateway_instance_ids=("gateway-0-stable", "gateway-1-stable"),
        monotonic_ns=clock,
    )

    assert client.calls == 3
    assert report == {
        "enabled": True,
        "mode": "readiness_only",
        "probes": 3,
        "gateway_count": 2,
        "gateway_pids": [7000, 7001],
        "gateway_instance_ids": ["gateway-0-stable", "gateway-1-stable"],
        "ledger_mutations": 0,
        "elapsed_ns": 0,
    }


def test_readiness_prewarm_rejects_generation_change_and_bad_probe_count():
    with pytest.raises(ValueError, match=r"integer in \[0,16\]"):
        prewarm_readiness_v4(_ReadyClient(), probes=True)
    with pytest.raises(ContractError, match="identity changed"):
        prewarm_readiness_v4(
            _ReadyClient(change_generation_at=2),
            probes=2,
        )


def test_transport_timing_splits_http_timeline_from_client_residual():
    clock = _Clock(0)
    delegate = _Delegate(clock, cost_ns=5)
    observer = TransportTimingObserverV4(delegate, monotonic_ns=clock)
    marker = observer.mark()

    # First shard: request starts at t=10 and response is parsed at t=110.
    clock.value = 110
    observer.record(
        "http_batch_request_v4",
        duration_ns=100,
        request_body_bytes=1000,
        response_body_bytes=100,
        path="/v4/steps/exchange-batch",
    )
    # Second shard overlaps: request starts at t=40 and completes at t=160.
    clock.value = 160
    observer.record(
        "http_batch_request_v4",
        duration_ns=120,
        request_body_bytes=2000,
        response_body_bytes=200,
        path="/v4/steps/exchange-batch",
    )
    # A v3 anchor registration in the same bracket is accounted too.
    clock.value = 190
    observer.record(
        "http_batch_request_v3",
        duration_ns=20,
        request_body_bytes=300,
        response_body_bytes=30,
        path="/v3/anchors/register-batch",
    )
    summary = observer.summarize(
        marker,
        call_started_ns=0,
        call_finished_ns=220,
    )

    assert summary["instrumented"] is True
    assert summary["client_call_wall_ns"] == 220
    assert summary["http_request_count"] == 3
    assert summary["http_request_duration_sum_ns"] == 240
    assert summary["http_rtt_critical_path_ns"] == 120
    assert summary["http_timeline_span_ns"] == 180
    assert summary["client_non_http_upper_bound_ns"] == 40
    assert summary["first_http_start_lag_ns"] == 10
    assert summary["last_http_observed_tail_ns"] == 30
    assert summary["last_telemetry_return_tail_ns"] == 25
    assert summary["transport_telemetry_emit_sum_ns"] == 15
    assert summary["request_body_bytes"] == 3300
    assert summary["response_body_bytes"] == 330
    assert len(delegate.events) == 3


def test_transport_timing_suppresses_success_fsync_but_forwards_errors():
    clock = _Clock(0)
    delegate = _Delegate(clock, cost_ns=7)
    observer = TransportTimingObserverV4(
        delegate,
        delegate_success_events=False,
        monotonic_ns=clock,
    )
    marker = observer.mark()
    clock.value = 100
    observer.record(
        "http_batch_request_v4",
        duration_ns=40,
        request_body_bytes=10,
        response_body_bytes=2,
        path="/v4/steps/exchange-batch",
        outcome="success",
    )
    assert delegate.events == []
    clock.value = 200
    observer.record(
        "http_batch_request_v4",
        duration_ns=50,
        request_body_bytes=10,
        response_body_bytes=0,
        path="/v4/steps/exchange-batch",
        outcome="error",
        retryable=True,
    )
    assert len(delegate.events) == 1
    summary = observer.summarize(
        marker,
        call_started_ns=0,
        call_finished_ns=250,
    )
    assert summary["http_request_count"] == 2
    assert summary["transport_telemetry_emit_sum_ns"] == 7


def test_orchestrator_attaches_transport_breakdown_to_pending_step():
    observer = TransportTimingObserverV4()
    orchestrator = PipelinedBatchStepOrchestratorV4(
        _OneLaneOwner(),
        _TimedOneLaneH200(observer),
        num_envs=1,
        transport_timing=observer,
    )
    pending = orchestrator.bootstrap(
        [
            StepKey(
                run_id="v4-timing-test",
                round_id=1,
                actor_id="v4-timing-lane-000",
                rollout_window_id="window-000000",
                rollout_step_index=0,
                episode_id="episode-000000",
                episode_step_index=0,
            )
        ]
    )

    assert pending.transport_timing["instrumented"] is True
    assert pending.transport_timing["http_request_count"] == 1
    assert pending.transport_timing["request_body_bytes"] == 123
    assert pending.transport_timing["response_body_bytes"] == 45


def test_native_trainer_has_hash_bound_cost_aware_and_prewarm_wiring():
    project = Path(__file__).parents[1]
    source = (project / "code/src/dagger_rtx/native_service_trainer.py").read_text(
        encoding="utf-8"
    )

    assert 'h200.get("sharding_strategy", "modulo")' in source
    assert '"cost_aware_window_pinned"' in source
    assert "cost_aware_sharding=(" in source
    assert 'h200.get("readiness_prewarm_probes", 0)' in source
    assert "prewarm_readiness_v4(" in source
    assert "transport_timing=v4_transport_timing" in source
    assert "telemetry=v4_transport_timing" in source
    assert 'h200.get(\n                    "suppress_success_transport_events", False' in source
    assert "not v4_suppress_success_transport_events" in source

    fragment = json.loads(
        (project / "config/v4_exchange_optimization.v1.fragment.json").read_text(
            encoding="utf-8"
        )
    )
    assert fragment["h200"]["suppress_success_transport_events"] is True
    assert fragment["semantics"]["default_if_absent"][
        "suppress_success_transport_events"
    ] is False
