from __future__ import annotations

import pytest

from dagger_rtx.contracts import CONTRACT_VERSION, ContractError
from dagger_rtx.http_transport import TransientTransportError
from dagger_rtx.lean_boundary_v3 import CONTRACT_VERSION_V3
from dagger_rtx.lean_client_v3 import LeanH200BatchClientV3
from dagger_rtx.owner import CrashUncertain, OwnerConflict
from dagger_rtx.window_runner import (
    BatchWindowRunner,
    RolloutLifecycle,
    TransportRetryPolicy,
    verify_ready_with_transient_retry,
)


class _Owner:
    num_envs = 2

    def __init__(self):
        self.begin_requests = None

    def begin_windows(self, requests):
        self.begin_requests = list(requests)
        return [{"ok": True} for _ in requests]


def test_readiness_retry_is_bounded_get_only_and_retries_only_transient_failures():
    class Ready:
        def __init__(self):
            self.calls = 0

        def verify_ready(self):
            self.calls += 1
            if self.calls < 3:
                raise TransientTransportError("injected connection reset")
            return {"ok": True, "pid": 7000}

    ready = Ready()
    sleeps = []
    assert verify_ready_with_transient_retry(
        ready,
        retry_policy=TransportRetryPolicy(
            max_attempts=3,
            initial_backoff_s=0.25,
            backoff_multiplier=2,
            max_backoff_s=2,
        ),
        sleep=sleeps.append,
    ) == {"ok": True, "pid": 7000}
    assert ready.calls == 3
    assert sleeps == [0.25, 0.5]

    class Invalid:
        calls = 0

        def verify_ready(self):
            self.calls += 1
            raise ContractError("gateway identity changed")

    invalid = Invalid()
    with pytest.raises(ContractError, match="identity changed"):
        verify_ready_with_transient_retry(invalid, sleep=lambda _delay: None)
    assert invalid.calls == 1


def test_readiness_retry_preserves_final_transient_error_when_exhausted():
    class Offline:
        def __init__(self):
            self.calls = 0

        def verify_ready(self):
            self.calls += 1
            raise TransientTransportError(f"reset-{self.calls}")

    offline = Offline()
    with pytest.raises(TransientTransportError, match="reset-3"):
        verify_ready_with_transient_retry(
            offline,
            retry_policy=TransportRetryPolicy(
                max_attempts=3,
                initial_backoff_s=0,
                backoff_multiplier=1,
                max_backoff_s=0,
            ),
            sleep=lambda _delay: None,
        )
    assert offline.calls == 3


def test_readiness_retry_uses_only_real_v3_get_ready_route(tmp_path):
    class RecordingTransport:
        def __init__(self):
            self.get_calls = []
            self.production_calls = []

        def get_json(self, path, *, max_response_bytes):
            self.get_calls.append((path, max_response_bytes))
            if len(self.get_calls) < 3:
                raise TransientTransportError("injected readiness reset")
            return {
                "ok": True,
                "pid": 7000,
                "role": "render_student",
                "schema": CONTRACT_VERSION,
                "supported_contract_versions": [
                    CONTRACT_VERSION,
                    CONTRACT_VERSION_V3,
                ],
                "features": {
                    "minimal_compressed_prepare_batch": True,
                    "content_addressed_anchor_cache": True,
                    "source_specific_observation_payload": True,
                    "learner_only_persistence": True,
                    "shared_filesystem_chunk_publish": True,
                    "batched_commit": True,
                },
                "fake_backend": False,
            }

        def post_binary(self, *args, **kwargs):
            self.production_calls.append(("post_binary", args, kwargs))
            raise AssertionError("readiness retry must not prepare or register anchors")

        def post_json(self, *args, **kwargs):
            self.production_calls.append(("post_json", args, kwargs))
            raise AssertionError("readiness retry must not commit")

    transport = RecordingTransport()
    client = LeanH200BatchClientV3(transport, tmp_path / "v3-client")
    sleeps = []
    result = verify_ready_with_transient_retry(client, sleep=sleeps.append)

    assert result["pid"] == 7000
    assert transport.get_calls == [("/ready", 1 << 20)] * 3
    assert transport.production_calls == []
    assert sleeps == [0.25, 0.5]


class _Orchestrator:
    num_envs = 2

    def __init__(self):
        self.keys = []

    def run_batch(self, keys):
        self.keys.append(list(keys))
        index = keys[0].rollout_step_index
        return {
            "phase": "COMPLETE",
            "local_commits": [
                {
                    "result": {"done": lane == 0 and index == 1, "timeout": False},
                    "sealed": index == 80,
                }
                for lane in range(2)
            ],
            "remote_acknowledgements": [{}, {}] if index == 80 else [{}, {}],
        }


def test_window_runner_uses_lane_identities_and_resets_only_done_lane():
    owner = _Owner()
    orchestrator = _Orchestrator()
    lifecycle_calls = []
    runner = BatchWindowRunner(
        owner,
        orchestrator,
        run_id="runner-test",
        round_id=1,
        actor_prefix="rtx-gpu0",
        lifecycle=RolloutLifecycle(
            begin=lambda: lifecycle_calls.append("begin"),
            finish=lambda: lifecycle_calls.append("finish"),
        ),
    )
    result = runner.run_window(7)
    assert result["transition_batches"] == 81
    assert result["physical_transition_rows"] == 162
    assert lifecycle_calls == ["begin", "finish"]
    assert len(orchestrator.keys) == 81
    before_done = orchestrator.keys[1]
    after_done = orchestrator.keys[2]
    assert before_done[0].episode_id == "episode-000-00000000"
    assert before_done[0].episode_step_index == 1
    assert after_done[0].episode_id == "episode-000-00000001"
    assert after_done[0].episode_step_index == 0
    assert after_done[1].episode_id == "episode-001-00000000"
    assert after_done[1].episode_step_index == 2
    assert owner.begin_requests[0]["step_key"]["actor_id"] == "rtx-gpu0-lane-000"
    assert owner.begin_requests[1]["step_key"]["actor_id"] == "rtx-gpu0-lane-001"


def test_window_runner_clears_lifecycle_on_orchestrator_failure():
    owner = _Owner()

    class Broken(_Orchestrator):
        def run_batch(self, keys):
            raise RuntimeError("injected")

    calls = []
    runner = BatchWindowRunner(
        owner,
        Broken(),
        run_id="runner-fail",
        round_id=1,
        actor_prefix="rtx",
        lifecycle=RolloutLifecycle(
            begin=lambda: calls.append("begin"), finish=lambda: calls.append("finish")
        ),
    )
    try:
        runner.run_window(0)
    except RuntimeError as exc:
        assert str(exc) == "injected"
    else:
        raise AssertionError("injected error was not raised")
    assert calls == ["begin", "finish"]


def test_window_runner_clears_lifecycle_when_begin_partially_fails():
    owner = _Owner()
    calls = []

    def partial_begin():
        calls.append("begin-partial")
        raise RuntimeError("partial native rollout initialization")

    runner = BatchWindowRunner(
        owner,
        _Orchestrator(),
        run_id="runner-begin-fail",
        round_id=1,
        actor_prefix="rtx",
        lifecycle=RolloutLifecycle(
            begin=partial_begin, finish=lambda: calls.append("finish")
        ),
    )
    with pytest.raises(RuntimeError, match="partial native"):
        runner.run_window(0)
    assert calls == ["begin-partial", "finish"]
    assert owner.begin_requests is None


class _AlwaysFails:
    num_envs = 2

    def __init__(self, failure):
        self.failure = failure
        self.calls = 0

    def run_batch(self, keys):
        self.calls += 1
        raise self.failure


def test_guard_recovery_does_not_capture_or_advance_until_transport_is_ready():
    orchestrator = _Orchestrator()
    health_calls, delays = [], []

    def recovering():
        health_calls.append(1)
        if len(health_calls) <= 4:
            raise TransientTransportError("owned tunnel is reconnecting")

    runner = BatchWindowRunner(_Owner(), orchestrator, run_id="supervised-recovery", round_id=1,
        actor_prefix="rtx", request_guard=recovering,
        retry_policy=TransportRetryPolicy(max_attempts=60, initial_backoff_s=1,
                                         max_backoff_s=2, max_recovery_s=90), sleep=delays.append)
    runner.run_step(0, 0)
    assert len(health_calls) == 5 and delays == [1, 2, 2, 2]
    assert len(orchestrator.keys) == 1
    assert [c.episode_step_index for c in runner.cursors] == [1, 1]


def test_guard_recovery_deadline_stops_before_any_simulator_work():
    orchestrator = _Orchestrator()
    clock = [0.0]

    def unavailable():
        raise TransientTransportError("forward unavailable")

    runner = BatchWindowRunner(_Owner(), orchestrator, run_id="supervised-timeout", round_id=1,
        actor_prefix="rtx", request_guard=unavailable,
        retry_policy=TransportRetryPolicy(max_attempts=60, initial_backoff_s=1,
                                         max_backoff_s=2, max_recovery_s=5),
        sleep=lambda delay: clock.__setitem__(0, clock[0] + delay), monotonic=lambda: clock[0])
    with pytest.raises(TransientTransportError):
        runner.run_step(0, 0)
    assert clock[0] == 5 and not orchestrator.keys
    assert [c.episode_step_index for c in runner.cursors] == [0, 0]


@pytest.mark.parametrize("outage_seconds,completed", [(116.0, True), (301.0, False)])
def test_production_outage_hold_preserves_exactly_once_cursor(tmp_path, outage_seconds, completed):
    from dagger_rtx.connection_health import qualification_connection_options
    policy = qualification_connection_options({
        "DAGGER_CONNECTION_STATUS": str(tmp_path / "status.json"),
        "DAGGER_CONNECTION_CONFIG_SHA256": "a" * 64,
    }, run_id="production-outage", gateway_pids=(11,), production=True)["retry_policy"]
    clock = [0.0]
    orchestrator = _Orchestrator()

    def recovering():
        if clock[0] < outage_seconds:
            assert not orchestrator.keys  # No capture, prepare or physical step while gated.
            raise TransientTransportError("pinned route reconnecting")

    runner = BatchWindowRunner(_Owner(), orchestrator, run_id="production-outage", round_id=1,
        actor_prefix="rtx", request_guard=recovering, retry_policy=policy,
        sleep=lambda delay: clock.__setitem__(0, clock[0] + delay), monotonic=lambda: clock[0])
    if completed:
        runner.run_step(0, 0)
        assert 116 <= clock[0] < 118
        assert len(orchestrator.keys) == 1
        assert [c.episode_step_index for c in runner.cursors] == [1, 1]
    else:
        with pytest.raises(TransientTransportError):
            runner.run_step(0, 0)
        assert clock[0] == 300
        assert not orchestrator.keys
        assert [c.episode_step_index for c in runner.cursors] == [0, 0]


def test_window_runner_bounds_transient_retries():
    orchestrator = _AlwaysFails(TransientTransportError("network unavailable"))
    delays = []
    runner = BatchWindowRunner(
        _Owner(),
        orchestrator,
        run_id="runner-retry-bound",
        round_id=1,
        actor_prefix="rtx",
        retry_policy=TransportRetryPolicy(
            max_attempts=3,
            initial_backoff_s=0.125,
            backoff_multiplier=2,
            max_backoff_s=1,
        ),
        sleep=delays.append,
    )
    with pytest.raises(TransientTransportError, match="network unavailable"):
        runner.run_step(0, 0)
    assert orchestrator.calls == 3
    assert delays == [0.125, 0.25]


@pytest.mark.parametrize(
    "failure",
    [
        ContractError("bad wire contract"),
        OwnerConflict("owner state conflict"),
        CrashUncertain("native step uncertain"),
    ],
)
def test_window_runner_never_retries_semantic_or_uncertain_failures(failure):
    orchestrator = _AlwaysFails(failure)
    runner = BatchWindowRunner(
        _Owner(),
        orchestrator,
        run_id="runner-no-semantic-retry",
        round_id=1,
        actor_prefix="rtx",
        retry_policy=TransportRetryPolicy(
            max_attempts=9, initial_backoff_s=0, max_backoff_s=0
        ),
    )
    with pytest.raises(type(failure)):
        runner.run_step(0, 0)
    assert orchestrator.calls == 1


def test_production_episode_namespace_preserves_history_across_two_windows():
    owner, orchestrator = _Owner(), _Orchestrator()
    runner = BatchWindowRunner(owner, orchestrator, run_id='production-namespace-test', round_id=1,
        actor_prefix='production-g0', episode_namespace='prod-r1-e02-c003')
    runner.run_window(148, manage_lifecycle=False)
    runner.run_window(149, manage_lifecycle=False)
    assert len(orchestrator.keys) == 162
    first, following = orchestrator.keys[0], orchestrator.keys[81]
    assert first[1].episode_id == following[1].episode_id == 'prod-r1-e02-c003-episode-001-00000000'
    assert following[1].episode_step_index == 81
    assert following[1].rollout_window_id == 'window-000149'
    assert following[0].episode_id == 'prod-r1-e02-c003-episode-000-00000001'
    assert following[0].episode_step_index == 79


def test_distinct_cohort_namespaces_cannot_reuse_episode_ids():
    def runner(namespace):
        return BatchWindowRunner(_Owner(), _Orchestrator(), run_id='production-namespace-test', round_id=1,
            actor_prefix='production-g0', episode_namespace=namespace)
    before = runner('prod-r1-e01-c000').keys(0, 0)
    after = runner('prod-r1-e01-c001').keys(2, 0)
    assert {row.episode_id for row in before}.isdisjoint(row.episode_id for row in after)
    assert [row.actor_id for row in before] == [row.actor_id for row in after]
    with pytest.raises(ContractError): runner('../invalid-namespace')
