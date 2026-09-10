from __future__ import annotations

import pytest

from dagger_rtx.contracts import ContractError
from dagger_rtx.http_transport import TransientTransportError
from dagger_rtx.owner import CrashUncertain, OwnerConflict
from dagger_rtx.window_runner import (
    BatchWindowRunner,
    RolloutLifecycle,
    TransportRetryPolicy,
)


class _Owner:
    num_envs = 2

    def __init__(self):
        self.begin_requests = None

    def begin_windows(self, requests):
        self.begin_requests = list(requests)
        return [{"ok": True} for _ in requests]


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
