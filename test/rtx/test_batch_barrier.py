from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

import dagger_rtx.batch_owner as batch_owner_module
from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.boundary import H200_COMMIT_PATH, H200_PREPARE_PATH
from dagger_rtx.chunks import COMPACT_AUDIT_ROW_FIELDS
from dagger_rtx.contracts import CONTRACT_VERSION, ROUND_BETAS, StepKey
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.fake_h200 import FakeH200Server
from dagger_rtx.http_transport import BoundedBearerJsonTransport, TransientTransportError
from dagger_rtx.owner import CrashUncertain
from dagger_rtx.window_runner import BatchWindowRunner, TransportRetryPolicy


def _token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text("test-token\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _key(
    lane: int,
    rollout_step: int,
    episode_id: str,
    episode_step: int,
) -> StepKey:
    return StepKey(
        run_id="batch-test-run",
        round_id=1,
        actor_id=f"actor-lane-{lane}",
        rollout_window_id=f"window-lane-{lane}",
        rollout_step_index=rollout_step,
        episode_id=episode_id,
        episode_step_index=episode_step,
    )


def _begin(owner: IsaacBatchTransitionOwner, keys: list[StepKey]) -> None:
    owner.begin_windows(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"begin-lane-{lane}",
                "step_key": key.as_dict(),
            }
            for lane, key in enumerate(keys)
        ]
    )


def _prepare_direct(owner: IsaacBatchTransitionOwner, keys: list[StepKey]):
    return owner.prepare_batch(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"prepare-{key.actor_id}-{key.rollout_step_index}",
                "step_key": key.as_dict(),
                "beta": ROUND_BETAS[key.round_id],
            }
            for key in keys
        ]
    )


def _teacher_commit_requests(local):
    return [
        {
            "contract_version": CONTRACT_VERSION,
            "step_key": prepared["step_key"],
            "prepare_receipt": prepared["prepare_receipt"],
            "selection_id": f"selection-lane-{lane}",
            "selection_receipt": f"{lane + 1:064x}",
            "selected_source": "teacher",
            "beta": prepared["metadata"]["beta"],
            "selected_action": prepared["arrays"]["teacher_action"],
        }
        for lane, prepared in enumerate(local)
    ]


def _make_owner(tmp_path: Path, tiny_profile, **backend_kwargs):
    backend = DeterministicFakeBatchRaven(
        tiny_profile,
        num_envs=2,
        **backend_kwargs,
    )
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    keys = [_key(0, 0, "episode-l0-a", 0), _key(1, 0, "episode-l1-a", 0)]
    _begin(owner, keys)
    return owner, backend, keys


class _FailSecondPrepareOnce:
    def __init__(self, delegate):
        self.delegate = delegate
        self.prepare_calls = 0
        self.failed = False

    def post(self, path, payload, *, max_response_bytes):
        if path == H200_PREPARE_PATH:
            self.prepare_calls += 1
            if self.prepare_calls == 2 and not self.failed:
                self.failed = True
                raise RuntimeError("injected second-lane prepare failure")
        return self.delegate.post(path, payload, max_response_bytes=max_response_bytes)


class _FailAfterNthRemoteEffectOnce:
    """Lose one response after the remote side has durably applied it."""

    def __init__(self, delegate, target_path, nth, on_failure):
        self.delegate = delegate
        self.target_path = target_path
        self.nth = nth
        self.on_failure = on_failure
        self.calls = []
        self.failed = False

    def post(self, path, payload, *, max_response_bytes):
        response = self.delegate.post(
            path, payload, max_response_bytes=max_response_bytes
        )
        if path == self.target_path:
            self.calls.append(payload["idempotency_key"])
            if len(self.calls) == self.nth and not self.failed:
                self.failed = True
                self.on_failure()
                raise TransientTransportError("injected lost response after remote effect")
        return response


def _make_window_runner(tmp_path, tiny_profile, transport):
    backend = DeterministicFakeBatchRaven(tiny_profile, num_envs=2)
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runner-owner",
        tmp_path / "runner-staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    orchestrator = BatchStepOrchestrator(
        owner, transport, tmp_path / "runner-orchestrator", num_envs=2
    )
    delays = []
    runner = BatchWindowRunner(
        owner,
        orchestrator,
        run_id="bounded-retry-run",
        round_id=1,
        actor_prefix="bounded-retry",
        retry_policy=TransportRetryPolicy(
            max_attempts=2, initial_backoff_s=0, max_backoff_s=0
        ),
        sleep=delays.append,
    )
    runner.begin_window(0)
    return runner, backend, delays


class _ThreadSafeInMemoryH200:
    """Concurrent deterministic boundary used for the production-sized B=28 gate."""

    def __init__(self, shape_profile):
        from dagger_rtx.fake_h200 import FakeH200State

        self.state = FakeH200State("unused", shape_profile=shape_profile)
        self._state_lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def post(self, path, payload, *, max_response_bytes):
        del max_response_bytes
        lane = int(payload["step_key"]["actor_id"].rsplit("-", 1)[-1])
        with self._activity_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            # Make overlap deterministic without making the 81-step gate slow.
            time.sleep((lane % 8 + 1) * 0.0001)
            with self._state_lock:
                if path == H200_PREPARE_PATH:
                    return self.state.prepare(payload)
                if path == H200_COMMIT_PATH:
                    return self.state.commit(payload)
                raise AssertionError(path)
        finally:
            with self._activity_lock:
                self.active -= 1


def test_runner_retries_partial_h200_prepare_in_process_without_any_partial_step(
    tmp_path, tiny_profile
):
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        token = _token_file(tmp_path)
        holder = {}

        def at_failure():
            # Both scalar H200 lane effects occurred, but the vectorized barrier
            # must still be entirely before the irreversible transition.
            assert holder["backend"].batch_step_calls == 0

        transport = _FailAfterNthRemoteEffectOnce(
            BoundedBearerJsonTransport(remote.base_url, token),
            H200_PREPARE_PATH,
            2,
            at_failure,
        )
        runner, backend, delays = _make_window_runner(
            tmp_path, tiny_profile, transport
        )
        holder["backend"] = backend
        result = runner.run_step(0, 0)

        assert result["phase"] == "COMPLETE"
        assert backend.batch_step_calls == 1
        assert remote.state.prepare_effects == 2
        assert remote.state.commit_effects == 2
        assert transport.calls[:2] == transport.calls[2:]
        assert delays == [0]


def test_runner_retries_partial_h200_commit_ack_in_process_without_double_step(
    tmp_path, tiny_profile
):
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        token = _token_file(tmp_path)
        holder = {}

        def at_failure():
            # The local vector step already happened, and response loss must
            # never cause it to be entered a second time.
            assert holder["backend"].batch_step_calls == 1

        transport = _FailAfterNthRemoteEffectOnce(
            BoundedBearerJsonTransport(remote.base_url, token),
            H200_COMMIT_PATH,
            2,
            at_failure,
        )
        runner, backend, delays = _make_window_runner(
            tmp_path, tiny_profile, transport
        )
        holder["backend"] = backend
        result = runner.run_step(0, 0)

        assert result["phase"] == "COMPLETE"
        assert backend.batch_step_calls == 1
        assert remote.state.prepare_effects == 2
        assert remote.state.commit_effects == 2
        assert transport.calls[:2] == transport.calls[2:]
        assert delays == [0]


def test_batch_barrier_never_partially_steps_when_one_h200_prepare_fails(
    tmp_path, tiny_profile
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        transport = _FailSecondPrepareOnce(
            BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path))
        )
        orchestrator = BatchStepOrchestrator(
            owner, transport, tmp_path / "orchestrator", num_envs=2
        )
        with pytest.raises(RuntimeError, match="second-lane"):
            orchestrator.run_batch(keys)
        assert backend.batch_step_calls == 0
        assert owner.status()["phase"] == "PREPARED"

        result = orchestrator.run_batch(keys)
        assert result["phase"] == "COMPLETE"
        assert backend.batch_step_calls == 1
        assert backend.selected_batches[0].shape == (2, 26)
        # Lane zero's successful prepare was replayed with the same H200 key;
        # the H200 fake records one semantic effect per lane.
        assert remote.state.prepare_effects == 2


def test_batch_commit_failure_is_crash_uncertain_and_never_resteps(
    tmp_path, tiny_profile
):
    owner, backend, keys = _make_owner(
        tmp_path, tiny_profile, raise_at_batch_calls={0}
    )
    local = _prepare_direct(owner, keys)
    # Teacher commits are valid scalar wire messages, gathered by the batch owner.
    requests = _teacher_commit_requests(local)
    with pytest.raises(CrashUncertain):
        owner.commit_batch(requests)
    assert backend.batch_step_calls == 1
    state = owner.status()
    assert state["phase"] == "CRASH_UNCERTAIN"
    assert state["batch_commit_intent"]["selected_batch_sha256"]

    with pytest.raises(CrashUncertain):
        owner.commit_batch(requests)
    assert backend.batch_step_calls == 1
    restarted = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    with pytest.raises(CrashUncertain):
        restarted.commit_batch(requests)
    assert backend.batch_step_calls == 1


def test_batch_committing_intent_is_durable_before_the_one_vector_step(
    tmp_path, tiny_profile
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    local = _prepare_direct(owner, keys)
    original_step = backend.step_batch
    observed = []

    def attest_intent(selected_actions, teacher_actions):
        durable = json.loads(
            (tmp_path / "runtime" / "batch_owner_state.json").read_text(encoding="utf-8")
        )
        observed.append(durable)
        assert durable["phase"] == "BATCH_COMMITTING"
        assert len(durable["batch_commit_intent"]["step_keys"]) == 2
        assert len(durable["batch_commit_intent"]["commit_digests"]) == 2
        assert backend.batch_step_calls == 0
        return original_step(selected_actions, teacher_actions)

    backend.step_batch = attest_intent
    responses = owner.commit_batch(_teacher_commit_requests(local))
    assert len(responses) == 2
    assert observed
    assert backend.batch_step_calls == 1
    # Exact retry is served from durable receipts, never through env.step.
    assert owner.commit_batch(_teacher_commit_requests(local)) == responses
    assert backend.batch_step_calls == 1


def test_commit_intent_journal_failure_is_fail_closed_before_vector_step(
    tmp_path, tiny_profile, monkeypatch
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    local = _prepare_direct(owner, keys)
    real_append = owner.journal.append

    def fail_commit_intent(entry):
        if entry.get("event") == "BATCH_COMMIT_INTENT":
            raise OSError("injected commit-intent journal failure")
        return real_append(entry)

    monkeypatch.setattr(owner.journal, "append", fail_commit_intent)
    with pytest.raises(CrashUncertain, match="post-batch-intent operation"):
        owner.commit_batch(_teacher_commit_requests(local))

    assert backend.batch_step_calls == 0
    durable = json.loads(
        (tmp_path / "runtime" / "batch_owner_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert durable["phase"] == "CRASH_UNCERTAIN"
    assert durable["batch_commit_intent"]["selected_batch_sha256"]
    with pytest.raises(CrashUncertain):
        owner.commit_batch(_teacher_commit_requests(local))
    assert backend.batch_step_calls == 0


def test_mid_capture_failure_is_fail_closed_before_any_env_step(
    tmp_path, tiny_profile
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    original_capture = backend.capture_batch
    observed_intents = []

    def fail_after_first_lane(request_keys):
        rows = list(original_capture(request_keys))
        durable = json.loads(
            (tmp_path / "runtime" / "batch_owner_state.json").read_text(
                encoding="utf-8"
            )
        )
        observed_intents.append(durable["batch_prepare_intent"])
        assert durable["phase"] == "BATCH_PREPARING"
        assert len(durable["batch_prepare_intent"]["step_keys"]) == 2
        yield rows[0]
        raise RuntimeError("injected failure after first captured lane")

    backend.capture_batch = fail_after_first_lane
    with pytest.raises(CrashUncertain, match="prepare retry is forbidden"):
        _prepare_direct(owner, keys)
    assert observed_intents
    assert backend.capture_batch_calls == 1
    assert backend.batch_step_calls == 0
    assert owner.status()["phase"] == "CRASH_UNCERTAIN"

    with pytest.raises(CrashUncertain, match="prepare retry is forbidden"):
        _prepare_direct(owner, keys)
    assert backend.capture_batch_calls == 1
    assert backend.batch_step_calls == 0

    restarted = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    with pytest.raises(CrashUncertain, match="prepare retry is forbidden"):
        _prepare_direct(restarted, keys)
    assert backend.capture_batch_calls == 1
    assert backend.batch_step_calls == 0


def test_partial_prepare_receipt_write_is_fail_closed_and_not_retryable(
    tmp_path, tiny_profile, monkeypatch
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    real_atomic_write_json = batch_owner_module.atomic_write_json
    injected = []

    def fail_second_lane_receipt(path, value, mode=0o600):
        path = Path(path)
        if "receipts" in path.parts and "actor-lane-1" in path.parts:
            durable = json.loads(
                (tmp_path / "runtime" / "batch_owner_state.json").read_text(
                    encoding="utf-8"
                )
            )
            assert durable["phase"] == "BATCH_PREPARING"
            injected.append(path)
            raise OSError("injected second-lane receipt write failure")
        return real_atomic_write_json(path, value, mode)

    monkeypatch.setattr(batch_owner_module, "atomic_write_json", fail_second_lane_receipt)
    with pytest.raises(CrashUncertain, match="prepare retry is forbidden"):
        _prepare_direct(owner, keys)
    assert len(injected) == 1
    assert backend.capture_batch_calls == 1
    assert backend.batch_step_calls == 0
    state = owner.status()
    assert state["phase"] == "CRASH_UNCERTAIN"
    assert state["batch_prepare_intent"]["step_keys"] == [
        key.as_dict() for key in keys
    ]
    first_receipt = (
        tmp_path
        / "runtime"
        / "receipts"
        / "batch-test-run"
        / "D1"
        / "actor-lane-0"
        / "window-lane-0"
        / "prepare-000.json"
    )
    assert first_receipt.is_file()

    with pytest.raises(CrashUncertain, match="prepare retry is forbidden"):
        _prepare_direct(owner, keys)
    assert backend.capture_batch_calls == 1
    assert backend.batch_step_calls == 0

    restarted = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    with pytest.raises(CrashUncertain, match="prepare retry is forbidden"):
        _prepare_direct(restarted, keys)
    assert backend.capture_batch_calls == 1
    assert backend.batch_step_calls == 0


def test_restart_from_durable_batch_preparing_is_crash_uncertain(
    tmp_path, tiny_profile
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    owner.state["phase"] = "BATCH_PREPARING"
    owner.state["batch_prepare_intent"] = {
        "step_keys": [key.as_dict() for key in keys],
        "request_digests": ["a" * 64, "b" * 64],
        "request_ids": ["prepare-lane-0", "prepare-lane-1"],
    }
    owner._persist()

    restarted = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    state = restarted.status()
    assert state["phase"] == "CRASH_UNCERTAIN"
    assert "active BATCH_PREPARING" in state["uncertain_reason"]
    with pytest.raises(CrashUncertain, match="active BATCH_PREPARING"):
        _prepare_direct(restarted, keys)
    assert backend.capture_batch_calls == 0
    assert backend.batch_step_calls == 0


def test_b2_vectorized_window_done_reset_and_two_independent_seals(
    tmp_path, tiny_profile
):
    # Lane 0 terminates after rollout index 2; lane 1 continues uninterrupted.
    owner, backend, initial = _make_owner(
        tmp_path, tiny_profile, done_at_calls={0: {2}}
    )
    token = _token_file(tmp_path)
    episode_ids = ["episode-l0-a", "episode-l1-a"]
    episode_steps = [0, 0]
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        orchestrator = BatchStepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, token),
            tmp_path / "orchestrator",
            num_envs=2,
        )
        final = None
        for rollout_step in range(81):
            if rollout_step == 3:
                episode_ids[0] = "episode-l0-b"
                episode_steps[0] = 0
            keys = [
                _key(lane, rollout_step, episode_ids[lane], episode_steps[lane])
                for lane in range(2)
            ]
            final = orchestrator.run_batch(keys)
            assert final["phase"] == "COMPLETE"
            episode_steps = [value + 1 for value in episode_steps]

        assert final is not None
        assert backend.batch_step_calls == 81
        assert len(backend.selected_batches) == 81
        assert all(batch.shape == (2, 26) for batch in backend.selected_batches)
        assert backend.bind_lane_calls == [2, 1]
        assert remote.state.prepare_effects == 162
        assert remote.state.commit_effects == 162

    owner_state = owner.status()
    assert owner_state["phase"] == "SEALED"
    lane0, lane1 = owner_state["lanes"]
    assert len(lane0["episode_bindings"]) == 2
    assert len(lane1["episode_bindings"]) == 1
    for lane in range(2):
        root = (
            tmp_path
            / "staging"
            / "batch-test-run"
            / "D1"
            / f"actor-lane-{lane}"
            / f"window-lane-{lane}"
        )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["row_count"] == 81
        assert manifest["artifact_role"] == "rtx_identity_outcome_audit"
        assert not manifest["learner_complete"]
        with np.load(root / "sealed.npz", allow_pickle=False) as data:
            expected_fields = set(COMPACT_AUDIT_ROW_FIELDS) | {
                "prompt",
                "round_id",
                "rollout_window_id",
            }
            assert set(data.files) == expected_fields
            assert data["behavior_actions"].shape == (81, 26)
            assert all(str(value) for value in data["selection_receipts"])
            assert np.array_equal(data["rollout_step_indices"], np.arange(81))
            if lane == 0:
                assert bool(data["dones"][2])
                assert str(data["episode_ids"][3]) == "episode-l0-b"
                assert int(data["episode_step_indices"][3]) == 0
            else:
                assert not bool(data["dones"].any())
                assert len(set(data["episode_ids"].tolist())) == 1


def test_remote_commit_retry_does_not_repeat_vectorized_step(tmp_path, tiny_profile):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        remote.state.fail_next_commit = 1
        orchestrator = BatchStepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path)),
            tmp_path / "orchestrator",
            num_envs=2,
        )
        with pytest.raises(RuntimeError, match="HTTP 503"):
            orchestrator.run_batch(keys)
        assert backend.batch_step_calls == 1
        result = orchestrator.run_batch(keys)
        assert result["phase"] == "COMPLETE"
        assert backend.batch_step_calls == 1
        assert remote.state.commit_effects == 2


def test_b28_full_window_eight_way_fanout_seals_every_actor_and_preserves_resets(
    tmp_path, tiny_profile
):
    num_envs = 28
    backend = DeterministicFakeBatchRaven(
        tiny_profile,
        num_envs=num_envs,
        done_at_calls={0: {2}, 13: {40}, 27: {79}},
    )
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "b28-owner",
        tmp_path / "b28-staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    remote = _ThreadSafeInMemoryH200(tiny_profile)
    orchestrator = BatchStepOrchestrator(
        owner,
        remote,
        tmp_path / "b28-orchestrator",
        num_envs=num_envs,
        max_parallel_requests=8,
    )
    runner = BatchWindowRunner(
        owner,
        orchestrator,
        run_id="b28-full-window-test",
        round_id=1,
        actor_prefix="b28-full-window-g0",
        retry_policy=TransportRetryPolicy(
            max_attempts=2, initial_backoff_s=0, max_backoff_s=0
        ),
        sleep=lambda _: None,
    )

    result = runner.run_window(0)

    assert result["transition_batches"] == 81
    assert result["physical_transition_rows"] == 28 * 81
    assert backend.batch_step_calls == 81
    assert all(batch.shape == (28, 26) for batch in backend.selected_batches)
    assert remote.max_active == 8
    assert remote.state.prepare_effects == 28 * 81
    assert remote.state.commit_effects == 28 * 81
    final = result["final"]
    assert len(final["local_commits"]) == 28
    assert all(commit["sealed"] is True for commit in final["local_commits"])
    assert len(final["remote_acknowledgements"]) == 28
    assert len(
        {
            ack["remote_commit_receipt"]
            for ack in final["remote_acknowledgements"]
        }
    ) == 28
    assert backend.bind_lane_calls[0] == 2
    assert backend.bind_lane_calls[13] == 2
    assert backend.bind_lane_calls[27] == 2
    assert all(
        calls == 1
        for lane, calls in enumerate(backend.bind_lane_calls)
        if lane not in {0, 13, 27}
    )

    state = owner.status()
    assert state["phase"] == "SEALED"
    assert len(state["lanes"]) == 28
    for lane in range(28):
        actor = f"b28-full-window-g0-lane-{lane:03d}"
        root = (
            tmp_path
            / "b28-staging"
            / "b28-full-window-test"
            / "D1"
            / actor
            / "window-000000"
        )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["actor_id"] == actor
        assert manifest["row_count"] == 81
        assert manifest["artifact_role"] == "rtx_identity_outcome_audit"
        assert manifest["learner_complete"] is False
        assert (root / "sealed.npz").is_file()
