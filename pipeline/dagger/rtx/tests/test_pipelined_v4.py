from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    ROUND_BETAS,
    ContractError,
    StepKey,
    canonical_json,
    encode_array,
    payload_digest,
    sha256_bytes,
)
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.http_transport import TransientTransportError
from dagger_rtx.lean_boundary_v4 import (
    CONTRACT_VERSION_V4,
    H200_EXCHANGE_BATCH_PATH_V4,
    H200_FINALIZE_BATCH_PATH_V4,
    H200_PREPARE_BATCH_PATH_V4,
    lightweight_capture_digest_v4,
)
from dagger_rtx.lean_client_v4 import (
    LeanH200BatchClientV4,
    ShardedLeanH200BatchClientV4,
)
from dagger_rtx.ndarray_batch_v4 import unpack_ndarray_batch_v4
from dagger_rtx.pipelined_window_v4 import (
    AggregateHotPathJournalV4,
    PipelinedBatchStepOrchestratorV4,
    PipelinedBatchWindowRunnerV4,
)
from dagger_rtx.phase_telemetry import PhaseBoundaryTelemetry
from dagger_rtx.window_runner import TransportRetryPolicy


class _AnchorRegistrar:
    def __init__(self):
        self.calls = 0

    def register_missing_anchors(self, local_prepares):
        self.calls += 1
        return []


class RecordingV4Transport:
    def __init__(self, gateway: int):
        self.gateway = gateway
        self.instance = f"gateway-{gateway}-generation-1"
        self.calls = Counter()
        self.bodies: dict[str, list[bytes]] = {}
        self.prepare_effects: dict[str, tuple[str, dict]] = {}
        self.commit_effects: dict[str, tuple[str, dict]] = {}
        self.prepare_actor_groups: list[list[str]] = []
        self.exchange_commit_groups: list[list[str]] = []
        self.exchange_prepare_groups: list[list[str]] = []
        self.fail_exchange_after_one_prepare_once = False
        self.fail_prepare_after_effect_once = False
        self.closed = False

    def get_json(self, path, *, max_response_bytes):
        del max_response_bytes
        assert path == "/ready"
        self.calls[path] += 1
        return {
            "ok": True,
            "pid": 7000 + self.gateway,
            "role": "render_student",
            "schema": CONTRACT_VERSION,
            "supported_contract_versions": [CONTRACT_VERSION, CONTRACT_VERSION_V4],
            "features": {
                "ephemeral_window_cache": True,
                "pipelined_commit_prepare": True,
                "minimal_compressed_prepare_batch": True,
                "source_specific_observation_payload": True,
                "learner_only_persistence": True,
            },
            "gateway_instance_id": self.instance,
            "fake_backend": False,
        }

    def _selection(self, request, arrays):
        idem = request["idempotency_key"]
        digest = payload_digest(request["metadata"], arrays)
        prior = self.prepare_effects.get(idem)
        if prior is not None:
            if prior[0] != digest:
                raise ContractError("prepare idempotency conflict")
            return prior[1]
        assert digest == request["wire_digest"]
        student = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        lane = int(request["step_key"]["actor_id"].rsplit("-", 1)[-1])
        draw = 0.25 if lane % 2 == 0 else 0.95
        source = "teacher" if draw < request["beta"] else "student"
        teacher = arrays["teacher_action"]
        selected = (
            teacher
            if source == "teacher"
            else np.concatenate((student, teacher[11:])).astype(np.float32)
        )
        row = {
            "contract_version": CONTRACT_VERSION_V4,
            "idempotency_key": idem,
            "step_key": request["step_key"],
            "prepare_receipt": request["prepare_receipt"],
            "capture_digest": request["capture_digest"],
            "wire_digest": request["wire_digest"],
            "beta": request["beta"],
            "selection_id": sha256_bytes(canonical_json({"idem": idem, "digest": digest})),
            "selection_draw": draw,
            "selected_source": source,
            "student_action_11": encode_array(student),
            "selected_action": encode_array(selected),
            "observation_digest": sha256_bytes(
                canonical_json({"wire_digest": request["wire_digest"]})
            ),
        }
        row["selection_receipt"] = sha256_bytes(canonical_json(row))
        self.prepare_effects[idem] = (digest, row)
        return row

    def _commit(self, request):
        idem = request["idempotency_key"]
        digest = sha256_bytes(canonical_json(request))
        prior = self.commit_effects.get(idem)
        if prior is not None:
            if prior[0] != digest:
                raise ContractError("commit idempotency conflict")
            return prior[1]
        row = {
            "contract_version": CONTRACT_VERSION_V4,
            "idempotency_key": idem,
            "step_key": request["step_key"],
            "commit_receipt": request["commit_receipt"],
            "status": "committed",
        }
        row["remote_commit_receipt"] = sha256_bytes(canonical_json(row))
        self.commit_effects[idem] = (digest, row)
        return row

    def post_binary(self, path, body, *, batch_id, max_response_bytes):
        del max_response_bytes
        self.calls[path] += 1
        self.bodies.setdefault(path, []).append(body)
        header, groups = unpack_ndarray_batch_v4(body)
        assert header["batch_id"] == batch_id
        if header["gateway_instance_id"] != self.instance:
            raise ContractError("gateway generation binding mismatch")
        if path == H200_PREPARE_BATCH_PATH_V4:
            self.prepare_actor_groups.append(
                [row["step_key"]["actor_id"] for row in header["lanes"]]
            )
            rows = [
                self._selection(request, arrays)
                for request, arrays in zip(header["lanes"], groups, strict=True)
            ]
            response = {
                "contract_version": CONTRACT_VERSION_V4,
                "kind": "prepare_batch_response",
                "batch_id": batch_id,
                "gateway_instance_id": self.instance,
                "lanes": rows,
            }
            if self.fail_prepare_after_effect_once:
                self.fail_prepare_after_effect_once = False
                raise TransientTransportError("lost step-0 response")
            return response
        assert path == H200_EXCHANGE_BATCH_PATH_V4
        self.exchange_commit_groups.append(
            [row["step_key"]["actor_id"] for row in header["previous_commits"]]
        )
        self.exchange_prepare_groups.append(
            [row["step_key"]["actor_id"] for row in header["lanes"]]
        )
        acks = [self._commit(row) for row in header["previous_commits"]]
        rows = []
        for index, (request, arrays) in enumerate(
            zip(header["lanes"], groups, strict=True)
        ):
            rows.append(self._selection(request, arrays))
            if self.fail_exchange_after_one_prepare_once and index == 0:
                self.fail_exchange_after_one_prepare_once = False
                raise TransientTransportError("partial exchange response lost")
        return {
            "contract_version": CONTRACT_VERSION_V4,
            "kind": "exchange_batch_response",
            "batch_id": batch_id,
            "gateway_instance_id": self.instance,
            "previous_commit_acks": acks,
            "lanes": rows,
        }

    def post_json(self, path, payload, *, max_response_bytes):
        del max_response_bytes
        assert path == H200_FINALIZE_BATCH_PATH_V4
        assert payload == json.loads(canonical_json(payload))
        assert payload["gateway_instance_id"] == self.instance
        self.calls[path] += 1
        rows = [self._commit(row) for row in payload["lanes"]]
        return {
            "contract_version": CONTRACT_VERSION_V4,
            "kind": "finalize_batch_response",
            "batch_id": payload["batch_id"],
            "gateway_instance_id": self.instance,
            "lanes": rows,
        }

    def close(self):
        self.closed = True


def _key(lane, step=0, *, episode=0, episode_step=None):
    return StepKey(
        run_id="v4-test-run",
        round_id=1,
        actor_id=f"v4-test-lane-{lane:03d}",
        rollout_window_id="window-000000",
        rollout_step_index=step,
        episode_id=f"episode-{lane:03d}-{episode:04d}",
        episode_step_index=step if episode_step is None else episode_step,
    )


def _owner(tmp_path: Path, tiny_profile, *, count=2, persist_chunks=False, **kwargs):
    backend = DeterministicFakeBatchRaven(
        tiny_profile, num_envs=count, **kwargs
    )
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "owner",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
        persist_chunks=persist_chunks,
    )
    return owner, backend


def _client(tmp_path, tiny_profile, count=2):
    transports = [RecordingV4Transport(index) for index in range(count)]
    clients = [
        LeanH200BatchClientV4(
            transport,
            anchor_registrar=_AnchorRegistrar(),
            shape_profile=tiny_profile,
        )
        for transport in transports
    ]
    sharded = ShardedLeanH200BatchClientV4(clients)
    assert sharded.verify_ready()["gateway_count"] == count
    return sharded, transports


def _runner(tmp_path, tiny_profile, *, count=2, persist_chunks=False, **kwargs):
    phase_telemetry = kwargs.pop("phase_telemetry", None)
    owner, backend = _owner(
        tmp_path, tiny_profile, count=count, persist_chunks=persist_chunks, **kwargs
    )
    client, transports = _client(tmp_path / "client", tiny_profile, min(count, 8))
    journal = AggregateHotPathJournalV4(tmp_path / "hot-path.jsonl")
    orchestrator = PipelinedBatchStepOrchestratorV4(
        owner,
        client,
        num_envs=count,
        journal=journal,
        phase_telemetry=phase_telemetry,
    )
    runner = PipelinedBatchWindowRunnerV4(
        owner,
        orchestrator,
        run_id="v4-test-run",
        round_id=1,
        actor_prefix="v4-test",
        retry_policy=TransportRetryPolicy(
            max_attempts=2, initial_backoff_s=0, max_backoff_s=0
        ),
        phase_telemetry=phase_telemetry,
    )
    return runner, orchestrator, backend, client, transports


def test_v4_step0_exact_keys_and_v3_lean_projection(tiny_profile, tmp_path):
    runner, orchestrator, backend, client, transports = _runner(
        tmp_path, tiny_profile
    )
    runner.begin_window(0)
    pending = orchestrator.bootstrap(runner.keys(0, 0))
    assert backend.batch_step_calls == 1
    header, groups = unpack_ndarray_batch_v4(
        transports[0].bodies[H200_PREPARE_BATCH_PATH_V4][0]
    )
    assert set(header) == {
        "contract_version",
        "kind",
        "batch_id",
        "gateway_instance_id",
        "lanes",
        "blob_codec",
        "raw_nbytes",
        "raw_sha256",
    }
    assert set(header["lanes"][0]) == {
        "contract_version",
        "idempotency_key",
        "step_key",
        "prepare_receipt",
        "capture_digest",
        "wire_digest",
        "beta",
        "metadata",
        "arrays",
    }
    assert set(groups[0]) in (
        {
            "actor_obs",
            "teacher_action",
            "depth_ctrl__head_color",
            "rgb_robot__head_color",
            "depth_ctrl__hand_left",
            "rgb_robot__hand_left",
            "depth_ctrl__hand_right",
            "rgb_robot__hand_right",
        },
        {
            "actor_obs",
            "teacher_action",
            "raytraced_rgb__head_color",
            "raytraced_rgb__hand_left",
            "raytraced_rgb__hand_right",
        },
    )
    forbidden = {"depth_metric__head_color", "depth_raw__head_color", "homie_action"}
    assert not (forbidden & set(groups[0]))
    assert pending.commit_idempotency_keys[0].startswith("h200-v4-commit-")
    local = pending.local_prepares[0]
    assert local["capture_digest"] == lightweight_capture_digest_v4(
        local["step_key"], local["wire_digest"]
    )
    client.close()


def test_v4_owner_digest_hashes_only_algorithmically_consumed_arrays(
    tiny_profile, tmp_path, monkeypatch
):
    import dagger_rtx.batch_owner as batch_owner_module

    owner, _backend = _owner(tmp_path, tiny_profile, count=1)
    key = _key(0)
    owner.begin_windows(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "begin-v4-digest",
                "step_key": key.as_dict(),
            }
        ]
    )
    observed = []
    real_payload_digest = batch_owner_module.payload_digest

    def traced_payload_digest(metadata, arrays):
        observed.append(tuple(sorted(arrays)))
        return real_payload_digest(metadata, arrays)

    monkeypatch.setattr(batch_owner_module, "payload_digest", traced_payload_digest)
    local = owner.prepare_batch_lean_v4(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "prepare-v4-digest",
                "step_key": key.as_dict(),
                "beta": ROUND_BETAS[1],
            }
        ]
    )[0]
    expected_generated = (
        "actor_obs",
        "depth_ctrl__hand_left",
        "depth_ctrl__hand_right",
        "depth_ctrl__head_color",
        "rgb_robot__hand_left",
        "rgb_robot__hand_right",
        "rgb_robot__head_color",
        "teacher_action",
    )
    expected_raytraced = (
        "actor_obs",
        "raytraced_rgb__hand_left",
        "raytraced_rgb__hand_right",
        "raytraced_rgb__head_color",
        "teacher_action",
    )
    assert observed in ([expected_generated], [expected_raytraced])
    forbidden = {"teacher_obs", "critic_obs", "homie_obs", "gt_obj_pos"}
    assert not forbidden.intersection(observed[0])
    assert not any(name.startswith(("depth_metric__", "depth_raw")) for name in observed[0])
    assert local["capture_digest"] == lightweight_capture_digest_v4(
        key, local["wire_digest"]
    )
    request_digest = sha256_bytes(
        canonical_json(
            {
                "step_key": key.as_dict(),
                "request_id": "prepare-v4-digest",
                "beta": ROUND_BETAS[1],
            }
        )
    )
    assert local["prepare_receipt"] == sha256_bytes(
        canonical_json(
            {
                "request_digest": request_digest,
                "capture_digest": local["capture_digest"],
            }
        )
    )


def test_v4_partial_exchange_retry_never_double_steps_and_preserves_reset(
    tiny_profile, tmp_path
):
    runner, orchestrator, backend, client, transports = _runner(
        tmp_path, tiny_profile, done_at_calls={0: {0}}
    )
    runner.begin_window(0)
    prior = orchestrator.bootstrap(runner.keys(0, 0))
    runner._advance({"local_commits": list(prior.local_commits)})
    keys1 = runner.keys(0, 1)
    assert keys1[0].episode_id != prior.keys[0].episode_id
    assert keys1[0].episode_step_index == 0
    assert keys1[1].episode_id == prior.keys[1].episode_id
    assert keys1[1].episode_step_index == 1
    transports[0].fail_exchange_after_one_prepare_once = True
    with pytest.raises(TransientTransportError, match="partial exchange"):
        orchestrator.exchange(prior, keys1)
    assert backend.batch_step_calls == 1
    assert sum(len(item.commit_effects) for item in transports) == 2
    # Two step-0 prepares plus one current prepare on each settled shard.
    assert sum(len(item.prepare_effects) for item in transports) == 4
    current, acks = orchestrator.exchange(prior, keys1)
    assert len(acks) == 2
    assert backend.batch_step_calls == 2
    assert sum(len(item.commit_effects) for item in transports) == 2
    assert sum(len(item.prepare_effects) for item in transports) == 4
    assert transports[0].bodies[H200_EXCHANGE_BATCH_PATH_V4][0] == transports[0].bodies[
        H200_EXCHANGE_BATCH_PATH_V4
    ][1]
    assert backend.bind_lane_calls == [2, 1]
    assert current.keys[0] == keys1[0]
    client.close()


def test_v4_b28_modulo_sharding_preserves_both_halves_and_vector_order(
    tiny_profile, tmp_path
):
    runner, orchestrator, backend, client, transports = _runner(
        tmp_path, tiny_profile, count=28
    )
    runner.begin_window(0)
    prior = orchestrator.bootstrap(runner.keys(0, 0))
    runner._advance({"local_commits": list(prior.local_commits)})
    current, acks = orchestrator.exchange(prior, runner.keys(0, 1))
    assert backend.batch_step_calls == 2
    assert [item["step_key"]["actor_id"] for item in acks] == [
        f"v4-test-lane-{lane:03d}" for lane in range(28)
    ]
    assert [item["step_key"]["actor_id"] for item in current.selections] == [
        f"v4-test-lane-{lane:03d}" for lane in range(28)
    ]
    expected = [
        [f"v4-test-lane-{lane:03d}" for lane in range(shard, 28, 8)]
        for shard in range(8)
    ]
    assert [item.exchange_commit_groups[0] for item in transports] == expected
    assert [item.exchange_prepare_groups[0] for item in transports] == expected
    client.close()


def test_v4_gateway_generation_change_is_fatal_before_retry_step(
    tiny_profile, tmp_path
):
    runner, orchestrator, backend, client, transports = _runner(
        tmp_path, tiny_profile
    )
    runner.begin_window(0)
    prior = orchestrator.bootstrap(runner.keys(0, 0))
    runner._advance({"local_commits": list(prior.local_commits)})
    transports[0].fail_exchange_after_one_prepare_once = True
    transports[0].instance = "gateway-0-generation-2"
    with pytest.raises(ContractError, match="generation binding mismatch"):
        orchestrator.exchange(prior, runner.keys(0, 1))
    assert backend.batch_step_calls == 1
    client.close()


def test_v4_full_window_has_81_steps_80_exchanges_final_flush_and_81_rows(
    tiny_profile, tmp_path
):
    phase_telemetry = PhaseBoundaryTelemetry(
        tmp_path / "phase.jsonl",
        tmp_path / "progress.json",
        run_id="v4-test-run",
        round_id=1,
        total_vector_steps=81,
        batch_size=1,
    )
    runner, _orchestrator, backend, client, transports = _runner(
        tmp_path,
        tiny_profile,
        count=1,
        persist_chunks=True,
        phase_telemetry=phase_telemetry,
    )
    result = runner.run_window(0)
    assert result["transition_batches"] == 81
    assert backend.batch_step_calls == 81
    remote = transports[0]
    assert remote.calls[H200_PREPARE_BATCH_PATH_V4] == 1
    assert remote.calls[H200_EXCHANGE_BATCH_PATH_V4] == 80
    assert remote.calls[H200_FINALIZE_BATCH_PATH_V4] == 1
    assert len(remote.prepare_effects) == 81
    assert len(remote.commit_effects) == 81
    assert len(result["final"]["remote_acknowledgements"]) == 1
    rows = [json.loads(line) for line in (tmp_path / "hot-path.jsonl").read_text().splitlines()]
    vector_rows = [row for row in rows if row["event"] == "vector_step"]
    assert len(vector_rows) == 81
    assert [row["rollout_step_index"] for row in vector_rows] == list(range(81))
    assert all(row["env_step_calls"] == 1 for row in vector_rows)
    phase_telemetry.close()
    phase_rows = [
        json.loads(line) for line in (tmp_path / "phase.jsonl").read_text().splitlines()
    ]
    ended = [row for row in phase_rows if row["event"] == "end"]
    assert sum(row["phase"] == "capture_teacher" for row in ended) == 81
    assert sum(row["phase"] == "prepare_or_exchange" for row in ended) == 81
    assert sum(row["phase"] == "env_step" for row in ended) == 81
    assert sum(row["phase"] == "finalize" for row in ended) == 1
    assert sum(row["phase"] == "vector_step_barrier" for row in ended) == 81
    assert ended[-1]["counts"]["completion_fraction"] == 1.0
    client.close()


def test_atomic_cursor_validation_never_partially_advances(tiny_profile, tmp_path):
    runner, _orchestrator, _backend, client, _transports = _runner(
        tmp_path, tiny_profile
    )
    before = [(item.episode_serial, item.episode_step_index) for item in runner.cursors]
    malformed = {
        "local_commits": [
            {"result": {"done": False, "timeout": False}},
            {"result": {"done": "not-bool", "timeout": False}},
        ]
    }
    with pytest.raises(ContractError, match="must be bool"):
        runner._advance(malformed)
    assert [(item.episode_serial, item.episode_step_index) for item in runner.cursors] == before
    client.close()


def test_v4_changed_prepare_under_same_idempotency_key_conflicts(
    tiny_profile, tmp_path
):
    owner, _backend = _owner(tmp_path, tiny_profile, count=1)
    key = _key(0)
    owner.begin_windows(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "begin-0",
                "step_key": key.as_dict(),
            }
        ]
    )
    local = owner.prepare_batch_lean_v4(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "prepare-0",
                "step_key": key.as_dict(),
                "beta": ROUND_BETAS[1],
            }
        ]
    )
    transport = RecordingV4Transport(0)
    client = LeanH200BatchClientV4(
        transport, anchor_registrar=_AnchorRegistrar(), shape_profile=tiny_profile
    )
    client.verify_ready()
    assert len(client.prepare_batch(local, ["stable-v4-idempotency-key"])) == 1
    changed = copy.deepcopy(local)
    changed[0]["arrays"]["actor_obs"][0] += 1
    with pytest.raises(ContractError, match="idempotency conflict"):
        client.prepare_batch(changed, ["stable-v4-idempotency-key"])
    client.close()
