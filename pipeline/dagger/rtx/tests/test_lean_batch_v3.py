from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
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
from dagger_rtx.http_transport import RemoteHttpError, TransientTransportError
from dagger_rtx.lean_boundary_v3 import (
    CONTRACT_VERSION_V3,
    H200_ANCHOR_BATCH_PATH_V3,
    H200_COMMIT_BATCH_PATH_V3,
    H200_PREPARE_BATCH_PATH_V3,
)
from dagger_rtx.lean_client_v3 import (
    LeanH200BatchClientV3,
    ShardedLeanH200BatchClientV3,
)
from dagger_rtx.ndarray_batch_v3 import unpack_ndarray_batch_v3


def _key(lane: int, step: int = 0) -> StepKey:
    return StepKey(
        run_id="lean-v3-test",
        round_id=1,
        actor_id=f"lane-{lane:03d}",
        rollout_window_id=f"window-{lane:03d}",
        rollout_step_index=step,
        episode_id=f"episode-{lane:03d}",
        episode_step_index=step,
    )


def _make_owner(tmp_path: Path, tiny_profile, *, num_envs: int = 2):
    backend = DeterministicFakeBatchRaven(tiny_profile, num_envs=num_envs)
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "owner",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
        persist_chunks=False,
    )
    keys = [_key(lane) for lane in range(num_envs)]
    owner.begin_windows(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"begin-{lane}",
                "step_key": key.as_dict(),
            }
            for lane, key in enumerate(keys)
        ]
    )
    return owner, backend, keys


class RecordingV3Transport:
    """Small H200-compatible v3 endpoint with route-level idempotency."""

    def __init__(self, gateway_index: int):
        self.gateway_index = gateway_index
        self.calls = Counter()
        self.bodies: dict[str, list[bytes]] = {}
        self.anchors: dict[str, dict[str, np.ndarray]] = {}
        self.prepares: dict[str, tuple[str, dict]] = {}
        self.commits: dict[str, tuple[str, dict]] = {}
        self.prepare_actor_groups: list[list[str]] = []
        self.commit_actor_groups: list[list[str]] = []
        self.prepare_effects = 0
        self.commit_effects = 0
        self.fail_prepare_after_effect_once = False
        self.fail_commit_after_effect_once = False
        self.closed = False

    def get_json(self, path, *, max_response_bytes):
        del max_response_bytes
        assert path == "/ready"
        return {
            "ok": True,
            "pid": 1000 + self.gateway_index,
            "role": "render_student",
            "schema": CONTRACT_VERSION,
            "supported_contract_versions": [CONTRACT_VERSION, CONTRACT_VERSION_V3],
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

    def post_binary(self, path, body, *, batch_id, max_response_bytes):
        del max_response_bytes
        self.calls[path] += 1
        self.bodies.setdefault(path, []).append(body)
        header, groups = unpack_ndarray_batch_v3(body)
        assert header["batch_id"] == batch_id
        assert header["blob_codec"] == "zlib-1"
        if path == H200_ANCHOR_BATCH_PATH_V3:
            rows = []
            for record, arrays in zip(header["anchors"], groups, strict=True):
                digest = record["anchor_sha256"]
                status = "cached" if digest in self.anchors else "registered"
                self.anchors.setdefault(digest, arrays)
                rows.append(
                    {
                        "anchor_sha256": digest,
                        "status": status,
                        "bytes": sum(value.nbytes for value in arrays.values()) + 1,
                    }
                )
            return {
                "contract_version": CONTRACT_VERSION_V3,
                "kind": "anchor_batch_response",
                "batch_id": batch_id,
                "anchors": rows,
            }
        assert path == H200_PREPARE_BATCH_PATH_V3
        self.prepare_actor_groups.append(
            [row["step_key"]["actor_id"] for row in header["lanes"]]
        )
        rows = []
        for index, (request, arrays) in enumerate(
            zip(header["lanes"], groups, strict=True)
        ):
            assert payload_digest(request["metadata"], arrays) == request["wire_digest"]
            assert "homie_action" not in arrays
            anchor_sha = request["metadata"]["episode_assignment"]["anchor_sha256"]
            if request["metadata"]["observation_source"] == "generated_controls":
                if anchor_sha not in self.anchors:
                    raise RemoteHttpError(412, b"missing_anchor")
            idem = request["idempotency_key"]
            request_digest = sha256_bytes(body + index.to_bytes(2, "little"))
            prior = self.prepares.get(idem)
            if prior is not None:
                if prior[0] != request_digest:
                    raise ContractError("prepare idempotency conflict")
                rows.append(prior[1])
                continue
            student = np.linspace(-1, 1, 11, dtype=np.float32)
            actor_lane = int(request["step_key"]["actor_id"].rsplit("-", 1)[-1])
            draw = 0.25 if actor_lane % 2 == 0 else 0.95
            source = "teacher" if draw < request["beta"] else "student"
            homie = arrays["teacher_action"][11:]
            selected = (
                arrays["teacher_action"]
                if source == "teacher"
                else np.concatenate((student, homie)).astype(np.float32)
            )
            row = {
                "contract_version": CONTRACT_VERSION_V3,
                "idempotency_key": idem,
                "step_key": request["step_key"],
                "prepare_receipt": request["prepare_receipt"],
                "capture_digest": request["capture_digest"],
                "wire_digest": request["wire_digest"],
                "beta": request["beta"],
                "selection_id": request_digest,
                "selection_draw": draw,
                "selected_source": source,
                "student_action_11": encode_array(student),
                "selected_action": encode_array(selected),
                "observation_digest": sha256_bytes(
                    canonical_json(
                        {
                            "step_key": request["step_key"],
                            "source": request["metadata"]["observation_source"],
                            "wire_digest": request["wire_digest"],
                        }
                    )
                ),
            }
            row["selection_receipt"] = sha256_bytes(canonical_json(row))
            self.prepares[idem] = (request_digest, row)
            self.prepare_effects += 1
            rows.append(row)
        response = {
            "contract_version": CONTRACT_VERSION_V3,
            "kind": "prepare_batch_response",
            "batch_id": batch_id,
            "lanes": rows,
        }
        if self.fail_prepare_after_effect_once:
            self.fail_prepare_after_effect_once = False
            raise TransientTransportError("lost v3 shard prepare response")
        return response

    def post_json(self, path, payload, *, max_response_bytes):
        del max_response_bytes
        assert path == H200_COMMIT_BATCH_PATH_V3
        assert payload == json.loads(canonical_json(payload))
        self.calls[path] += 1
        self.commit_actor_groups.append(
            [row["step_key"]["actor_id"] for row in payload["lanes"]]
        )
        rows = []
        for request in payload["lanes"]:
            idem = request["idempotency_key"]
            digest = sha256_bytes(canonical_json(request))
            prior = self.commits.get(idem)
            if prior is not None:
                if prior[0] != digest:
                    raise ContractError("commit idempotency conflict")
                rows.append(prior[1])
                continue
            row = {
                "contract_version": CONTRACT_VERSION_V3,
                "idempotency_key": idem,
                "step_key": request["step_key"],
                "commit_receipt": request["commit_receipt"],
                "status": "committed",
            }
            row["remote_commit_receipt"] = sha256_bytes(canonical_json(row))
            self.commits[idem] = (digest, row)
            self.commit_effects += 1
            rows.append(row)
        response = {
            "contract_version": CONTRACT_VERSION_V3,
            "kind": "commit_batch_response",
            "batch_id": payload["batch_id"],
            "lanes": rows,
        }
        if self.fail_commit_after_effect_once:
            self.fail_commit_after_effect_once = False
            raise TransientTransportError("lost v3 shard commit response")
        return response

    def close(self):
        self.closed = True


def _sharded_client(tmp_path, tiny_profile, count=2):
    remotes = [RecordingV3Transport(index) for index in range(count)]
    gateways = [
        LeanH200BatchClientV3(
            remote,
            tmp_path / f"gateway-{index}",
            shape_profile=tiny_profile,
        )
        for index, remote in enumerate(remotes)
    ]
    return ShardedLeanH200BatchClientV3(gateways), remotes


def test_v3_two_gateway_end_to_end_keeps_all_lane_barrier_and_one_env_step(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    client, remotes = _sharded_client(tmp_path / "client", tiny_profile)
    assert client.verify_ready()["gateway_count"] == 2
    orchestrator = BatchStepOrchestrator(
        owner, client, tmp_path / "orchestrator", num_envs=2
    )
    result = orchestrator.run_batch(keys)

    assert result["phase"] == "COMPLETE"
    assert result["wire_contract_version"] == CONTRACT_VERSION_V3
    assert backend.batch_step_calls == 1
    assert backend.selected_batches[0].shape == (2, 26)
    assert [remote.prepare_actor_groups[0] for remote in remotes] == [
        ["lane-000"],
        ["lane-001"],
    ]
    assert sum(remote.prepare_effects for remote in remotes) == 2
    assert sum(remote.commit_effects for remote in remotes) == 2
    assert len(result["capture_digests"]) == 2
    for remote in remotes:
        header, _groups = unpack_ndarray_batch_v3(
            remote.bodies[H200_PREPARE_BATCH_PATH_V3][0]
        )
        assert header["lanes"][0]["capture_digest"] in result["capture_digests"]
    client.close()
    assert all(remote.closed for remote in remotes)


def test_v3_b28_modulo_sharding_and_original_order_are_deterministic(
    tiny_profile, tmp_path
):
    owner, _backend, keys = _make_owner(tmp_path, tiny_profile, num_envs=28)
    local = owner.prepare_batch_lean(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"prepare-{lane}",
                "step_key": key.as_dict(),
                "beta": ROUND_BETAS[1],
            }
            for lane, key in enumerate(keys)
        ]
    )
    client, remotes = _sharded_client(tmp_path / "client", tiny_profile, count=8)
    selections = client.prepare_batch(
        local, [f"h200-v3-prepare-{lane}" for lane in range(28)]
    )

    assert [row["step_key"]["actor_id"] for row in selections] == [
        f"lane-{lane:03d}" for lane in range(28)
    ]
    assert [remote.prepare_actor_groups[0] for remote in remotes] == [
        [f"lane-{lane:03d}" for lane in range(shard, 28, 8)]
        for shard in range(8)
    ]
    client.close()


def test_v3_failed_prepare_shard_blocks_step_and_retry_is_idempotent(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    client, remotes = _sharded_client(tmp_path / "client", tiny_profile)
    remotes[1].fail_prepare_after_effect_once = True
    orchestrator = BatchStepOrchestrator(
        owner, client, tmp_path / "orchestrator", num_envs=2
    )

    with pytest.raises(TransientTransportError, match="prepare response"):
        orchestrator.run_batch(keys)
    assert backend.batch_step_calls == 0
    assert sum(remote.prepare_effects for remote in remotes) == 2

    result = orchestrator.run_batch(keys)
    assert result["phase"] == "COMPLETE"
    assert backend.batch_step_calls == 1
    assert sum(remote.prepare_effects for remote in remotes) == 2
    client.close()


def test_v3_failed_commit_shard_retries_without_second_env_step(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    client, remotes = _sharded_client(tmp_path / "client", tiny_profile)
    remotes[0].fail_commit_after_effect_once = True
    orchestrator = BatchStepOrchestrator(
        owner, client, tmp_path / "orchestrator", num_envs=2
    )

    with pytest.raises(TransientTransportError, match="commit response"):
        orchestrator.run_batch(keys)
    assert backend.batch_step_calls == 1
    assert sum(remote.commit_effects for remote in remotes) == 2

    result = orchestrator.run_batch(keys)
    assert result["phase"] == "COMPLETE"
    assert backend.batch_step_calls == 1
    assert sum(remote.commit_effects for remote in remotes) == 2
    client.close()


def test_v3_changed_lane_body_reuses_key_as_conflict(tiny_profile, tmp_path):
    owner, _backend, keys = _make_owner(tmp_path, tiny_profile)
    local = owner.prepare_batch_lean(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"prepare-{lane}",
                "step_key": key.as_dict(),
                "beta": ROUND_BETAS[1],
            }
            for lane, key in enumerate(keys)
        ]
    )
    client, _remotes = _sharded_client(tmp_path / "client", tiny_profile)
    keys_idem = ["v3-idem-0", "v3-idem-1"]
    assert len(client.prepare_batch(local, keys_idem)) == 2

    changed = copy.deepcopy(local)
    changed[0]["arrays"]["actor_obs"][0] += 1
    changed[0]["capture_digest"] = payload_digest(
        changed[0]["metadata"], changed[0]["arrays"]
    )
    with pytest.raises(ContractError, match="idempotency conflict"):
        client.prepare_batch(changed, keys_idem)
    client.close()
