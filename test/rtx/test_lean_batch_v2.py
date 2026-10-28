from __future__ import annotations

from collections import Counter
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
from dagger_rtx.lean_boundary_v2 import (
    CONTRACT_VERSION_V2,
    H200_ANCHOR_BATCH_PATH_V2,
    H200_COMMIT_BATCH_PATH_V2,
    H200_PREPARE_BATCH_PATH_V2,
    project_lean_arrays,
)
from dagger_rtx.lean_client_v2 import LeanH200BatchClient
from dagger_rtx.ndarray_batch_v2 import unpack_ndarray_batch


def _key(lane: int, step: int = 0) -> StepKey:
    return StepKey(
        run_id="lean-v2-test",
        round_id=1,
        actor_id=f"lane-{lane:03d}",
        rollout_window_id=f"window-{lane:03d}",
        rollout_step_index=step,
        episode_id=f"episode-{lane:03d}",
        episode_step_index=step,
    )


def _make_owner(tmp_path: Path, tiny_profile):
    backend = DeterministicFakeBatchRaven(tiny_profile, num_envs=2)
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "owner",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
        persist_chunks=False,
    )
    keys = [_key(0), _key(1)]
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


class RecordingV2Transport:
    """Deterministic H200-compatible in-memory v2 boundary."""

    def __init__(self):
        self.calls = Counter()
        self.bodies: dict[str, list[bytes]] = {}
        self.anchors: dict[str, dict[str, np.ndarray]] = {}
        self.prepares: dict[str, tuple[str, dict]] = {}
        self.commits: dict[str, tuple[str, dict]] = {}
        self.anchor_effects = 0
        self.prepare_effects = 0
        self.commit_effects = 0
        self.fail_anchor_after_effect_once = False
        self.fail_commit_after_effect_once = False
        self.fail_commit_after_first_lane_once = False
        self.drop_last_prepare_lane = False

    def get_json(self, path, *, max_response_bytes):
        del max_response_bytes
        assert path == "/ready"
        return {
            "ok": True,
            "role": "render_student",
            "schema": CONTRACT_VERSION,
            "supported_contract_versions": [CONTRACT_VERSION, CONTRACT_VERSION_V2],
            "features": {
                "lean_binary_prepare_batch": True,
                "content_addressed_anchor_cache": True,
                "uncompressed_batch_persistence": True,
                "batched_commit": True,
            },
        }

    def post_binary(self, path, body, *, batch_id, max_response_bytes):
        del max_response_bytes
        self.calls[path] += 1
        self.bodies.setdefault(path, []).append(body)
        header, groups = unpack_ndarray_batch(body)
        assert header["batch_id"] == batch_id
        if path == H200_ANCHOR_BATCH_PATH_V2:
            rows = []
            for record, arrays in zip(header["anchors"], groups, strict=True):
                digest = record["anchor_sha256"]
                status = "cached" if digest in self.anchors else "registered"
                if digest not in self.anchors:
                    self.anchors[digest] = arrays
                    self.anchor_effects += 1
                rows.append(
                    {
                        "anchor_sha256": digest,
                        "status": status,
                        "bytes": sum(value.nbytes for value in arrays.values()) + 848,
                    }
                )
            response = {
                "contract_version": CONTRACT_VERSION_V2,
                "kind": "anchor_batch_response",
                "batch_id": batch_id,
                "anchors": rows,
            }
            if self.fail_anchor_after_effect_once:
                self.fail_anchor_after_effect_once = False
                raise TransientTransportError("lost anchor response after effect")
            return response
        if path != H200_PREPARE_BATCH_PATH_V2:
            raise AssertionError(path)
        rows = []
        for index, (request, arrays) in enumerate(
            zip(header["lanes"], groups, strict=True)
        ):
            assert payload_digest(request["metadata"], arrays) == request["wire_digest"]
            anchor_sha = request["metadata"]["episode_assignment"]["anchor_sha256"]
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
            draw = 0.25 if index == 0 else 0.95
            source = "teacher" if draw < request["beta"] else "student"
            selected = (
                arrays["teacher_action"]
                if source == "teacher"
                else np.concatenate((student, arrays["homie_action"])).astype(np.float32)
            )
            row = {
                "contract_version": CONTRACT_VERSION_V2,
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
        if self.drop_last_prepare_lane:
            rows = rows[:-1]
        return {
            "contract_version": CONTRACT_VERSION_V2,
            "kind": "prepare_batch_response",
            "batch_id": batch_id,
            "lanes": rows,
        }

    def post_json(self, path, payload, *, max_response_bytes):
        del max_response_bytes
        assert path == H200_COMMIT_BATCH_PATH_V2
        assert payload == json.loads(canonical_json(payload))
        self.calls[path] += 1
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
                "contract_version": CONTRACT_VERSION_V2,
                "idempotency_key": idem,
                "step_key": request["step_key"],
                "commit_receipt": request["commit_receipt"],
                "status": "committed",
            }
            row["remote_commit_receipt"] = sha256_bytes(canonical_json(row))
            self.commits[idem] = (digest, row)
            self.commit_effects += 1
            rows.append(row)
            if self.fail_commit_after_first_lane_once:
                self.fail_commit_after_first_lane_once = False
                raise TransientTransportError(
                    "injected partial commit batch after first lane"
                )
        response = {
            "contract_version": CONTRACT_VERSION_V2,
            "kind": "commit_batch_response",
            "batch_id": payload["batch_id"],
            "lanes": rows,
        }
        if self.fail_commit_after_effect_once:
            self.fail_commit_after_effect_once = False
            raise TransientTransportError("lost commit response after effect")
        return response


def _local_lean(owner, keys):
    return owner.prepare_batch_lean(
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


def test_lean_projection_omits_raw_depth_and_anchor(tiny_profile, tmp_path):
    owner, _backend, keys = _make_owner(tmp_path, tiny_profile)
    local = _local_lean(owner, keys)
    generated = next(
        item for item in local if item["metadata"]["observation_source"] == "generated_controls"
    )
    projected = project_lean_arrays(generated, shape_profile=tiny_profile)

    assert all(not name.startswith("depth_metric__") for name in projected)
    assert all(not name.startswith("depth_rawq__") for name in projected)
    assert all(not name.startswith("depth_raw__") for name in projected)
    assert all(not name.startswith("anchor_rgb__") for name in projected)
    assert all(not name.startswith("raytraced_rgb__") for name in projected)
    assert len(projected) == 13

    metadata = dict(generated["metadata"])
    metadata["contract_version"] = CONTRACT_VERSION_V2
    before = payload_digest(metadata, projected)
    generated["arrays"]["depth_metric__head_color"][0, 0] += 10
    assert payload_digest(
        metadata, project_lean_arrays(generated, shape_profile=tiny_profile)
    ) == before
    generated["arrays"]["depth_ctrl__head_color"][0, 0] ^= 1
    assert payload_digest(
        metadata, project_lean_arrays(generated, shape_profile=tiny_profile)
    ) != before

    receipt = next((tmp_path / "owner").rglob("prepare-000.json"))
    durable = json.loads(receipt.read_text())
    assert "arrays" not in durable
    assert "data_b64" not in receipt.read_text()
    assert "array_descriptors" in durable


def test_v2_one_prepare_one_commit_anchor_cache_and_one_vector_step(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    remote = RecordingV2Transport()
    client = LeanH200BatchClient(
        remote, tmp_path / "lean-client", shape_profile=tiny_profile
    )
    assert client.verify_ready()["role"] == "render_student"
    orchestrator = BatchStepOrchestrator(
        owner,
        client,
        tmp_path / "orchestrator",
        num_envs=2,
        max_parallel_requests=2,
    )
    result = orchestrator.run_batch(keys)

    assert result["phase"] == "COMPLETE"
    assert result["wire_contract_version"] == CONTRACT_VERSION_V2
    assert backend.batch_step_calls == 1
    assert backend.selected_batches[0].shape == (2, 26)
    assert remote.calls[H200_ANCHOR_BATCH_PATH_V2] == 1
    assert remote.calls[H200_PREPARE_BATCH_PATH_V2] == 1
    assert remote.calls[H200_COMMIT_BATCH_PATH_V2] == 1
    assert remote.prepare_effects == 2
    assert remote.commit_effects == 2
    assert len(client.cached_anchor_sha256s) == 2

    header, groups = unpack_ndarray_batch(
        remote.bodies[H200_PREPARE_BATCH_PATH_V2][0]
    )
    assert [lane["step_key"] for lane in header["lanes"]] == [
        key.as_dict() for key in keys
    ]
    for lane, arrays in zip(header["lanes"], groups, strict=True):
        assert lane["metadata"]["contract_version"] == CONTRACT_VERSION_V2
        assert lane["metadata"]["episode_assignment"]["anchor_sha256"] in client.cached_anchor_sha256s
        assert not any(name.startswith("anchor_rgb__") for name in arrays)
        assert not any(name.startswith(("depth_metric__", "depth_rawq__", "depth_raw__")) for name in arrays)


def test_invalid_prepare_batch_blocks_vector_step(tiny_profile, tmp_path):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    remote = RecordingV2Transport()
    remote.drop_last_prepare_lane = True
    client = LeanH200BatchClient(
        remote, tmp_path / "lean-client", shape_profile=tiny_profile
    )
    orchestrator = BatchStepOrchestrator(
        owner, client, tmp_path / "orchestrator", num_envs=2
    )

    with pytest.raises(ContractError, match="lane count"):
        orchestrator.run_batch(keys)
    assert backend.batch_step_calls == 0
    assert remote.calls[H200_PREPARE_BATCH_PATH_V2] == 1
    assert remote.calls[H200_COMMIT_BATCH_PATH_V2] == 0


def test_lost_anchor_and_commit_responses_retry_without_double_step(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    remote = RecordingV2Transport()
    remote.fail_anchor_after_effect_once = True
    remote.fail_commit_after_effect_once = True
    client = LeanH200BatchClient(
        remote, tmp_path / "lean-client", shape_profile=tiny_profile
    )
    orchestrator = BatchStepOrchestrator(
        owner, client, tmp_path / "orchestrator", num_envs=2
    )

    with pytest.raises(TransientTransportError, match="anchor"):
        orchestrator.run_batch(keys)
    assert backend.batch_step_calls == 0
    assert not client.cached_anchor_sha256s

    with pytest.raises(TransientTransportError, match="commit"):
        orchestrator.run_batch(keys)
    assert backend.batch_step_calls == 1
    assert remote.commit_effects == 2

    result = orchestrator.run_batch(keys)
    assert result["phase"] == "COMPLETE"
    assert backend.batch_step_calls == 1
    assert remote.anchor_effects == 2
    assert remote.calls[H200_ANCHOR_BATCH_PATH_V2] == 2
    assert remote.calls[H200_PREPARE_BATCH_PATH_V2] == 1
    assert remote.calls[H200_COMMIT_BATCH_PATH_V2] == 2


def test_partial_remote_commit_batch_retries_without_second_vector_step(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    remote = RecordingV2Transport()
    remote.fail_commit_after_first_lane_once = True
    client = LeanH200BatchClient(
        remote, tmp_path / "lean-client", shape_profile=tiny_profile
    )
    orchestrator = BatchStepOrchestrator(
        owner, client, tmp_path / "orchestrator", num_envs=2
    )

    with pytest.raises(TransientTransportError, match="partial commit"):
        orchestrator.run_batch(keys)
    assert backend.batch_step_calls == 1
    assert remote.commit_effects == 1

    result = orchestrator.run_batch(keys)
    assert result["phase"] == "COMPLETE"
    assert backend.batch_step_calls == 1
    assert remote.commit_effects == 2
    assert remote.calls[H200_COMMIT_BATCH_PATH_V2] == 2


def test_stale_local_anchor_cache_reregisters_once_on_remote_412(
    tiny_profile, tmp_path
):
    owner, backend, keys = _make_owner(tmp_path, tiny_profile)
    local = _local_lean(owner, keys)
    remote = RecordingV2Transport()
    client = LeanH200BatchClient(
        remote, tmp_path / "lean-client", shape_profile=tiny_profile
    )
    idempotency_keys = [f"h200-v2-prepare-{lane}" for lane in range(2)]

    first = client.prepare_batch(local, idempotency_keys)
    assert len(first) == 2
    assert remote.calls[H200_ANCHOR_BATCH_PATH_V2] == 1
    remote.anchors.clear()

    retry = client.prepare_batch(local, idempotency_keys)
    assert len(retry) == 2
    assert remote.calls[H200_ANCHOR_BATCH_PATH_V2] == 2
    assert remote.calls[H200_PREPARE_BATCH_PATH_V2] == 3
    assert backend.batch_step_calls == 0
