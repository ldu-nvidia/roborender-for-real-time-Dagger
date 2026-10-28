from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from dagger_rtx.boundary import build_h200_commit, build_h200_prepare
from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    RavenShapeProfile,
    StepKey,
    canonical_json,
    decode_arrays,
    payload_digest,
    sha256_bytes,
)
from dagger_rtx.fake_h200 import FakeH200Server, observation_input_digest
from dagger_rtx.http_service import OwnerHttpServer
from dagger_rtx.http_transport import BoundedBearerJsonTransport, TransientTransportError
from dagger_rtx.owner_client import HttpOwnerClient
from dagger_rtx.orchestrator import StepOrchestrator

from test.rtx.conftest import PROJECT_ROOT, make_key, make_owner, prepare


def _token_file(tmp_path):
    path = tmp_path / "token"
    path.write_text("test-token\n")
    path.chmod(0o600)
    return path


def test_real_loopback_http_contract_and_complete_retry(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    token = _token_file(tmp_path)
    with OwnerHttpServer(owner, token) as local, FakeH200Server(shape_profile=tiny_profile) as remote:
        owner_client = HttpOwnerClient(BoundedBearerJsonTransport(local.base_url, token))
        transport = BoundedBearerJsonTransport(remote.base_url, token)
        orchestrator = StepOrchestrator(owner_client, transport, tmp_path / "orchestrator")
        result = orchestrator.run_step(make_key())
        assert result["phase"] == "COMPLETE"
        assert backend.step_calls == 1
        assert remote.state.prepare_calls == 1
        assert remote.state.commit_calls == 1
        schema = json.loads((PROJECT_ROOT / "config/render_student_boundary.v1.schema.json").read_text())
        Draft202012Validator({"$ref": "#/$defs/commitRequest", "$defs": schema["$defs"]}).validate(
            remote.state.last_commit_request
        )
        Draft202012Validator({"$ref": "#/$defs/commitResponse", "$defs": schema["$defs"]}).validate(
            result["remote_ack"]
        )
        again = orchestrator.run_step(make_key())
        assert again == result
        assert backend.step_calls == 1
        assert remote.state.prepare_calls == 1
        assert remote.state.commit_calls == 1


def test_remote_prepare_retry_occurs_before_any_step(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        remote.state.fail_next_prepare = 1
        orchestrator = StepOrchestrator(
            owner, BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path)), tmp_path / "orchestrator"
        )
        with pytest.raises(TransientTransportError, match="HTTP 503"):
            orchestrator.run_step(make_key())
        assert backend.step_calls == 0
        assert orchestrator.run_step(make_key())["phase"] == "COMPLETE"
        assert backend.step_calls == 1


def test_remote_commit_retry_never_repeats_env_step(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        remote.state.fail_next_commit = 1
        orchestrator = StepOrchestrator(
            owner, BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path)), tmp_path / "orchestrator"
        )
        with pytest.raises(TransientTransportError, match="HTTP 503"):
            orchestrator.run_step(make_key())
        assert backend.step_calls == 1
        state_path = tmp_path / "orchestrator/test-run/D1/actor-0/window-0/step-000.json"
        assert json.loads(state_path.read_text())["phase"] == "LOCAL_COMMITTED"
        assert orchestrator.run_step(make_key())["phase"] == "COMPLETE"
        assert backend.step_calls == 1


def test_wrong_bearer_is_rejected_before_step(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        bad = tmp_path / "bad-token"
        bad.write_text("wrong\n")
        bad.chmod(0o600)
        orchestrator = StepOrchestrator(
            owner, BoundedBearerJsonTransport(remote.base_url, bad), tmp_path / "orchestrator"
        )
        with pytest.raises(RuntimeError, match="HTTP 401"):
            orchestrator.run_step(make_key())
        assert backend.step_calls == 0


def test_h200_canonical_response_with_trailing_newline_is_accepted(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    with FakeH200Server(response_trailing_newline=True, shape_profile=tiny_profile) as remote:
        orchestrator = StepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path)),
            tmp_path / "orchestrator",
        )
        assert orchestrator.run_step(make_key())["phase"] == "COMPLETE"
        assert backend.step_calls == 1


def test_owner_service_refuses_nonloopback_and_unsafe_token(tmp_path, tiny_profile):
    owner, _ = make_owner(tmp_path, tiny_profile)
    token = _token_file(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        OwnerHttpServer(owner, token, host="0.0.0.0")
    token.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        OwnerHttpServer(owner, token)


def test_remote_prepare_post_effect_retry_is_durable_and_observation_digest_is_canonical(
    tmp_path, tiny_profile
):
    owner, backend = make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        remote.state.fail_after_next_prepare = 1
        orchestrator = StepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path)),
            tmp_path / "orchestrator",
        )
        with pytest.raises(RuntimeError, match="HTTP 503"):
            orchestrator.run_step(make_key())
        assert backend.step_calls == 0
        assert remote.state.prepare_calls == 1
        assert remote.state.prepare_effects == 1

        assert orchestrator.run_step(make_key())["phase"] == "COMPLETE"
        assert backend.step_calls == 1
        assert remote.state.prepare_calls == 2
        assert remote.state.prepare_effects == 1
        request = remote.state.last_prepare_request
        response = next(iter(remote.state.prepares.values()))[1]
        arrays = decode_arrays(request["arrays"])
        key = StepKey.from_mapping(request["step_key"])
        assert response["observation_digest"] == observation_input_digest(key, request["metadata"], arrays)


def test_remote_commit_post_effect_retry_never_duplicates_commit_or_env_step(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        remote.state.fail_after_next_commit = 1
        orchestrator = StepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path)),
            tmp_path / "orchestrator",
        )
        with pytest.raises(RuntimeError, match="HTTP 503"):
            orchestrator.run_step(make_key())
        assert backend.step_calls == 1
        assert remote.state.commit_calls == 1
        assert remote.state.commit_effects == 1
        durable_snapshot = remote.state.durable_snapshot()

    # Simulate the H200 process disappearing after its durable commit effect but
    # before the 200 response reached RTX.  Only the persisted ledger is restored.
    with FakeH200Server(
        shape_profile=tiny_profile,
        durable_snapshot=durable_snapshot,
    ) as restarted_remote:
        restarted_orchestrator = StepOrchestrator(
            owner,
            BoundedBearerJsonTransport(restarted_remote.base_url, _token_file(tmp_path)),
            tmp_path / "orchestrator",
        )
        assert restarted_orchestrator.run_step(make_key())["phase"] == "COMPLETE"
        assert backend.step_calls == 1
        assert restarted_remote.state.commit_calls == 1
        assert restarted_remote.state.commit_effects == 1


def test_production_shape_prepare_conforms_over_http(tmp_path):
    profile = RavenShapeProfile()
    owner, backend = make_owner(tmp_path, profile)
    token = _token_file(tmp_path)
    with FakeH200Server(shape_profile=profile) as remote:
        orchestrator = StepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, token),
            tmp_path / "orchestrator",
        )
        result = orchestrator.run_step(make_key())
        assert result["phase"] == "COMPLETE"
        assert backend.step_calls == 1
        schema = json.loads(
            (PROJECT_ROOT / "config/render_student_boundary.v1.schema.json").read_text()
        )
        Draft202012Validator({"$ref": "#/$defs/prepareRequest", "$defs": schema["$defs"]}).validate(
            remote.state.last_prepare_request
        )
        response = next(iter(remote.state.prepares.values()))[1]
        Draft202012Validator({"$ref": "#/$defs/prepareResponse", "$defs": schema["$defs"]}).validate(
            response
        )


def test_h200_rejects_cross_field_digest_and_conditional_raytraced_violations(tmp_path, tiny_profile):
    owner, _ = make_owner(tmp_path, tiny_profile)
    local = prepare(owner, make_key())
    valid = build_h200_prepare(local, "h200-prepare-valid")
    token = _token_file(tmp_path)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        transport = BoundedBearerJsonTransport(remote.base_url, token)

        wrong_beta = copy.deepcopy(valid)
        wrong_beta["idempotency_key"] = "h200-prepare-wrong-beta"
        wrong_beta["beta"] = 0.5
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/prepare", wrong_beta, max_response_bytes=1 << 20)

        wrong_digest = copy.deepcopy(valid)
        wrong_digest["idempotency_key"] = "h200-prepare-wrong-digest"
        wrong_digest["capture_digest"] = "0" * 64
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/prepare", wrong_digest, max_response_bytes=1 << 20)

        wrong_assignment = copy.deepcopy(valid)
        wrong_assignment["idempotency_key"] = "h200-prepare-wrong-assignment"
        wrong_assignment["metadata"]["episode_assignment"]["use_raytraced"] = False
        wrong_assignment_arrays = decode_arrays(wrong_assignment["arrays"])
        wrong_assignment["capture_digest"] = payload_digest(
            wrong_assignment["metadata"], wrong_assignment_arrays
        )
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/prepare", wrong_assignment, max_response_bytes=1 << 20)

        missing_ray = copy.deepcopy(valid)
        missing_ray["idempotency_key"] = "h200-prepare-missing-ray"
        for view in ("head_color", "hand_left", "hand_right"):
            del missing_ray["arrays"][f"raytraced_rgb__{view}"]
        decoded = decode_arrays(missing_ray["arrays"])
        missing_ray["capture_digest"] = payload_digest(missing_ray["metadata"], decoded)
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/prepare", missing_ray, max_response_bytes=1 << 20)

        generated = copy.deepcopy(valid)
        generated["idempotency_key"] = "h200-prepare-generated"
        generated["metadata"]["observation_source"] = "generated_controls"
        generated["metadata"]["episode_assignment"]["observation_source"] = "generated_controls"
        generated["metadata"]["episode_assignment"]["use_raytraced"] = False
        for view in ("head_color", "hand_left", "hand_right"):
            del generated["arrays"][f"raytraced_rgb__{view}"]
        generated_arrays = decode_arrays(generated["arrays"])
        generated["capture_digest"] = payload_digest(generated["metadata"], generated_arrays)
        response = transport.post("/v1/steps/prepare", generated, max_response_bytes=1 << 20)
        assert response["observation_digest"] == observation_input_digest(
            make_key(), generated["metadata"], generated_arrays
        )

        unexpected_ray = copy.deepcopy(generated)
        unexpected_ray["idempotency_key"] = "h200-prepare-unexpected-ray"
        unexpected_ray["arrays"]["raytraced_rgb__head_color"] = valid["arrays"][
            "raytraced_rgb__head_color"
        ]
        unexpected_arrays = decode_arrays(unexpected_ray["arrays"])
        unexpected_ray["capture_digest"] = payload_digest(unexpected_ray["metadata"], unexpected_arrays)
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/prepare", unexpected_ray, max_response_bytes=1 << 20)


def test_h200_commit_binds_selection_and_sealed_window_only_at_index_80(tmp_path, tiny_profile):
    owner, _ = make_owner(tmp_path, tiny_profile)
    local = prepare(owner, make_key())
    remote_prepare = build_h200_prepare(local, "h200-prepare-bind")
    token = _token_file(tmp_path)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        transport = BoundedBearerJsonTransport(remote.base_url, token)
        selection = transport.post("/v1/steps/prepare", remote_prepare, max_response_bytes=1 << 20)
        local_commit = owner.commit_step(
            {
                "contract_version": CONTRACT_VERSION,
                "step_key": selection["step_key"],
                "prepare_receipt": selection["prepare_receipt"],
                "selection_id": selection["selection_id"],
                "selected_source": selection["selected_source"],
                "beta": selection["beta"],
                "selected_action": selection["selected_action"],
            }
        )
        commit = build_h200_commit(selection, local_commit, "h200-commit-bind")

        bad_binding = copy.deepcopy(commit)
        bad_binding["idempotency_key"] = "h200-commit-bad-binding"
        bad_binding["selection_receipt"] = "0" * 64
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/commit", bad_binding, max_response_bytes=1 << 20)

        bad_seal = copy.deepcopy(commit)
        bad_seal["idempotency_key"] = "h200-commit-bad-seal"
        bad_seal["sealed_window"] = {"not": "allowed-before-index-80"}
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/commit", bad_seal, max_response_bytes=1 << 20)

        bad_receipt = copy.deepcopy(commit)
        bad_receipt["idempotency_key"] = "h200-commit-bad-receipt"
        bad_receipt["commit_receipt"] = "0" * 64
        with pytest.raises(RuntimeError, match="HTTP 400"):
            transport.post("/v1/steps/commit", bad_receipt, max_response_bytes=1 << 20)

        assert transport.post("/v1/steps/commit", commit, max_response_bytes=1 << 20)["status"] == "committed"


def test_index_80_commit_requires_and_validates_actual_sealed_audit(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    token = _token_file(tmp_path)
    with FakeH200Server(shape_profile=tiny_profile) as remote:
        orchestrator = StepOrchestrator(
            owner,
            BoundedBearerJsonTransport(remote.base_url, token),
            tmp_path / "orchestrator",
        )
        result = None
        for index in range(81):
            result = orchestrator.run_step(make_key(index))
        assert result["phase"] == "COMPLETE"
        assert backend.step_calls == 81
        assert remote.state.prepare_effects == 81
        assert remote.state.commit_effects == 81
        assert remote.state.last_commit_request["step_key"]["rollout_step_index"] == 80
        assert remote.state.last_commit_request["sealed_window"]["artifact_role"] == "rtx_identity_outcome_audit"
        assert remote.state.last_commit_request["sealed_window"]["round_ids"] == [1] * 81
        schema = json.loads(
            (PROJECT_ROOT / "config/render_student_boundary.v1.schema.json").read_text()
        )
        validator = Draft202012Validator(
            {"$ref": "#/$defs/commitRequest", "$defs": schema["$defs"]}
        )
        validator.validate(remote.state.last_commit_request)
        missing_seal = copy.deepcopy(remote.state.last_commit_request)
        missing_seal["sealed_window"] = None
        assert list(validator.iter_errors(missing_seal))
