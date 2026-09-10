from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import dagger_rtx.native_entrypoints as entrypoint_module
from dagger_rtx.backend import BackendInvariantError
from dagger_rtx.contracts import PINNED_RAVEN_ROOT
from dagger_rtx.native_entrypoints import (
    EXPECTED_RAW_ARCHIVE_AST_SHA256,
    EXPECTED_SOURCE_SHA256,
    TEACHER_HOMIE_SOURCE,
    VIDEOGEN_SOURCE,
    NativeRavenRuntime,
    _compile_raw_archive_extractor,
    verify_origin_sources,
)


EXPECTED_SOURCE_IDENTITIES = {
    "distill_trainer_videogen.py": (
        43015,
        "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b",
    ),
    "distill_trainer_obj_pred_homie_api.py": (
        37335,
        "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584",
    ),
}


def test_default_runtime_root_is_exact_pinned_raven_snapshot():
    assert entrypoint_module.RAVEN_ROOT == Path(PINNED_RAVEN_ROOT)
    assert VIDEOGEN_SOURCE.is_file()
    assert TEACHER_HOMIE_SOURCE.is_file()


def test_source_and_raw_archive_ast_provenance_matches_retained_bytes():
    attestation = verify_origin_sources()
    assert set(attestation) == {"source_sha256", "raw_archive_ast_sha256"}
    assert attestation["raw_archive_ast_sha256"] == EXPECTED_RAW_ARCHIVE_AST_SHA256
    assert set(EXPECTED_SOURCE_SHA256) == {VIDEOGEN_SOURCE, TEACHER_HOMIE_SOURCE}
    for path, expected_sha in EXPECTED_SOURCE_SHA256.items():
        expected_size, expected_identity = EXPECTED_SOURCE_IDENTITIES[path.name]
        assert path.stat().st_size == expected_size
        assert expected_sha == expected_identity
        assert attestation["source_sha256"][str(path)] == expected_identity


def test_source_attestation_fails_closed_on_a_byte_hash_mismatch(monkeypatch):
    original = entrypoint_module._sha256

    def mismatched(path):
        if path == VIDEOGEN_SOURCE:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(entrypoint_module, "_sha256", mismatched)
    with pytest.raises(BackendInvariantError, match="source hash mismatch"):
        verify_origin_sources()


def test_extracted_archive_callback_contains_only_attested_compute_surface():
    extractor = _compile_raw_archive_extractor()
    assert Path(extractor.__code__.co_filename).resolve() == VIDEOGEN_SOURCE.resolve()
    names = set(extractor.__code__.co_names)
    assert {"log", "clip", "round", "resize", "INTER_NEAREST"} <= names
    assert not ({"save", "savez", "open", "makedirs", "storage"} & names)


def test_teacher_homie_gate_attests_exact_native_semantic_calls():
    tree = ast.parse(TEACHER_HOMIE_SOURCE.read_text(encoding="utf-8"))
    policy = entrypoint_module._function(
        tree, "TRLDistillTrainerObjPredHomieAPI", "policy_step"
    )
    attrs = {
        node.attr for node in ast.walk(policy) if isinstance(node, ast.Attribute)
    }
    assert {"homie_walk_model", "homie_stand_model", "act_inference"} <= attrs
    assert any(
        isinstance(node, ast.Constant) and node.value == "gt_actions"
        for node in ast.walk(policy)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rollout_with_obj_pred"
        for node in ast.walk(policy)
    )


def test_entrypoint_source_does_not_reimplement_scientific_formulas():
    source = inspect.getsource(entrypoint_module)
    # The implementation compiles hash-attested native AST nodes instead of
    # transcribing h5_v4 or raw-depth math into the RTX service.
    assert "ast.FunctionDef(" in source
    assert "_h5v4_control" not in source
    assert "np.percentile" not in source
    assert "np.log(" not in source
    assert "16383" not in source


def test_runtime_initialization_rejects_random_episode_offsets_before_reset():
    class NeverResetEnv:
        def reset_all(self):
            raise AssertionError("invalid configuration must fail before reset")

    trainer = type(
        "Trainer",
        (),
        {"config": {"init_at_random_ep_len": True}, "env": NeverResetEnv()},
    )()
    with pytest.raises(BackendInvariantError, match="init_at_random_ep_len=false"):
        NativeRavenRuntime.initialize(trainer)


def test_runtime_initialization_does_not_own_ref_model_rollout_lifecycle():
    initialize_source = inspect.getsource(NativeRavenRuntime.initialize)
    assert "ref_model.init_rollout" not in initialize_source
    assert "policy_model.init_rollout" not in initialize_source
    assert "clear_rollout" not in initialize_source
