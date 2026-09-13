from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    pending = [extractor.__code__]
    names = set()
    while pending:
        code = pending.pop()
        names.update(code.co_names)
        pending.extend(value for value in code.co_consts if inspect.iscode(value))
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


def test_runtime_initialization_matches_raven_random_episode_offsets(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(entrypoint_module, "verify_origin_sources", lambda: {})
    staged_module = types.ModuleType("gr00t.rl.envs.base_task.staged_task_base")
    staged_module.StagedTaskBase = type("StagedTaskBase", (), {})
    monkeypatch.setitem(
        sys.modules, "gr00t.rl.envs.base_task.staged_task_base", staged_module
    )

    class RandomizedEnv:
        num_envs = 64
        max_episode_length = 512

        def __init__(self):
            self.reset_calls = 0
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64)

        def reset_all(self):
            self.reset_calls += 1
            return {"actor_obs": torch.zeros(self.num_envs, 1)}

    env = RandomizedEnv()
    trainer = type(
        "Trainer",
        (),
        {
            "config": {"init_at_random_ep_len": True},
            "env": env,
            "accelerator": SimpleNamespace(device=torch.device("cpu")),
            "_train_rollout_mode": lambda self: None,
        },
    )()
    runtime = NativeRavenRuntime.initialize(trainer)

    assert env.reset_calls == 1
    assert runtime.obs_dict["actor_obs"].device.type == "cpu"
    assert bool((env.episode_length_buf >= 0).all())
    assert bool((env.episode_length_buf < env.max_episode_length).all())
    assert torch.unique(env.episode_length_buf).numel() > 1


def test_runtime_initialization_randomizes_staged_task_clocks(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(entrypoint_module, "verify_origin_sources", lambda: {})

    class StagedTaskBase:
        pass

    staged_module = types.ModuleType("gr00t.rl.envs.base_task.staged_task_base")
    staged_module.StagedTaskBase = StagedTaskBase
    monkeypatch.setitem(
        sys.modules, "gr00t.rl.envs.base_task.staged_task_base", staged_module
    )

    class RandomizedStagedEnv(StagedTaskBase):
        num_envs = 64
        max_episode_length = 512

        def __init__(self):
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64)
            self.max_stage_time = torch.tensor([300, 50, 50, 100, 100])
            self.time_in_stage_buf = torch.zeros(self.num_envs, dtype=torch.int64)
            self.actual_time_in_stage_buf = torch.full(
                (self.num_envs,), -1, dtype=torch.int64
            )

        def reset_all(self):
            return {"actor_obs": torch.zeros(self.num_envs, 1)}

    env = RandomizedStagedEnv()
    trainer = type(
        "Trainer",
        (),
        {
            "config": {"init_at_random_ep_len": True},
            "env": env,
            "accelerator": SimpleNamespace(device=torch.device("cpu")),
            "_train_rollout_mode": lambda self: None,
        },
    )()
    NativeRavenRuntime.initialize(trainer)

    assert bool((env.time_in_stage_buf >= 0).all())
    assert bool((env.time_in_stage_buf < env.max_stage_time[0]).all())
    assert torch.unique(env.time_in_stage_buf).numel() > 1
    assert torch.equal(env.actual_time_in_stage_buf, env.time_in_stage_buf)


def test_runtime_initialization_does_not_own_ref_model_rollout_lifecycle():
    initialize_source = inspect.getsource(NativeRavenRuntime.initialize)
    assert "ref_model.init_rollout" not in initialize_source
    assert "policy_model.init_rollout" not in initialize_source
    assert "clear_rollout" not in initialize_source
