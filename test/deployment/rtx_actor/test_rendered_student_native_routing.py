"""Execute native orchestration on CPU; never create Isaac, clients or GPUs."""
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import initial_robot_refresh_diagnostic as refresh
import rendered_success_observations as observations
import rendered_student_success_contract as rendered
from explicit_motion_panel import canonical, run_panel, validate_panel
from native_artifact_io import publish_json
from teacher_action_diagnostic import DISTILL, RAVEN_PPO, validate_mode
from test.deployment.rtx_actor.test_production_success_contract import accepted
from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config, cohort_result


def evaluator_function(tmp_path, evaluator):
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_explicit_panel")
    class OwnedPath(type(Path())):
        def is_relative_to(self, *others):
            return super().is_relative_to(tmp_path if others == ("/dagger-runtime",) else others[0])
    scope = {"validate_panel": validate_panel, "validate_mode": validate_mode,
        "require_corrected_execution": refresh.require_corrected_execution,
        "validate_initial_refresh_panel": refresh.validate_panel_mode,
        "STUDENT_CLEAR_ONCE": refresh.STUDENT_CLEAR_ONCE,
        "SMOKE_STUDENT_CLEAR_ONCE": refresh.SMOKE_STUDENT_CLEAR_ONCE,
        "DISTILL": DISTILL, "RAVEN_PPO": RAVEN_PPO, "Path": OwnedPath,
        "ExplicitMotionSuccessEvaluator": evaluator, "publish_json": publish_json, "run_panel": run_panel}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native-rendered-success-routing", "exec"), scope)
    return scope["evaluate_explicit_panel"]


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("fault", [None, "lost_callback", "wrong_receipt"])
def test_every_native_cohort_installs_bound_renderer_and_failure_never_falls_back(
        accepted, tmp_path, monkeypatch, smoke, fault):
    from smoke_success_contract import make_panel
    panel = make_panel(accepted) if smoke else accepted
    config = rendering_config(panel)
    prepared, adapters, policy_actions = [], [], []
    failure_index = min(1, len(panel["cohorts"]) - 1)
    # Stale inherited state must not be reused by the next explicit cohort.
    trainer = SimpleNamespace(env=object(), _success_observation_transform=object(),
        _rendered_paired_baseline={"stale": True}, _rendered_cohort_index=999,
        _rendered_success_evaluation_binding={"stale": True})

    class Adapter:
        def __init__(self, owner, **kwargs):
            assert owner is trainer
            self.binding = deepcopy(kwargs["evaluation_binding"])
            self.index = self.binding["cohort_index"]
            assert self.binding == rendered.observation_binding(panel, config, self.index)
            assert kwargs["anchor_seed"] == (config["anchors"]["seed"] + self.index) % (2 ** 32)
            assert kwargs["anchor_files"] == config["anchors"]["files"]
            assert kwargs["renderer_identity"] == config["renderer"]["identity"]
            assert kwargs["client_factory"](0) is not kwargs["client_factory"](1)
            self.closed = False
            adapters.append(self)

        def __call__(self, obs, dones, step):
            assert step == 0
            return {**obs, "vision_obs": obs["vision_obs"] + 1}

        def receipt(self):
            return {"evaluation_binding": deepcopy(self.binding)}

        def close(self):
            self.closed = True

    class Evaluator:
        def evaluate_cohort(self, assignment, **kwargs):
            index = self._rendered_success_evaluation_binding["cohort_index"]
            assert self.env is trainer.env and self._require_rendered_success_observations
            assert not hasattr(self, "_rendered_paired_baseline")
            assert not hasattr(self, "_rendered_cohort_index")
            assert self._success_observation_transform is adapters[-1]
            assert assignment == panel["cohorts"][index]["assignment"]
            assert kwargs["checkpoint_sha256"] == config["checkpoint_sha256"]
            if fault == "lost_callback" and index == failure_index:
                del self._success_observation_transform
            obs = {"vision_obs": torch.zeros(len(assignment["lanes"]), 3), "actor_obs": torch.zeros(1)}
            transformed = observations.apply_observation_transform(self, obs, torch.zeros(len(assignment["lanes"])), 0)
            assert transformed["vision_obs"].eq(1).all()
            assert transformed["actor_obs"] is obs["actor_obs"]
            policy_actions.append(index)
            result = cohort_result(panel, index=index, config=config)
            if fault == "wrong_receipt" and index == failure_index:
                result["rendered_observation_receipt"]["evaluation_binding"]["checkpoint_sha256"] = "f" * 64
            return result

    monkeypatch.setattr(observations, "GeneratedSuccessObservations", Adapter)
    monkeypatch.setitem(sys.modules, "pipeline.dagger.services.render_student",
                        SimpleNamespace(_Image3FLoopbackClient=lambda *args: object()))
    execute = evaluator_function(tmp_path, Evaluator)
    output = tmp_path / "rendered-native"
    kwargs = dict(root=output, checkpoint_path="/diagnostic-assets/selected.pt",
        checkpoint_sha256=config["checkpoint_sha256"], prepare_cohort=prepared.append,
        report=lambda row: None, rendered_observation_config=config, **rendered.evaluation_options(panel))
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            execute(trainer, panel, **kwargs)
        assert len(prepared) == failure_index + 1
        if fault == "lost_callback":
            assert failure_index not in policy_actions
        assert not (output / "result.json").exists()
    else:
        value = execute(trainer, panel, **kwargs)
        assert value["completed_episodes"] == panel["episode_count"]
        assert len(prepared) == len(adapters) == len(policy_actions) == len(panel["cohorts"])
        assert len(list((output / "cohort-receipts").glob("*.json"))) == len(panel["cohorts"])
        rendered.validate_rendered_result(panel, value, config["checkpoint_sha256"], config)
    assert adapters and all(adapter.closed for adapter in adapters)


def test_direct_panel_call_without_config_fails_before_scene_or_output(accepted, tmp_path):
    execute = evaluator_function(tmp_path, object)
    prepared = []
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="requires rendered observation configuration"):
        execute(SimpleNamespace(), accepted, root=output, checkpoint_path="unused.pt", checkpoint_sha256="b" * 64,
            prepare_cohort=prepared.append, report=lambda row: None, **rendered.evaluation_options(accepted))
    assert not prepared and not output.exists()


def test_resident_generic_receipt_binds_actual_source_and_forwards_config(accepted, tmp_path, monkeypatch):
    config = rendering_config(accepted)
    paths = {"panel.json": tmp_path / "panel.json", "config.json": tmp_path / "config.json"}
    paths["panel.json"].write_bytes(canonical(accepted))
    paths["config.json"].write_bytes(canonical(config))
    payload = dict(panel_file="panel.json", rendered_observation_config_file="config.json",
        rendered_observation_config_sha256=hashlib.sha256(canonical(config)).hexdigest(),
        checkpoint_sha256=config["checkpoint_sha256"], arm="bc", output_name="source-bound-test",
        **rendered.evaluation_options(accepted))
    events, published = [], []
    owner = SimpleNamespace(runtime_root=tmp_path, _consumed_outputs=set(), artifact=lambda key: paths[key],
        validate=lambda operation, request: events.append("admission"), source_sha256="f" * 64,
        trainer=SimpleNamespace(env=SimpleNamespace(simulator=SimpleNamespace(sim=SimpleNamespace(
            pause=lambda: events.append("pause"))))), scene_initializations=149, phase_resets=148,
        profile="explicit_evaluation", _kit_unchanged=lambda: events.append("same-kit"))
    result = {"test_stub": True}
    def evaluate(trainer, panel, **kwargs):
        assert events == ["admission"] and panel == accepted and trainer is owner.trainer
        assert kwargs["rendered_observation_config"] == config
        assert kwargs["training_support_selection"] is kwargs["rendered_paired_baseline"] is None
        assert kwargs["checkpoint_sha256"] == config["checkpoint_sha256"]
        events.append("evaluation")
        return result
    monkeypatch.setitem(sys.modules, "isaac6_motion_evaluator", SimpleNamespace(evaluate_explicit_panel=evaluate))
    tree = ast.parse((HERE / "resident_isaac_trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResidentRavenBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run_explicit_test")
    scope = {"Path": Path, "os": os, "json": json,
        "atomic_write_json": lambda path, value, **kwargs: published.append((path, deepcopy(value)))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "native-generic-receipt", "exec"), scope)
    receipt = scope["_run_explicit_test"](owner, payload, lambda *args, **kwargs: None)
    assert receipt["source_sha256"] == owner.source_sha256
    assert receipt["result"] is result and receipt["native_initializations"] == 1
    assert receipt["phase_resets"] == 148 and receipt["optimizer_steps"] == 0
    assert published[0][1] == receipt and owner.active_output is None
    assert events == ["admission", "evaluation", "same-kit", "pause"]
