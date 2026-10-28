"""Actual pinned Raven actor/wrapper lifecycle, without Isaac or a GPU claim.

Actor lifecycle/inference and the distillation wrapper are unmodified AST
methods. Only the backbone, observation buffering and outer PPO loop are tiny
CPU dependencies. In particular clear_rollout is NOT a test double.
"""
import ast
import hashlib
import logging
from types import SimpleNamespace

import pytest
import torch

from test.deployment.source_paths import REPO_ROOT
from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import RolloutLifecycle


RAVEN = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl"


def methods(path, cls, names, expected=None):
    raw = path.read_bytes()
    if expected:
        assert hashlib.sha256(raw).hexdigest() == expected
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == cls)
    selected = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    return selected


@pytest.fixture
def code():
    actor = methods(RAVEN / "modules/actor_critic_modules.py", "Actor",
                    {"init_rollout", "clear_rollout", "act_inference"},
                    "cf0f272e6bef2aa5ab001a80fd6661c3dc29b97785dde1c4166f4687fa24ff56")
    reference = methods(RAVEN / "trainer/distill_trainer.py", "TRLDistillTrainer",
                        {"_rollout_step"},
                        "dc386dcf8b2be0c151403fd0e8841129c939d99dc5e9e7850e92829b3ad50ca2")
    current = methods(REPO_ROOT / "pipeline/dagger/rtx/code/src/dagger_rtx/native_service_trainer.py",
                      "NativeDaggerRTXServiceTrainer", {"_rollout_lifecycle"})

    class OuterPPO:
        def _rollout_step(self, model, observation):
            model.init_rollout()
            result = [self.ref_model.act_inference(observation) for _ in range(81)]
            model.clear_rollout()
            return result

    nodes = [ast.ClassDef(name="Actor", bases=[], keywords=[], decorator_list=[], body=actor),
             ast.ClassDef(name="Reference", bases=[ast.Name(id="OuterPPO", ctx=ast.Load())],
                          keywords=[], decorator_list=[], body=reference),
             ast.ClassDef(name="Current", bases=[], keywords=[], decorator_list=[], body=current)]
    namespace = dict(TensorDict=dict, OuterPPO=OuterPPO, logging=logging,
                     RolloutLifecycle=RolloutLifecycle, __name__=__name__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "actual_actor_and_wrappers", "exec"), namespace)
    return SimpleNamespace(**{k: namespace[k] for k in ("Actor", "Reference", "Current")})


def teacher(code):
    value = code.Actor()
    value.distribution = None  # Actual Actor constructor's initial state.

    def update(observation, mask, dones):
        value.obs_dict_buffer = {key: tensor[:, None] for key, tensor in observation.items()}
        value.dones_buffer = dones
        return mask

    value._update_obs_buffer = update
    value.forward = lambda obs_dict, **kw: obs_dict["x"] * 2
    return value


def policy():
    return SimpleNamespace(init_rollout=lambda: None, clear_rollout=lambda: None)


def test_actual_actor_second_clear_raises_after_clearing_buffers(code):
    actor = teacher(code)
    for window in range(2):
        actor.init_rollout()
        actor.act_inference({"x": torch.ones(16, 11)})
        if window == 0:
            actor.clear_rollout()
        else:
            with pytest.raises(AttributeError, match="distribution"):
                actor.clear_rollout()
        assert actor.steps == 0 and actor.obs_dict_buffer == {} and actor.dones_buffer is None


def test_five_81_step_windows_match_actual_raven_wrapper(code, caplog):
    original, current = code.Reference(), code.Current()
    original.ref_model, current.ref_model = teacher(code), teacher(code)
    current.policy_model = policy()
    lifecycle = current._rollout_lifecycle()
    observation = {"x": torch.arange(16 * 11).reshape(16, 11).float()}
    rng = torch.get_rng_state().clone()
    for _ in range(5):
        expected = original._rollout_step(policy(), observation)
        lifecycle.begin()
        actual = [current.ref_model.act_inference(observation) for _ in range(81)]
        lifecycle.finish()
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        for actor in (original.ref_model, current.ref_model):
            assert actor.steps == 0 and actor.obs_dict_buffer == {} and actor.dones_buffer is None
            assert not hasattr(actor, "distribution")
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum("teacher clear_rollout failed" in r.message for r in caplog.records) == 4


@pytest.mark.parametrize("hook", ["init_rollout", "clear_rollout"])
@pytest.mark.parametrize("error", [AttributeError, RuntimeError])
def test_optional_teacher_hook_failure_matches_original(code, hook, error):
    def fail():
        raise error("optional teacher hook")

    original, current = code.Reference(), code.Current()
    original.ref_model = SimpleNamespace(init_rollout=lambda: None, clear_rollout=lambda: None,
                                        act_inference=lambda *a: torch.zeros(1))
    setattr(original.ref_model, hook, fail)
    current.ref_model = original.ref_model
    current.policy_model = policy()
    original._rollout_step(policy(), {})
    lifecycle = current._rollout_lifecycle()
    lifecycle.begin()
    lifecycle.finish()


@pytest.mark.parametrize("hook", ["init_rollout", "clear_rollout"])
def test_student_errors_propagate_without_running_later_teacher_hook(code, hook):
    calls = []

    def fail():
        raise RuntimeError("student fault")

    value = code.Current()
    value.policy_model = policy()
    setattr(value.policy_model, hook, fail)
    value.ref_model = SimpleNamespace(init_rollout=lambda: calls.append("begin"),
                                      clear_rollout=lambda: calls.append("finish"))
    lifecycle = value._rollout_lifecycle()
    with pytest.raises(RuntimeError, match="student fault"):
        (lifecycle.begin if hook == "init_rollout" else lifecycle.finish)()
    assert calls == []
