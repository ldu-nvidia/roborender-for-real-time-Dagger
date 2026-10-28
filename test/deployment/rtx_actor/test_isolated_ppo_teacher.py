"""CPU-only private teacher and forbidden-action tests; no native imports."""
from test.deployment.source_paths import ACTOR_ROOT
import copy
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, str(ACTOR_ROOT))
import isolated_ppo_teacher as isolation
import teacher_action_diagnostic as diagnostic
from test.deployment.rtx_actor.test_teacher_action_diagnostic import _checkpoint_attested_trainer


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.std = torch.nn.Parameter(torch.tensor([1.2, .8]))
        self.register_buffer("rms", torch.arange(3, dtype=torch.float32))
        self.algo_config = OmegaConf.create({"max_noise_std": .001, "use_clampped_std": False})
        self.max_noise_std, self.clamp_noise_std = .001, True
        self.distribution, self.dones_buffer = None, None
        self.obs_dict_buffer, self.steps = {}, 0
        self.eval()

    def forward(self, value):
        return self.linear(value - self.rms)

    def rollout(self, value):
        with torch.no_grad():
            self.std.clamp_(max=self.max_noise_std)
            self.distribution = torch.distributions.Normal(self(value), self.std)
            self.obs_dict_buffer = {"obs": value}
            self.steps += 1
            return self.distribution.sample()


def owner(teacher=None):
    student = torch.nn.Linear(3, 2)
    return SimpleNamespace(ref_model=teacher or Teacher(), policy_model=student,
                           unwrapped_model=student, policy_step=lambda: None)


def facade(trainer):
    return SimpleNamespace(**vars(trainer))


def test_build_preserves_all_rng_shared_graph_and_strict_initial_state():
    trainer = owner()
    before, rng = isolation._Snapshot(trainer.ref_model), isolation._rng_state()
    private = isolation.build_isolated_teacher(trainer)
    assert private is not trainer.ref_model
    assert isolation._rng_equal(rng, isolation._rng_state())
    before.assert_equal(trainer.ref_model)
    assert not any(m.training for m in private.modules())
    assert not any(p.requires_grad for p in private.parameters())
    assert all(torch.equal(value, private.state_dict()[name])
               for name, value in trainer.ref_model.state_dict().items())
    assert private.algo_config is not trainer.ref_model.algo_config
    private.algo_config.max_noise_std = 9
    with torch.no_grad():
        private.std.zero_()
        private.rms.add_(3)
    before.assert_equal(trainer.ref_model)
    assert not torch.cuda.is_initialized()


def test_nested_omegaconf_parent_topology_is_private_without_changing_interpolation():
    trainer = owner()
    config = OmegaConf.create({"shared_value": 0.001, "algo": {
        "max_noise_std": "${shared_value}", "backbone": {"width": 3}},
        "robot": {"action_dim": 2}})
    trainer.ref_model.algo_config = config.algo
    trainer.ref_model.linear.env_config = config
    # Match custom_instantiate's nested copy whose parent is retained by
    # OmegaConf but which is not itself in the original parent's child tree.
    loose_backbone = copy.deepcopy(config.algo.backbone)
    trainer.ref_model.linear.module_config_dict = loose_backbone
    before, rng = isolation._Snapshot(trainer.ref_model), isolation._rng_state()
    naive = copy.deepcopy(trainer.ref_model)
    with pytest.raises(RuntimeError, match="aliases shared teacher"):
        isolation._disjoint(trainer.ref_model, naive)
    private = isolation.build_isolated_teacher(trainer)
    private_root = private.linear.env_config
    assert private.algo_config is private_root.algo
    assert private.algo_config._get_parent() is private_root
    assert private.linear.module_config_dict._get_parent() is private_root.algo
    assert private.linear.module_config_dict is not private_root.algo.backbone
    assert private.algo_config.max_noise_std == trainer.ref_model.algo_config.max_noise_std == .001
    private_root.shared_value = .5
    assert private.algo_config.max_noise_std == .5
    assert trainer.ref_model.algo_config.max_noise_std == .001
    before.assert_equal(trainer.ref_model)
    assert isolation._rng_equal(rng, isolation._rng_state())


@pytest.mark.parametrize("change", [
    lambda teacher: setattr(teacher, "_original_ppo_std_policy", {}),
    lambda teacher: setattr(teacher, "steps", 1),
    lambda teacher: setattr(teacher, "distribution", object()),
    lambda teacher: teacher.obs_dict_buffer.update(obs=torch.zeros(1)),
    lambda teacher: setattr(teacher, "dones_buffer", torch.zeros(1)),
])
def test_build_rejects_used_or_ppo_mutated_teacher(change):
    trainer = owner()
    change(trainer.ref_model)
    with pytest.raises(RuntimeError, match="pristine pre-forward"):
        isolation.build_isolated_teacher(trainer)


def test_nonleaf_deepcopy_fails_cleanly_without_shared_fallback():
    trainer = owner()
    trainer.ref_model.cached_nonleaf = trainer.ref_model.std * 2
    before, rng = isolation._Snapshot(trainer.ref_model), isolation._rng_state()
    with pytest.raises(RuntimeError, match="deepcopy failed; no shared fallback"):
        isolation.build_isolated_teacher(trainer)
    before.assert_equal(trainer.ref_model)
    assert isolation._rng_equal(rng, isolation._rng_state())


def test_custom_deepcopy_alias_is_rejected():
    class AliasingTeacher(Teacher):
        def __deepcopy__(self, memo):
            return self
    trainer = owner(AliasingTeacher())
    with pytest.raises(RuntimeError, match="aliases shared teacher"):
        isolation.build_isolated_teacher(trainer)


def test_different_tensor_objects_with_shared_storage_are_rejected():
    trainer = owner()
    private = copy.deepcopy(trainer.ref_model)
    private.std = torch.nn.Parameter(trainer.ref_model.std.detach())
    assert private.std is not trainer.ref_model.std
    with pytest.raises(RuntimeError, match="aliases shared teacher"):
        isolation._disjoint(trainer.ref_model, private)


def test_rng_consuming_copy_rejected_and_rng_restored():
    class RandomTeacher(Teacher):
        def __deepcopy__(self, memo):
            random.random()
            np.random.random()
            torch.rand(1)
            return self
    trainer = owner(RandomTeacher())
    rng = isolation._rng_state()
    with pytest.raises(RuntimeError, match="construction consumed RNG"):
        isolation.build_isolated_teacher(trainer)
    assert isolation._rng_equal(rng, isolation._rng_state())
    assert not torch.cuda.is_initialized()


def test_private_ppo_mutation_does_not_touch_collection_teacher_and_restores_bindings():
    trainer = owner()
    private = isolation.build_isolated_teacher(trainer)
    evaluator = facade(trainer)
    shared, original_policy_step = trainer.ref_model, evaluator.policy_step
    before = isolation._Snapshot(shared)
    assert "forward" not in vars(trainer.policy_model)
    assert "rollout" not in vars(private)
    with isolation.teacher_only_execution(evaluator, private, shared_trainer=trainer) as receipt:
        assert evaluator.ref_model is private and trainer.ref_model is shared
        private.max_noise_std = 1.0
        private._original_ppo_std_policy = {"original_max_noise_std": 1.0}
        evaluator.ref_model.rollout(torch.zeros(2, 3))
        assert torch.equal(private.std, torch.tensor([1.0, .8]))
    assert receipt == {"schema": "roborender.isolated-ppo-teacher-execution.v1",
        "student_forward_calls": 0, "renderer_calls": 0, "distill_policy_step_calls": 0,
        "teacher_rollout_calls": 1, "shared_teacher_unchanged": True, "private_teacher_used": True}
    assert evaluator.ref_model is shared and evaluator.policy_step is original_policy_step
    assert "forward" not in vars(trainer.policy_model) and "rollout" not in vars(private)
    before.assert_equal(shared)


def test_actual_ppo_prepare_changes_only_private_teacher(tmp_path, monkeypatch):
    trainer, obs = _checkpoint_attested_trainer(tmp_path, monkeypatch)
    trainer.policy_model = torch.nn.Linear(3, 2)
    for role in ("walk", "stand"):
        homie = getattr(trainer.unwrapped_model, "homie_" + role + "_model")
        native_forward = homie.forward

        def output(value, native_forward=native_forward):
            assert not torch.is_grad_enabled()
            mean = native_forward(value)
            return {"actions": mean, "action_mean": mean}
        homie.forward = output
    private = isolation.build_isolated_teacher(trainer)
    evaluator = facade(trainer)
    before = isolation._Snapshot(trainer.ref_model)
    with isolation.teacher_only_execution(evaluator, private, shared_trainer=trainer) as receipt:
        proof = diagnostic.prepare_ppo_teacher(evaluator, original_ppo=True)
        assert proof["strict_mean_parameters_and_rms_equality"] is True
        assert private.max_noise_std == 1.0
        assert hasattr(private, "_original_ppo_std_policy")
        action, manipulation = diagnostic.raven_ppo_step(evaluator, obs)
        assert action["actions"].shape == (2, 26) and manipulation.shape == (2, 11)
        assert not action["actions"].requires_grad
    assert trainer.ref_model.max_noise_std == .001
    assert not hasattr(trainer.ref_model, "_original_ppo_std_policy")
    before.assert_equal(trainer.ref_model)
    assert isolation.validate_execution_receipt(receipt, vector_steps=1) == receipt


@pytest.mark.parametrize("path,counter", [
    (lambda evaluator: evaluator.policy_model(torch.zeros(1, 3)), "student_forward_calls"),
    (lambda evaluator: evaluator.policy_model.forward(torch.zeros(1, 3)), "student_forward_calls"),
    (lambda evaluator: evaluator.unwrapped_model.forward(torch.zeros(1, 3)), "student_forward_calls"),
    (lambda evaluator: evaluator.policy_step(), "distill_policy_step_calls"),
    (lambda evaluator: evaluator._image_dagger_relabel({}), "renderer_calls"),
    (lambda evaluator: evaluator._videogen_relabel(), "renderer_calls"),
])
def test_forbidden_paths_raise_before_actual_forward_and_restore(path, counter):
    trainer = owner()
    private, evaluator = isolation.build_isolated_teacher(trainer), facade(trainer)
    before = isolation._Snapshot(trainer.ref_model)
    with pytest.raises(RuntimeError, match="teacher-only execution forbids"):
        with isolation.teacher_only_execution(evaluator, private, shared_trainer=trainer) as receipt:
            path(evaluator)
    assert receipt[counter] == 1
    assert "forward" not in vars(trainer.policy_model)
    assert evaluator.ref_model is trainer.ref_model
    before.assert_equal(trainer.ref_model)


def test_swallowed_forbidden_call_cannot_produce_success_receipt():
    trainer = owner()
    with pytest.raises(RuntimeError, match="forbidden call was caught"):
        with isolation.teacher_only_execution(facade(trainer), isolation.build_isolated_teacher(trainer),
                                              shared_trainer=trainer) as receipt:
            try:
                trainer.policy_model.forward(torch.zeros(1, 3))
            except RuntimeError:
                pass
    assert receipt["student_forward_calls"] == 1


@pytest.mark.parametrize("name", isolation._RENDER_BINDINGS)
def test_renderer_or_observation_binding_rejected_before_execution(name):
    trainer = owner()
    evaluator = facade(trainer)
    setattr(evaluator, name, object())
    with pytest.raises(RuntimeError, match="rejects renderer/observation bindings"):
        with isolation.teacher_only_execution(evaluator, isolation.build_isolated_teacher(trainer),
                                              shared_trainer=trainer):
            pytest.fail("must reject before entering")
    assert "forward" not in vars(trainer.policy_model)


@pytest.mark.parametrize("change", [
    lambda t: setattr(t.ref_model, "max_noise_std", 1.0),
    lambda t: setattr(t.ref_model, "_original_ppo_std_policy", {}),
    lambda t: setattr(t.ref_model, "std", torch.nn.Parameter(t.ref_model.std.detach().clone())),
    lambda t: t.ref_model.std.data.add_(1),  # .data bypasses Tensor._version: byte check must catch it.
    lambda t: t.ref_model.obs_dict_buffer.update(obs=torch.zeros(1)),
    lambda t: setattr(t, "ref_model", copy.deepcopy(t.ref_model)),
])
def test_actual_shared_mutations_fail_not_just_boolean_attestation(change):
    trainer = owner()
    with pytest.raises(RuntimeError, match="shared teacher"):
        with isolation.teacher_only_execution(facade(trainer), isolation.build_isolated_teacher(trainer),
                                              shared_trainer=trainer) as receipt:
            change(trainer)
    assert receipt["shared_teacher_unchanged"] is False
    assert "forward" not in vars(trainer.policy_model)


def test_preexisting_forward_override_restored_exactly_after_exception():
    trainer = owner()
    original = lambda value: value + 10
    trainer.policy_model.forward = original
    evaluator = facade(trainer)
    with pytest.raises(ValueError, match="workload failed"):
        with isolation.teacher_only_execution(evaluator, isolation.build_isolated_teacher(trainer),
                                              shared_trainer=trainer) as receipt:
            raise ValueError("workload failed")
    assert trainer.policy_model.forward is original
    assert evaluator.ref_model is trainer.ref_model
    assert receipt["shared_teacher_unchanged"] is True
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", [
    lambda receipt: receipt.update(student_forward_calls=True),
    lambda receipt: receipt.update(renderer_calls=1),
    lambda receipt: receipt.update(distill_policy_step_calls=1),
    lambda receipt: receipt.update(teacher_rollout_calls=0),
    lambda receipt: receipt.update(shared_teacher_unchanged=False),
    lambda receipt: receipt.update(private_teacher_used=False),
    lambda receipt: receipt.update(extra="unbound"),
])
def test_execution_receipt_is_strict_and_requires_actual_rollouts(change):
    trainer = owner()
    private, evaluator = isolation.build_isolated_teacher(trainer), facade(trainer)
    with isolation.teacher_only_execution(evaluator, private, shared_trainer=trainer) as receipt:
        private.rollout(torch.zeros(1, 3))
    assert isolation.validate_execution_receipt(receipt, vector_steps=1) == receipt
    change(receipt)
    with pytest.raises(ValueError, match="invalid isolated PPO"):
        isolation.validate_execution_receipt(receipt, vector_steps=1)


def test_empty_context_cannot_claim_actual_private_teacher_use():
    trainer = owner()
    with isolation.teacher_only_execution(facade(trainer), isolation.build_isolated_teacher(trainer),
                                          shared_trainer=trainer) as receipt:
        pass
    assert receipt["private_teacher_used"] is False
    with pytest.raises(ValueError):
        isolation.validate_execution_receipt(receipt, vector_steps=0)


def test_receipt_import_and_validation_do_not_import_torch_or_numpy():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from isolated_ppo_teacher import validate_execution_receipt
receipt = {'schema': 'roborender.isolated-ppo-teacher-execution.v1',
           'student_forward_calls': 0, 'renderer_calls': 0,
           'distill_policy_step_calls': 0, 'teacher_rollout_calls': 1,
           'shared_teacher_unchanged': True, 'private_teacher_used': True}
assert validate_execution_receipt(receipt, vector_steps=1) == receipt
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code,
                    str(ACTOR_ROOT)], check=True, capture_output=True, text=True)
