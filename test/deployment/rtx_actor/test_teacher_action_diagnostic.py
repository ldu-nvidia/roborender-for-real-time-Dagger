"""CPU control-path and receipt tests; no Kit, GPU launch or allocation APIs."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

HERE = ACTOR_ROOT
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import teacher_action_diagnostic as diagnostic
from explicit_motion_panel import (
    aggregate_partitions, canonical, completed_progress, make_cohort_receipt,
    make_partition, run_panel, run_partition, summarize, validate_result,
)
from test.deployment.rtx_actor.test_explicit_motion_panel import panel, result


class FakeTeacher(torch.nn.Module):
    is_recurrent = False
    max_rollout_history = 1

    def __init__(self, input_key="teacher_obs"):
        super().__init__()
        self.input_key = input_key
        self.linear = torch.nn.Linear(226, 11)
        self.std = torch.nn.Parameter(torch.ones(11), requires_grad=False)
        self.clamp_noise_std = True
        self.max_noise_std = .001
        self.algo_config = {"max_noise_std": .001}
        self.running_mean_std = torch.nn.Module()
        for name, value in (("running_mean", torch.arange(226) / 100),
                            ("running_var", torch.ones(226)), ("count", torch.tensor(2.))):
            self.running_mean_std.register_buffer(name, value)
        self.running_mean_std.per_channel = True
        self.running_mean_std.norm_only = False
        self.running_mean_std.epsilon = 1e-5
        self.eval()
        self.calls = []

    def rollout(self, *, obs_dict):
        assert not torch.is_grad_enabled()
        self.calls.append("rollout")
        x = obs_dict[self.input_key]
        rms = self.running_mean_std
        normalized = torch.clamp((x - rms.running_mean) / torch.sqrt(rms.running_var + 1e-5), -5, 5)
        self.distribution = torch.distributions.Normal(self.linear(normalized), torch.ones(x.shape[0], 11))
        return {"actions": self.distribution.sample(), "action_mean": self.action_mean}

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    def act_inference(self, **kwargs):
        self.calls.append("act_inference")
        return torch.full((2, 11), .2)


def homie(values, calls, label):
    def forward(obs):
        assert not torch.is_grad_enabled()
        calls.append((label, obs.clone()))
        mean = torch.full((obs.shape[0], 15), values)
        return {"actions": mean + 1000, "action_mean": mean}
    return forward


def setup():
    calls = []
    trainer = SimpleNamespace(ref_model=FakeTeacher(), policy_model=object(),
        unwrapped_model=SimpleNamespace(homie_switch_threshold=.5,
            homie_walk_model=homie(1., calls, "walk"),
            homie_stand_model=homie(2., calls, "stand")),
        policy_step=lambda *_a, **_k: pytest.fail("PPO mode cannot invoke distill/student"))
    obs = {"teacher_obs": torch.arange(452, dtype=torch.float32).reshape(2, 226) / 100,
           "actor_obs": torch.zeros(2, 113), "homie_obs": torch.zeros(2, 516)}
    obs["homie_obs"][0, -86:-83] = torch.tensor([.3, .4, 0.])
    obs["homie_obs"][1, -86:-83] = torch.tensor([.6, 0., 0.])
    return trainer, obs, calls


def original_ppo_step():
    source = REPO / "pipeline/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer_homie_api.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == diagnostic.PPO_SOURCE_SHA256
    tree = ast.parse(source.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
        and any(isinstance(item, ast.Assign) and ast.unparse(item.value) == "policy_model.rollout(obs_dict=obs_dict)"
                for item in node.body))
    begin = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and ast.unparse(node.value) == "policy_model.rollout(obs_dict=obs_dict)")
    end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
               and any(ast.unparse(target) == "actor_state['actions']" for target in node.targets))
    wrapper = ast.parse("def original(self, policy_model, homie_walk_model, homie_stand_model, obs_dict):\n"
                        "    actor_state = {}\n    return actor_state\n").body[0]
    wrapper.body[1:1] = loop.body[begin:end + 1]
    namespace = {"torch": torch, "init_actor_critic_dict": {"num_one_step_obs": 86}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["original"]


@pytest.mark.parametrize("input_key", ["teacher_obs", "actor_obs"])
def test_ppo_mode_matches_pinned_source_outputs_rng_and_homie_call_order(input_key):
    trainer, obs, calls = setup()
    trainer.ref_model.input_key = input_key
    reference = deepcopy(trainer.ref_model)
    reference.input_key = "actor_obs"
    reference_obs = dict(obs, actor_obs=obs["teacher_obs"])
    untouched = {key: value.clone() for key, value in obs.items()}
    torch.manual_seed(182)
    with torch.no_grad():
        expected = original_ppo_step()(trainer.unwrapped_model, reference,
            trainer.unwrapped_model.homie_walk_model, trainer.unwrapped_model.homie_stand_model,
            reference_obs)
    expected_rng = torch.get_rng_state().clone()
    calls.clear()
    torch.manual_seed(182)
    actual, raw = diagnostic.diagnostic_step(trainer, obs, None, arm="teacher", mode="raven_ppo")
    assert set(actual) == {"actions"} and "gt_actions" not in actual
    assert torch.equal(actual["actions"], expected["actions"])
    assert torch.equal(raw, actual["actions"][:, :11])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert trainer.ref_model.calls == ["rollout"]
    assert [row[0] for row in calls] == ["walk", "stand"]
    assert all(torch.equal(row[1], obs["homie_obs"]) for row in calls)
    assert actual["actions"][0, 11:].eq(2).all()  # draw==threshold selects stand
    assert actual["actions"][1, 11:].eq(1).all()
    assert all(torch.equal(obs[key], untouched[key]) for key in obs)


@pytest.mark.parametrize("arm", ["teacher", "bc", "dagger"])
def test_default_distill_retains_existing_policy_step_and_keys(arm):
    trainer, obs, _ = setup()
    teacher = torch.cat([torch.ones(2, 11), torch.ones(2, 15)], dim=1)
    student = torch.cat([torch.full((2, 11), 2.), torch.ones(2, 15)], dim=1)
    old = {"gt_actions": teacher, "action_mean": student, "gt_obj_pos": object()}
    dones, seen = object(), []
    def policy_step(model, actual_obs, *, cur_dones):
        seen.append((model, actual_obs, cur_dones))
        return old
    trainer.policy_step = policy_step
    actual, raw = diagnostic.diagnostic_step(trainer, obs, dones, arm=arm)
    assert actual is old and raw is None
    assert seen == [(trainer.policy_model, obs, dones)]
    chosen = teacher if arm == "teacher" else student
    assert actual["actions"] is chosen and actual["action_mean"] is chosen
    assert trainer.ref_model.calls == []


def test_distill_numeric_hook_copies_raw_action_without_second_forward():
    trainer, obs, _ = setup()
    original = trainer.ref_model.act_inference
    def policy_step(*_args, **_kwargs):
        raw = trainer.ref_model.act_inference(obs_dict=obs)
        corrected = torch.cat([raw, torch.zeros(2, 15)], dim=1)
        corrected[:, :3] = -1
        return {"gt_actions": corrected, "action_mean": corrected.clone()}
    trainer.policy_step = policy_step
    actual, raw = diagnostic.diagnostic_step(trainer, obs, None, arm="teacher", capture_raw=True)
    assert trainer.ref_model.calls == ["act_inference"]
    assert torch.all(raw == .2) and torch.all(actual["actions"][:, :3] == -1)
    assert trainer.ref_model.act_inference == original


def test_original_delta_step_proves_full26_control_and_feedback_difference():
    source = REPO / "pipeline/GR00T-VisualSim2Real/gr00t/rl/envs/base_task/delta_action_base.py"
    original = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    step = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    step.decorator_list = []
    wrapper = ast.parse("class DeltaActionBase(Parent):\n    pass\n").body[0]
    wrapper.body = [step]
    class Parent:
        def step(self, actor_state):
            return actor_state["actions"].clone()
    scope = {"Parent": Parent, "torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), "exec"), scope)
    owners = [scope["DeltaActionBase"]() for _ in range(2)]
    for env in owners:
        env.config = {"zero_vel": True, "zero_finger": False}
        env._delta_action_indices = torch.arange(11)
        env._delta_action_scale = .1
        env._delta_actions = torch.zeros(2, 11)
        env._last_delta_actions = torch.zeros(2, 11)
    raw = torch.arange(52, dtype=torch.float32).reshape(2, 26) / 100
    for stage in (0, 1, 2, 3, 4):
        for env in owners:
            env.stage_buf = torch.full((2,), stage)
        ppo = owners[0].step({"actions": raw.clone()})
        corrected = raw.clone()
        if 1 <= stage < 4:
            corrected[:, :3] = -10 * owners[1]._delta_actions[:, :3]
        distill = owners[1].step({"actions": corrected, "gt_actions": corrected.clone()})
        assert ppo.shape == distill.shape == (2, 26)
        assert torch.equal(ppo[:, 3:], distill[:, 3:])
        if stage == 1:
            assert not torch.equal(owners[0]._last_delta_actions, owners[1]._last_delta_actions)
            assert torch.count_nonzero(ppo[:, :3]) == 0
            assert torch.count_nonzero(owners[0]._delta_actions[:, :3]) > 0
        if stage == 4:
            assert not torch.equal(ppo[:, :3], distill[:, :3])
    assert diagnostic.control_contract("raven_ppo")["teacher_velocity_feedback_slice"] == [44, 47]


def ppo_result(spec):
    value = result(spec)
    value.update(arm="teacher", teacher_action_mode="raven_ppo",
        teacher_control=diagnostic.control_contract("raven_ppo"),
        teacher_parameter_attestation={"schema": "roborender.teacher-parameter-attestation.v2",
            "checkpoint_sha256": diagnostic.TEACHER_SHA256,
            "strict_parameter_and_rms_equality": True, "teacher_input_key": "teacher_obs",
            "strict_mean_parameters_and_rms_equality": True,
            "rms_updates_enabled": False, "teacher_history_steps": 1, "teacher_is_recurrent": False,
            "std_attestation": {"inherited_clamp_noise_std": True,
                "inherited_max_noise_std": .001, "original_clamp_noise_std": True,
                "original_max_noise_std": 1.0,
                "binding_scope": "teacher_instance_only_before_first_ppo_forward",
                "trainer_and_student_config_modified": False,
                "checkpoint_std": [1.] * 11, "runtime_std": [1.] * 11,
                "runtime_state": "checkpoint_raw",
                "allowed_runtime_states": ["checkpoint_raw", "original_ppo_max_clamp"],
                "optimizer_or_rms_updates_allowed": False},
            "homie_parameter_attestations": {
                role: {"checkpoint_sha256": sha, "strict_parameter_equality": True, "updates_enabled": False}
                for role, sha in (("walk", diagnostic.WALK_SHA256), ("stand", diagnostic.STAND_SHA256))}})
    return value


def test_mode_is_bound_to_summary_and_cannot_reuse_distill_receipts():
    p = panel()
    legacy = result(p["cohorts"][0])
    legacy["arm"] = "teacher"
    receipt = make_cohort_receipt(p, 0, legacy, checkpoint_sha256="b" * 64, arm="teacher")
    assert "teacher_action_mode" not in receipt  # old receipt schema/bytes retained
    with pytest.raises(ValueError):
        completed_progress(p, [receipt], checkpoint_sha256="b" * 64,
                           arm="teacher", teacher_action_mode="raven_ppo")
    rows = [ppo_result(spec) for spec in p["cohorts"]]
    summary = summarize(p, rows)
    assert summary["action_mode"] == "raven_ppo_privileged_teacher_plus_homie_raw"
    with pytest.raises(ValueError):
        validate_result(p, summary, "b" * 64, allowed_arms=("teacher",))
    assert validate_result(p, summary, "b" * 64, allowed_arms=("teacher",),
                           teacher_action_mode="raven_ppo") == summary
    with pytest.raises(ValueError, match="mode"):
        summarize(p, [legacy, rows[1]])
    old = summarize(p, [result(spec) for spec in p["cohorts"]])
    assert "teacher_action_mode" not in old and old["action_mode"] == p["action_mode"]


def test_ppomode_cohort_resume_is_same_mode_only_and_keeps_success_criteria():
    p, executed = panel(), []
    first = make_cohort_receipt(p, 0, ppo_result(p["cohorts"][0]),
        checkpoint_sha256="b" * 64, arm="teacher", teacher_action_mode="raven_ppo")
    assert first["teacher_action_mode"] == "raven_ppo"
    actual = run_panel(p, execute_cohort=lambda spec, index: (executed.append(index), ppo_result(spec))[1],
        completed_receipts=[first], expected_checkpoint_sha256="b" * 64, expected_arm="teacher",
        teacher_action_mode="raven_ppo")
    assert executed == [1] and actual["successes"] == 2 and actual["success_rate"] == .5
    assert actual["goal_accounting_version"] == "roborender.raven.pre_reset_goal.v1"
    with pytest.raises(ValueError):
        run_panel(p, execute_cohort=lambda *_: pytest.fail("cannot reuse PPO as distill"),
            completed_receipts=[first], expected_checkpoint_sha256="b" * 64, expected_arm="teacher")
    parts = [run_partition(p, make_partition(p, [i]), execute_cohort=lambda spec, _: ppo_result(spec),
                checkpoint_sha256="b" * 64, arm="teacher", teacher_action_mode="raven_ppo")
             for i in range(2)]
    assert aggregate_partitions(p, parts, checkpoint_sha256="b" * 64, arm="teacher",
                                teacher_action_mode="raven_ppo") == actual


@pytest.mark.parametrize("bad", [None, "ppo", 1, {}, "raven-ppo"])
def test_unknown_mode_rejected(bad):
    with pytest.raises(ValueError):
        diagnostic.validate_mode(bad)


@pytest.mark.parametrize("arm", ["bc", "dagger"])
def test_ppo_mode_cannot_silently_replace_student_behavior(arm):
    with pytest.raises(ValueError):
        diagnostic.validate_mode("raven_ppo", arm)


@pytest.mark.parametrize("value", [-1, 25, True, .5, "24"])
def test_trace_bound_is_explicit_integer(value):
    with pytest.raises(ValueError):
        diagnostic.validate_trace_steps(value)


def test_native_trace_hook_is_before_autoreset_and_has_no_rgb_dump():
    tree = ast.parse((SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "observed_termination")
    events = []
    namespace = {"original": lambda: events.append("native-termination"), "captured": {},
        "terminal_snapshot": lambda env: {"terminal_qvel": env.qvel}, "numeric_active": True,
        "numeric_after": {}, "numeric_state": lambda env: {"qvel": env.qvel},
        "self": SimpleNamespace(env=SimpleNamespace(qvel=8.))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pre-reset-observer", "exec"), namespace)
    namespace["observed_termination"]()
    namespace["self"].env.qvel = 0.  # automatic reset occurs only after callback
    assert namespace["numeric_after"] == {"qvel": 8.}
    assert namespace["captured"] == {"terminal_qvel": 8.}
    assert events == ["native-termination"]
    source = (SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text()
    assert '"post_step_before_auto_reset": numeric_after' in source
    assert '"teacher_obs": cpu(obs["teacher_obs"])' in source
    assert '"homie_obs": cpu(obs["homie_obs"])' in source


def test_parameter_attestation_includes_rms_and_forbids_wrong_live_weights(tmp_path, monkeypatch):
    trainer, _, _ = setup()
    path = tmp_path / "teacher.pt"
    torch.save({"policy_state_dict": trainer.ref_model.state_dict()}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(diagnostic, "TEACHER_SHA256", sha)
    paths = {}
    for role in ("walk", "stand"):
        model = torch.nn.Linear(516, 15).eval().requires_grad_(False)
        setattr(trainer.unwrapped_model, "homie_" + role + "_model", model)
        cp = tmp_path / (role + ".pt")
        torch.save({"model_state_dict": {**model.state_dict(), "critic.unused": torch.zeros(3)}}, cp)
        monkeypatch.setattr(diagnostic, role.upper() + "_SHA256", hashlib.sha256(cp.read_bytes()).hexdigest())
        paths["homie_" + role + "_model_path"] = str(cp)
    trainer.config = SimpleNamespace(network_load_dict=SimpleNamespace(
        teacher_actor=SimpleNamespace(path=str(path))), **paths)
    proof = diagnostic.prepare_ppo_teacher(trainer)
    assert proof["strict_parameter_and_rms_equality"] and proof["rms_updates_enabled"] is False
    assert all(row["strict_parameter_equality"] for row in proof["homie_parameter_attestations"].values())
    assert diagnostic.prepare_ppo_teacher(trainer) == proof
    # A buffer/parameter in-place mutation invalidates the metadata cache.
    with torch.no_grad():
        trainer.ref_model.running_mean_std.running_mean.add_(1)
    with pytest.raises(RuntimeError, match="RMS differs"):
        diagnostic.prepare_ppo_teacher(trainer)
    trainer2, _, _ = setup()
    trainer2.config = trainer.config
    trainer2.unwrapped_model = trainer.unwrapped_model
    with pytest.raises(RuntimeError, match="RMS differs"):
        diagnostic.prepare_ppo_teacher(trainer2)
    assert not torch.cuda.is_initialized()


def _checkpoint_attested_trainer(tmp_path, monkeypatch):
    trainer, obs, _ = setup()
    with torch.no_grad():
        trainer.ref_model.std.copy_(torch.tensor([1.0004, .7, 1.0003] + [.8] * 8))
    path = tmp_path / "teacher.pt"
    torch.save({"policy_state_dict": trainer.ref_model.state_dict()}, path)
    monkeypatch.setattr(diagnostic, "TEACHER_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    paths = {}
    for role in ("walk", "stand"):
        model = torch.nn.Linear(516, 15).eval().requires_grad_(False)
        setattr(trainer.unwrapped_model, "homie_" + role + "_model", model)
        cp = tmp_path / (role + ".pt")
        torch.save({"model_state_dict": model.state_dict()}, cp)
        monkeypatch.setattr(diagnostic, role.upper() + "_SHA256", hashlib.sha256(cp.read_bytes()).hexdigest())
        paths["homie_" + role + "_model_path"] = str(cp)
    trainer.config = SimpleNamespace(network_load_dict=SimpleNamespace(
        teacher_actor=SimpleNamespace(path=str(path))), max_noise_std=.001, **paths)
    return trainer, obs


def test_original_std_binding_then_real_actor_rollout_and_second_cohort_attest(tmp_path, monkeypatch):
    from types import MethodType
    trainer, obs = _checkpoint_attested_trainer(tmp_path, monkeypatch)
    teacher = trainer.ref_model
    raw = teacher.std.detach().clone()
    before = diagnostic.prepare_ppo_teacher(trainer, original_ppo=True)
    assert before["std_attestation"]["runtime_state"] == "checkpoint_raw"
    assert teacher.max_noise_std == 1.0
    assert teacher.algo_config["max_noise_std"] == trainer.config.max_noise_std == .001
    # Execute the ACTUAL pinned Actor.rollout/update_distribution AST on CPU,
    # not an imitation of its legal in-place clamp.
    source = REPO / "pipeline/GR00T-VisualSim2Real/gr00t/rl/trl/modules/actor_critic_modules.py"
    actor = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Actor")
    namespace = {"torch": torch, "Normal": torch.distributions.Normal, "TensorDict": dict}
    for name in ("rollout", "update_distribution"):
        node = deepcopy(next(n for n in actor.body if isinstance(n, ast.FunctionDef) and n.name == name))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
        setattr(teacher, name, MethodType(namespace[name], teacher))
    def buffered(value, *_a):
        teacher.obs_dict_buffer = {key: tensor.unsqueeze(1) for key, tensor in value.items()}
        return None
    teacher._update_obs_buffer = buffered
    teacher.forward = lambda obs_dict, **_kw: teacher.linear(obs_dict[teacher.input_key])
    teacher.steps = 0
    with torch.no_grad():
        teacher.rollout(obs)
        teacher.rollout(obs)
    assert torch.equal(teacher.std, raw.clamp(max=1.0))
    after = diagnostic.prepare_ppo_teacher(trainer, original_ppo=True)
    assert after["strict_parameter_and_rms_equality"] is False  # truthful raw-byte difference
    assert after["strict_mean_parameters_and_rms_equality"] is True
    assert after["std_attestation"]["runtime_state"] == "original_ppo_max_clamp"
    assert diagnostic.validate_parameter_attestation(after) == after
    assert diagnostic.prepare_ppo_teacher(trainer, original_ppo=True) == after
    with torch.no_grad():
        teacher.linear.weight.add_(1)
    with pytest.raises(RuntimeError, match="RMS differs.*linear.weight"):
        diagnostic.prepare_ppo_teacher(trainer, original_ppo=True)
    assert not torch.cuda.is_initialized()


def test_default_distill_std_attestation_stays_strict_and_does_not_bind(tmp_path, monkeypatch):
    trainer, _ = _checkpoint_attested_trainer(tmp_path, monkeypatch)
    raw = diagnostic.prepare_ppo_teacher(trainer)
    assert "std_attestation" not in raw
    assert trainer.ref_model.max_noise_std == .001
    assert not hasattr(trainer.ref_model, "_original_ppo_std_policy")
    with torch.no_grad():
        trainer.ref_model.std.clamp_(max=1.0)
    with pytest.raises(RuntimeError, match="RMS differs.*std"):
        diagnostic.prepare_ppo_teacher(trainer)


def test_original_std_rejects_arbitrary_drift_and_control_changes(tmp_path, monkeypatch):
    trainer, _ = _checkpoint_attested_trainer(tmp_path, monkeypatch)
    diagnostic.prepare_ppo_teacher(trainer, original_ppo=True)
    with torch.no_grad():
        trainer.ref_model.std.fill_(.001)
    with pytest.raises(RuntimeError, match="std differs"):
        diagnostic.prepare_ppo_teacher(trainer, original_ppo=True)
    trainer.ref_model.max_noise_std = .5
    with pytest.raises(RuntimeError, match="noise control changed"):
        diagnostic.prepare_ppo_teacher(trainer, original_ppo=True)


def test_numeric_readback_maps_native_finger_targets_and_copies_before_reset(monkeypatch):
    native_targets = torch.tensor([[11., 22., 33.]])
    sim = SimpleNamespace(
        _robot=SimpleNamespace(data=SimpleNamespace(joint_pos_target=SimpleNamespace(torch=native_targets))),
        dof_ids=[2, 0, 1], dof_pos=torch.tensor([[3., 1., 2.]]),
        dof_vel=torch.tensor([[.3, .1, .2]]), _rigid_body_pos=torch.ones(1, 1, 3),
        get_task_root_state=lambda _: torch.zeros(1, 13))
    env = SimpleNamespace(simulator=sim, _last_delta_actions=torch.zeros(1, 11),
        _delta_actions=torch.ones(1, 11), right_arm_dof_indices=[0, 1], right_palm_body_idx=0,
        _right_hand_dof_idx=[2, 0], actions_after_delay=torch.ones(1, 3),
        config=SimpleNamespace(task=SimpleNamespace(front_object="f", hold_object="h", tray_object="t")),
        get_hold_object_hand_distance=lambda: torch.zeros(1),
        get_hold_object_tray_distance=lambda: torch.ones(1))
    actual = diagnostic.numeric_state(env)
    assert actual["right_finger_joint_targets"] == [[22., 33.]]
    assert actual["right_finger_joint_positions"] == [[2., 3.]]
    assert actual["right_finger_cumulative_command"] == [[1.]]
    native_targets.zero_()
    env._delta_actions.zero_()
    assert actual["right_finger_joint_targets"] == [[22., 33.]]
    assert actual["right_finger_cumulative_command"] == [[1.]]
    import native_physics_readback
    env.num_envs = 1
    def unsupported_native_getter(_):
        raise AttributeError("backend drive getter absent")
    monkeypatch.setattr(native_physics_readback, "drive_readback", unsupported_native_getter)
    missing = diagnostic.numeric_state(env)
    assert missing["effective_drive_readback"] == {
        "status": "unavailable", "error_type": "AttributeError", "read_only": True}
    assert missing["right_finger_joint_targets"] == [[0., 0.]]


def test_teacher_attestation_is_rechecked_after_last_native_cohort():
    source = (SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text()
    assert source.index("teacher_attestation_at_start = teacher_attestation") > source.index(
        'raise RuntimeError("student state changed during inference-only diagnostic")')
    assert '"teacher_parameter_attestation_at_start": teacher_attestation_at_start' in source
    assert '"effective-physics-readback.json"' in source


def test_effective_native_sorted_teacher_feature_layout_is_44_not_45():
    import yaml
    reference = REPO / "pipeline/GR00T-VisualSim2Real/gr00t/rl"
    obs = yaml.safe_load((reference / "config/obs/loco_manip/obs_walk_stand_place_grasp_turn_homie.yaml").read_text())["obs"]
    dimensions = {key: value for item in obs["obs_dims"] for key, value in item.items()}
    substitutions = {"${robot.actions_dim}": 31, "${robot.dof_obs_size}": 43}
    names = obs["obs_dict"]["actor_obs"]
    offset, layout, values = 0, {}, {}
    for index, name in enumerate(sorted(names)):
        size = substitutions.get(dimensions[name], dimensions[name])
        assert type(size) is int
        layout[name] = (offset, offset + size)
        values[name] = torch.full((2, size), float(index))
        offset += size
    assert len(names) == 19 and offset == 226
    assert layout["delta_actions"] == (44, 55)
    assert layout["dof_pos"] == (55, 98) and layout["dof_vel"] == (98, 141)
    source = reference / "envs/legged_base_task/legged_robot_base.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_post_config_observation_callback")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    owner = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_dict={"teacher_obs": names})),
                            obs_buf_dict_raw={"teacher_obs": values})
    scope["_post_config_observation_callback"](owner)
    assert torch.equal(owner.obs_buf_dict["teacher_obs"][:, 44:55], values["delta_actions"])


@pytest.mark.parametrize("mutation", [
    lambda r: r["teacher_control"].update(env_step_keys=["actions", "gt_actions"]),
    lambda r: r["teacher_control"].update(rollout_sample_rng_consumed=False),
    lambda r: r["teacher_parameter_attestation"]["homie_parameter_attestations"]["walk"].update(
        checkpoint_sha256="f" * 64),
    lambda r: r["teacher_parameter_attestation"].update(rms_updates_enabled=True),
])
def test_ppo_receipt_cannot_claim_wrong_control_or_model_provenance(mutation):
    p = panel()
    rows = [ppo_result(spec) for spec in p["cohorts"]]
    mutation(rows[0])
    with pytest.raises(ValueError):
        summarize(p, rows)


def test_native_resume_v2_must_bind_mode_before_using_a_completed_cohort(tmp_path):
    tree = ast.parse((HERE / "resident_isaac_trainer.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_explicit_completed_receipts")
    def file_identity(path, expected, maximum):
        assert path.stat().st_size < maximum
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    scope = {"json": json, "file_identity": file_identity}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "native-mode-resume", "exec"), scope)
    p = panel()
    receipt = make_cohort_receipt(p, 0, ppo_result(p["cohorts"][0]), checkpoint_sha256="b" * 64,
                                  arm="teacher", teacher_action_mode="raven_ppo")
    from explicit_motion_panel import panel_sha256
    doc = {"schema": "roborender.explicit-motion-success-resume.v2", "panel_sha256": panel_sha256(p),
        "checkpoint_sha256": "b" * 64, "arm": "teacher", "teacher_action_mode": "raven_ppo",
        "cohort_receipts": [receipt]}
    path = tmp_path / "resume.json"
    backend = SimpleNamespace(artifact=lambda _: path)
    def check(document, mode="raven_ppo"):
        path.write_bytes(canonical(document))
        payload = {"completed_receipts_file": "resume.json",
            "completed_receipts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_sha256": "b" * 64, "arm": "teacher", "teacher_action_mode": mode}
        return scope["_explicit_completed_receipts"](backend, payload, p)
    assert check(doc) == [receipt]
    with pytest.raises(ValueError):
        check(doc, "distill")
    with pytest.raises(ValueError):
        check({key: value for key, value in {**doc,
            "schema": "roborender.explicit-motion-success-resume.v1"}.items() if key != "teacher_action_mode"})
