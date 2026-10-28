"""Execute pinned native reset AST with a recording CPU simulator (not PhysX)."""
from test.deployment.source_paths import ACTOR_ROOT, REPO_ROOT
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import initial_reset_contract as reset
import initial_robot_refresh_diagnostic as success
import materialize_isaac6 as port

ROOT = REPO_ROOT
BASE = "gr00t/rl/envs/base_task/loco_manip_base.py"
LEGGED = "gr00t/rl/envs/legged_base_task/legged_robot_base.py"
TASK = "gr00t/rl/envs/loco_manip/walk_stand_place_grasp_turn_homie.py"


def method(source, name):
    return deepcopy(next(n for n in ast.walk(ast.parse(source))
                         if isinstance(n, ast.FunctionDef) and n.name == name))


def function(node):
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<pinned-native-reset-probe>", "exec"), scope)
    return scope[node.name]


def source(relative=BASE):
    raw = (ROOT / port.SOURCE_PREFIX / relative).read_bytes()
    assert port.sha(raw) == port.PINS[relative]
    return raw.decode()


def environment(count, *, past=True):
    events = []
    env = SimpleNamespace(num_envs=count, device="cpu", use_past_obs=past,
        need_to_refresh_envs=torch.zeros(count, dtype=torch.bool),
        target_robot_root_states=torch.arange(count * 13).reshape(count, 13).float(),
        target_robot_dof_state=torch.zeros(count, 43, 2), init_root_states={"object": torch.ones(count, 13)},
        obs_buf_dict={"obs": torch.ones(count, 3)}, episode_length_buf=torch.zeros(count, dtype=torch.long),
        reset_buf=torch.zeros(count, dtype=torch.bool),
        config=SimpleNamespace(reset_from_dataset=SimpleNamespace(enable=True)))

    def reset_ids(ids):
        if len(ids):
            events.append(("reset_targets", ids.tolist()))
            env.need_to_refresh_envs[ids] = True
    env.reset_envs_idx = reset_ids
    env.simulator = SimpleNamespace(**{
        name: (lambda ids, *args, name=name: events.append((name, ids.tolist())))
        for name in ("set_actor_root_state_tensor", "set_dof_state_tensor",
                     "set_task_root_state_tensor", "set_task_visual_state_tensor")})
    for name in ("_refresh_sim_tensors", "_pre_compute_observations_callback", "_compute_observations",
                 "_post_compute_observations_callback", "_reset_past_obs_callback",
                 "_update_obs_buf_dict_with_past_obs", "_update_counters_each_step",
                 "_update_tasks_callback", "_check_termination", "_compute_reward"):
        setattr(env, name, lambda name=name: events.append((name, None)))
    return env, events


def post_prefix():
    raw = (ROOT / port.SOURCE_PREFIX / TASK).read_bytes()
    assert port.sha(raw) == "e1b5fd2262db184d3b766c5d8212916abd63a7aaeb59a8fe7577ab416391aab1"
    node = method(raw.decode(), "_post_physics_step")
    end = next(i for i, n in enumerate(node.body)
               if isinstance(n, ast.If) and "len(refresh_env_ids)" in ast.unparse(n.test))
    node.body = node.body[:end + 1]
    return function(node)


@pytest.mark.parametrize("count", [1, 2, 6, 16, 28, 32, 64])
@pytest.mark.parametrize("past", [False, True])
@pytest.mark.parametrize("relative", [BASE, LEGGED])
def test_actual_reset_ast_preserves_writes_observations_rng_and_ordinary_done(count, past, relative):
    old, old_events = environment(count, past=past)
    new, new_events = environment(count, past=past)
    original = function(method(source(relative), "reset_all"))
    corrected = function(method(port.transform(relative, source(relative)), "reset_all"))
    rng = torch.get_rng_state().clone()
    original(old)
    returned = corrected(new)
    assert old_events == new_events  # All native calls in exactly the same order.
    assert returned is new.obs_buf_dict
    assert torch.equal(old.target_robot_root_states, new.target_robot_root_states)
    assert torch.equal(old.target_robot_dof_state, new.target_robot_dof_state)
    assert torch.equal(old.init_root_states["object"], new.init_root_states["object"])
    assert torch.equal(torch.get_rng_state(), rng)
    assert old.need_to_refresh_envs.all() and not new.need_to_refresh_envs.any()
    receipt = reset.require_completed_initial_reset(new)
    assert receipt["generation"] == 1 and receipt["cleared_lanes"] == count
    old_events.clear(); new_events.clear()
    post = post_prefix()
    post(old); post(new)
    writes = lambda events: [row for row in events if row[0].startswith("set_")]
    assert [n for n, _ in writes(old_events)] == ["set_actor_root_state_tensor", "set_dof_state_tensor"]
    assert not writes(new_events)
    # A normal termination still has to write the new robot target on that lane.
    old_events.clear(); new_events.clear()
    old.reset_buf[0] = new.reset_buf[0] = True
    post(old); post(new)
    assert old_events == new_events
    assert writes(new_events) == [("set_actor_root_state_tensor", [0]), ("set_dof_state_tensor", [0])]
    assert not new.need_to_refresh_envs.any()


@pytest.mark.parametrize("relative", [BASE, LEGGED])
def test_only_reset_all_is_modified_and_body_only_wrapped(relative):
    original = ast.parse(source(relative))
    derived = ast.parse(port.transform(relative, source(relative)))
    node = next(n for n in ast.walk(derived) if isinstance(n, ast.FunctionDef) and n.name == "reset_all")
    assert isinstance(node.body[0], ast.ImportFrom) and node.body[0].module == "initial_reset_contract"
    assert ast.unparse(node.body[1]) == "begin_initial_reset(self)"
    assert ast.unparse(node.body[-2]) == "complete_initial_reset(self)"
    node.body = node.body[2:-2] + [node.body[-1]]
    assert ast.dump(original, include_attributes=False) == ast.dump(derived, include_attributes=False)


def test_native_task_inheritance_resolves_to_a_patched_reset_owner():
    # Build real class declarations without executing GPU/Isaac imports. This
    # catches an overlay aimed at an unused, similarly named base class.
    env_root = ROOT / port.SOURCE_PREFIX / 'gr00t/rl/envs'
    classes = {}
    for path in env_root.rglob('*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append((path, node))
    built = {}
    def build(name):
        if name in built:
            return built[name]
        if name in ('ABC', 'object') or name not in classes:
            return object
        assert len(classes[name]) == 1, name
        path, node = classes[name][0]
        bases = tuple(build(ast.unparse(base)) for base in node.bases) or (object,)
        namespace = {}
        if any(isinstance(m, ast.FunctionDef) and m.name == 'reset_all' for m in node.body):
            namespace['reset_all_owner'] = str(path.relative_to(ROOT / port.SOURCE_PREFIX))
        built[name] = type(name, bases, namespace)
        return built[name]
    owner = build('WalkStandPlaceGraspTurnHomie').reset_all_owner
    assert owner == LEGGED
    assert owner in port.PINS
    patched = ast.unparse(method(port.transform(owner, source(owner)), 'reset_all'))
    assert 'complete_initial_reset(self)' in patched


@pytest.mark.parametrize("bad", [torch.ones(2), torch.ones(3, dtype=torch.bool),
                                torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool)])
def test_invalid_flags_fail_without_clear_or_receipt(bad):
    env, _ = environment(2)
    reset.begin_initial_reset(env)
    env.need_to_refresh_envs = bad.clone()
    with pytest.raises(RuntimeError):
        reset.complete_initial_reset(env)
    assert torch.equal(env.need_to_refresh_envs, bad)
    assert env._rr_initial_reset_receipt is None
    with pytest.raises(RuntimeError):
        reset.require_completed_initial_reset(env)
    with pytest.raises(RuntimeError, match="incomplete"):
        reset.begin_initial_reset(env)


def test_failed_reset_invalidates_old_receipt_and_cannot_be_replayed():
    env, _ = environment(2)
    corrected = function(method(port.transform(BASE, source()), "reset_all"))
    corrected(env)
    def fail(*args):
        raise RuntimeError("simulator write failed")
    env.simulator.set_dof_state_tensor = fail
    with pytest.raises(RuntimeError, match="write failed"):
        corrected(env)
    with pytest.raises(RuntimeError, match="missing or inconsistent"):
        reset.require_completed_initial_reset(env)
    with pytest.raises(RuntimeError, match="incomplete"):
        corrected(env)


def test_no_completion_outside_reset_and_no_accepting_old_overlay():
    env, _ = environment(2)
    with pytest.raises(RuntimeError, match="outside"):
        reset.complete_initial_reset(env)
    with pytest.raises(RuntimeError, match="missing or inconsistent"):
        reset.require_completed_initial_reset(env)


@pytest.mark.parametrize("mode,arm,action", [(success.CLEAR_ONCE, "teacher", "raven_ppo"),
    (success.STUDENT_CLEAR_ONCE, "bc", "distill"), (success.STUDENT_CLEAR_ONCE, "dagger", "distill"),
    (success.SMOKE_STUDENT_CLEAR_ONCE, "bc", "distill")])
def test_success_receipt_identical_no_second_clear_and_repeated_cohorts(mode, arm, action):
    corrected = function(method(port.transform(BASE, source()), "reset_all"))
    env, _ = environment(2)
    legacy, _ = environment(2)
    legacy.need_to_refresh_envs.fill_(True)
    kwargs = dict(mode=mode, arm=arm, teacher_action_mode=action)
    expected = success.clear_already_applied_initial_refresh(legacy, **kwargs)
    for generation in range(1, 4):
        corrected(env)
        env._diagnostic_initial_refresh_clear_calls = 0
        proof = success.clear_already_applied_initial_refresh(env, **kwargs)
        assert proof == expected
        assert reset.require_completed_initial_reset(env)["generation"] == generation
        with pytest.raises(RuntimeError, match="outside"):
            reset.complete_initial_reset(env)
        # Even resetting the old evaluation counter cannot reuse an attestation.
        env._diagnostic_initial_refresh_clear_calls = 0
        with pytest.raises(RuntimeError, match="already applied"):
            success.clear_already_applied_initial_refresh(env, **kwargs)


def test_completed_receipt_is_copied_and_tampering_or_stale_generation_rejected():
    env, _ = environment(2)
    function(method(port.transform(BASE, source()), "reset_all"))(env)
    receipt = reset.require_completed_initial_reset(env)
    receipt["cleared_lanes"] = 0
    assert reset.require_completed_initial_reset(env)["cleared_lanes"] == 2
    env._rr_initial_reset_generation += 1
    with pytest.raises(RuntimeError, match="missing or inconsistent"):
        reset.require_completed_initial_reset(env)


def test_bc_and_dagger_entrypoint_guards_precede_first_step_and_randomization():
    native = (ROOT / "pipeline/dagger/rtx/code/src/dagger_rtx/native_entrypoints.py").read_text()
    init = ast.unparse(method(native, "initialize"))
    assert init.index("reset_all()") < init.index("require_completed_initial_reset(") < init.index("torch.randint_like(")
    bc = (ACTOR_ROOT / "isaac6_bc_production.py").read_text()
    step = ast.unparse(method(bc, "step"))
    assert step.index("require_completed_initial_reset(") < step.index("original_step(")
    assert "'initial_reset_completion': reset_receipt" in step
