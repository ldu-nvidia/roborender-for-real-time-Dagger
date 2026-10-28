"""Execute native adapter boundary methods without initializing Isaac.

These test state isolation and failure paths, not native scene qualification.
"""
from test.deployment.source_paths import ACTOR_ROOT
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

SOURCE = (ACTOR_ROOT / "resident_isaac_trainer.py")


def methods(*names, **globals_):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResidentRavenBackend")
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(globals_)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


@pytest.mark.parametrize('failure', [None, 'incomplete', 'optimizer', 'pause', 'kit', 'execute'])
def test_mode_boundary_only_returns_idle_after_valid_terminal_evidence(failure):
    from dagger_rtx.phase_modes import SCHEMA
    events = []
    sim = SimpleNamespace(pause=lambda: events.append('pause'),
                          is_playing=lambda: failure == 'pause')
    receipt = dict(complete=failure != 'incomplete', purpose='test',
                   optimizer_steps=1 if failure == 'optimizer' else 0)
    def execute(*args):
        events.append('execute')
        if failure == 'execute':
            raise RuntimeError('native fault')
        return receipt
    def kit():
        events.append('kit')
        if failure == 'kit':
            raise RuntimeError('Kit changed')
    backend = SimpleNamespace(validate=lambda *_: events.append('validate'), run=execute,
        native_device='cuda:0', kit_identity=123, _kit_unchanged=kit,
        trainer=SimpleNamespace(env=SimpleNamespace(simulator=SimpleNamespace(sim=sim))))
    scope = methods('_run_phase', torch=SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda device: events.append(('synchronize', device)))))
    call = lambda: scope['_run_phase'](backend,
        dict(schema=SCHEMA, mode='STUDENT_TEST', operation='evaluate_scaled_rendered_panel',
             parameters={'arm': 'bc'}), lambda event, **kw: events.append(event))
    if failure:
        with pytest.raises(RuntimeError):
            call()
        assert 'mode_idle' not in events
        assert 'pause' in events
    else:
        result = call()
        assert result['physics_idle'] is True and result['compute_released'] is False
        assert result['kit_identity'] == 123 and result['native_initializations'] == 1
        assert events == ['validate', 'mode_preparing', 'execute', 'mode_finalizing',
                          ('synchronize', 'cuda:0'), 'pause', 'kit', 'mode_idle']


def test_unbound_bc_mode_is_rejected_before_any_native_action():
    backend = SimpleNamespace()
    with pytest.raises(ValueError, match='bound resident BC adapter'):
        methods('validate')['validate'](backend, 'collect_bc_continuous', {})


@pytest.mark.parametrize('continuous,handoff,accepted', [
    (False, True, False), (True, False, False), (True, True, True),
])
def test_dagger_selector_requires_continuity_and_curriculum(continuous, handoff, accepted):
    from dagger_rtx.phase_modes import SCHEMA
    from dagger_rtx.production_segment import CONTINUOUS_SCHEMA
    calls = []
    backend = SimpleNamespace(validate=lambda *args: calls.append(args),
        _production_segment_inputs=lambda _: (None, {
            'schema': CONTINUOUS_SCHEMA if continuous else 'old-two-window-protocol'}, None))
    payload = dict(schema=SCHEMA, mode='DAGGER_TRAIN', operation='collect_production_segment',
                   parameters={'training_curriculum': {}} if handoff else {})
    call = lambda: methods('validate')['validate'](backend, 'run_phase', payload)
    if accepted:
        call()
    else:
        with pytest.raises(ValueError, match='continuous episodes'):
            call()
    assert len(calls) == 1 and calls[0][0] == 'collect_production_segment'


@pytest.mark.parametrize("failure", ["exception", "incomplete", "kit_identity"])
def test_phase_failure_pauses_without_reset_or_hiding_original_failure(failure):
    events = []
    original = RuntimeError("original native failure")
    def operation(*_):
        if failure == "exception":
            raise original
        return {"complete": failure != "incomplete"}
    def kit():
        if failure == "kit_identity":
            raise original
    backend = SimpleNamespace(profile="explicit_evaluation", scene_initializations=2,
        trainer=SimpleNamespace(env=SimpleNamespace(simulator=SimpleNamespace(
            sim=SimpleNamespace(pause=lambda: events.append("pause"))))),
        _run_guarded=operation, _kit_unchanged=kit, active_output="retain-failure-evidence")
    report = lambda phase, **fields: events.append(phase)
    with pytest.raises(RuntimeError) as caught:
        methods("run")["run"](backend, "evaluate_explicit_panel", {}, report)
    if failure != "incomplete":
        assert caught.value is original
    assert events == ["phase_transition_start", "pause"]
    assert backend.active_output == "retain-failure-evidence"


def test_pause_failure_preserves_primary_failure():
    original = RuntimeError("native failed")
    def fail(*_):
        raise original
    def pause():
        raise ValueError("pause unavailable")
    events = []
    backend = SimpleNamespace(profile="training", scene_initializations=1,
        trainer=SimpleNamespace(env=SimpleNamespace(simulator=SimpleNamespace(sim=SimpleNamespace(pause=pause)))),
        _run_guarded=fail)
    with pytest.raises(RuntimeError) as caught:
        methods("run")["run"](backend, "collect_round", {}, lambda phase, **kw: events.append(phase))
    assert caught.value is original
    assert events == ["phase_transition_start", "phase_failure_pause_failed"]


def test_success_boundary_clears_previous_progress_and_attests_same_kit():
    events = []
    receipt = {"complete": True, "optimizer_steps": 0}
    backend = SimpleNamespace(profile="explicit_evaluation", scene_initializations=3,
        _run_guarded=lambda *_: receipt, _kit_unchanged=lambda: events.append("kit"),
        active_output="old", active_arm="bc")
    assert methods("run")["run"](backend, "evaluate_explicit_panel", {},
        lambda phase, **kw: events.append(phase)) is receipt
    assert events == ["phase_transition_start", "kit", "phase_transition_complete"]
    assert backend.active_output is None and backend.active_arm is None


@pytest.mark.parametrize("profile,assignment", [
    ("typo", None), ("explicit_evaluation", None),
    ("explicit_evaluation", {"purpose": "training"}), ("training", {"purpose": "test"})])
def test_wrong_profile_or_split_rejected_before_scene_release(profile, assignment):
    backend = SimpleNamespace(_kit_unchanged=lambda: None)
    # No release/reuse dependencies are supplied: reaching either is a failure.
    with pytest.raises(ValueError):
        methods("_switch_scene")["_switch_scene"](backend, profile,
            seed=42, panel=None, report=lambda *_a, **_k: None, assignment=assignment)


def test_teacher_diagnostic_contamination_cannot_enter_collection():
    check = methods("_require_collection_teacher")["_require_collection_teacher"]
    backend = SimpleNamespace(trainer=SimpleNamespace(ref_model=SimpleNamespace()))
    check(backend)
    backend.trainer.ref_model._original_ppo_std_policy = {"original_max_noise_std": 1.0}
    with pytest.raises(RuntimeError, match="fresh pinned owner"):
        check(backend)


def test_training_preparation_restores_configuration_counters_and_episode_choices():
    events = []
    base = {"init_at_random_ep_len": True, "raven_exact_raytraced_augmentation": True}
    env = SimpleNamespace(num_envs=2, reset_count=123,
        reset_dataset=SimpleNamespace(resample=lambda: events.append("resample")))
    trainer = SimpleNamespace(env=env, config={"init_at_random_ep_len": False},
        cur_reward_sum=torch.ones(1), cur_episode_length=torch.ones(2), state=SimpleNamespace(),
        vg_seed=42, _id_anchors="test", _id_prompts="test", _id_use_raytraced=torch.ones(1))
    backend = SimpleNamespace(trainer=trainer, base_algo_config=base, phase_resets=2,
        cold_phase_state=SimpleNamespace(restore=lambda _: events.append("cold")),
        _require_collection_teacher=lambda: events.append("teacher"))
    methods("_prepare_training", torch=torch, copy=copy, np=np,
        seeded=lambda seed: events.append(("seed", seed)))["_prepare_training"](backend, 99)
    assert events == ["teacher", "cold", ("seed", 99), "resample"]
    assert trainer.config == base and trainer.config is not base
    assert trainer.cur_reward_sum.shape == (2,) and not trainer.cur_reward_sum.any()
    assert not trainer.cur_episode_length.any()
    assert trainer.state.cur_reward_sum is trainer.cur_reward_sum
    assert trainer._id_anchors is None and trainer._id_prompts is None and trainer._id_use_raytraced is None
    assert env.reset_count == 0 and env._resident_boundary_reset is True and backend.phase_resets == 3
    trainer.config["init_at_random_ep_len"] = False
    assert base["init_at_random_ep_len"] is True


def test_phase_environment_restored_even_on_error(monkeypatch):
    tree = ast.parse(SOURCE.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "phase_environment")
    scope = {"os": os, "contextmanager": contextmanager}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), scope)
    monkeypatch.setenv("DIAGNOSTIC_ARM", "prior")
    monkeypatch.delenv("DIAGNOSTIC_SEED", raising=False)
    with pytest.raises(RuntimeError):
        with scope["phase_environment"]({"DIAGNOSTIC_ARM": "bc", "DIAGNOSTIC_SEED": 17}):
            assert os.environ["DIAGNOSTIC_SEED"] == "17"
            raise RuntimeError("phase failed")
    assert os.environ["DIAGNOSTIC_ARM"] == "prior" and "DIAGNOSTIC_SEED" not in os.environ


def test_training_test_training_scene_roundtrip_restores_profile_and_batch(monkeypatch):
    events, created = [], []
    base = SimpleNamespace(num_envs=2,
        simulator=SimpleNamespace(config=SimpleNamespace(scene=SimpleNamespace(num_envs=2))),
        domain_rand=SimpleNamespace(image_augmentation=SimpleNamespace(enabled=True)))
    def environment(cfg, device):
        result = SimpleNamespace(config=cfg, num_envs=cfg.num_envs, simulator=SimpleNamespace())
        created.append(result)
        return result
    def configure(cfg, assignment, **kwargs):
        assert kwargs.get('motion_inventory') is None and kwargs.get('lane_offset',0)==0
        cfg.num_envs = len(assignment["lanes"])
        cfg.simulator.config.scene.num_envs = cfg.num_envs
    @contextmanager
    def binding(cfg, assignment, scenario, loader, **kwargs):
        assert kwargs.get('motion_inventory') is None and kwargs.get('lane_offset',0)==0
        events.append(("binding", assignment["purpose"]))
        yield {"assignment_sha256": assignment["purpose"]}
    monkeypatch.setitem(sys.modules, "native_motion_assignment", SimpleNamespace(
        configure_motion_environment=configure, explicit_scene_motion_binding=binding))
    monkeypatch.setitem(sys.modules, "gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt",
        SimpleNamespace(MotionLoaderWSPPT=object))
    namespace = methods("_switch_scene", "_switch_scene_admitted", hashlib=hashlib,
        canonical_json=lambda value: json.dumps(value, sort_keys=True).encode(),
        time=SimpleNamespace(monotonic=lambda: 0), copy=copy,
        gc=SimpleNamespace(collect=lambda: None), compact_host_heap=lambda _: None,
        OmegaConf=SimpleNamespace(set_struct=lambda *_: None),
        reset_rollout_for_scene=lambda model: events.append(("history-reset", model)),
        release_native_scene=lambda env, **_: {"released": env.num_envs},
        PanelResetEnvironment=environment, ResidentTrainingEnvironment=environment,
        ColdPhaseState=lambda env: ("cold", env), scenario=SimpleNamespace(),
        _training_table_metadata=object())
    backend = SimpleNamespace(profile="training", scene_seed=42, scene_initializations=1,
        trainer=SimpleNamespace(env=environment(copy.deepcopy(base), "cuda:0"),
            policy_model="student", ref_model="teacher"),
        _kit_unchanged=lambda: events.append("same-kit"),
        _require_collection_teacher=lambda: None, kit=SimpleNamespace(update=lambda: None),
        native_device="cuda:0", base_env_config=base)
    namespace["seeded"] = lambda seed: events.append(("seed", seed))
    backend._switch_scene_admitted = lambda *a, **kw: namespace["_switch_scene_admitted"](backend, *a, **kw)
    switch = namespace["_switch_scene"]
    test = {"purpose": "test", "lanes": [0]}
    train = {"purpose": "training", "lanes": [0, 1]}
    switch(backend, "explicit_evaluation", seed=17, panel=None, report=lambda *a, **kw: None, assignment=test)
    assert backend.trainer.env.num_envs == 1
    assert backend.trainer.env.config.domain_rand.image_augmentation.enabled is False
    switch(backend, "training", seed=23, panel=None, report=lambda *a, **kw: None, assignment=train)
    assert backend.trainer.env.num_envs == 2
    assert backend.trainer.env.config.domain_rand.image_augmentation.enabled is True
    assert backend.trainer.env.config.simulator._target_ == "resident_isaac_trainer.IsaacSim"
    assert backend.profile == "training" and backend.scene_seed == 23
    assert backend.scene_initializations == 3 and len(created) == 3
    assert base.domain_rand.image_augmentation.enabled is True
    assert events.count(("history-reset", "student")) == 2
    assert events.count(("history-reset", "teacher")) == 2
    assert ("seed", 17) in events and ("seed", 23) in events
    previous = backend.trainer.env
    switch(backend, "training", seed=23, panel=None, report=lambda *a, **kw: None, assignment=train)
    assert backend.trainer.env is previous and backend.scene_initializations == 3
