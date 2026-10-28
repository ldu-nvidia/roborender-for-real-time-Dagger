"""CPU method-level phase isolation; these are not native round-trip proof."""
from test.deployment.source_paths import PIPELINE_ROOT

import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SOURCE = PIPELINE_ROOT / "deployment/rtx_actor/resident_isaac_trainer.py"


def extracted(*names, **scope):
    tree = ast.parse(SOURCE.read_text())
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "ResidentRavenBackend")
    selected = [node for node in [*tree.body, *backend.body]
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


class Model:
    def __init__(self):
        self.rollout_initializations = 0
        self.observation_history = [object()]
        self.distribution = object()

    def init_rollout(self):
        self.rollout_initializations += 1
        self.observation_history = []

    def clear_rollout(self):
        raise AssertionError("a finished actor must not be cleared a second time")


def test_scene_reset_handles_already_finished_actor_without_double_clear():
    model = Model()
    del model.distribution  # Raven's completed window already deleted this.
    reset = extracted("reset_rollout_for_scene")["reset_rollout_for_scene"]
    reset(model)
    assert model.rollout_initializations == 1
    assert model.observation_history == [] and model.distribution is None


def test_equal_b16_train_test_train_test_never_reuses_prior_scene_state(monkeypatch):
    """Equal cardinality/seed is insufficient for reuse across phase profiles.

    The actual adapter methods run here, with the native environment constructor
    replaced by a small sentinel object. This proves replacement/ownership and
    model-buffer reset ordering, not Isaac's physical reset numerics.
    """
    events, environments = [], []
    base = SimpleNamespace(num_envs=16,
        simulator=SimpleNamespace(config=SimpleNamespace(scene=SimpleNamespace(num_envs=16))),
        domain_rand=SimpleNamespace(image_augmentation=SimpleNamespace(enabled=True)))

    def environment(cfg, device):
        env = SimpleNamespace(config=cfg, num_envs=cfg.num_envs, simulator=SimpleNamespace(),
                              histories=[], actions=[], goals=[], contacts=[], released=False)
        environments.append(env)
        return env

    def configure(cfg, assignment, *, motion_inventory=None, lane_offset=0):
        assert motion_inventory is None and lane_offset == 0
        cfg.num_envs = cfg.simulator.config.scene.num_envs = len(assignment["lanes"])

    @contextmanager
    def binding(cfg, assignment, scenario, loader, *, motion_inventory=None, lane_offset=0):
        assert motion_inventory is None and lane_offset == 0
        events.append(("assignment", assignment["purpose"]))
        yield {"assignment_sha256": assignment["purpose"]}

    @contextmanager
    def scene_lock(report):
        yield

    monkeypatch.setitem(sys.modules, "native_motion_assignment", SimpleNamespace(
        configure_motion_environment=configure, explicit_scene_motion_binding=binding))
    monkeypatch.setitem(sys.modules, "gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt",
                        SimpleNamespace(MotionLoaderWSPPT=object))
    monkeypatch.setitem(sys.modules, "scene_reconfiguration_lock",
                        SimpleNamespace(scene_reconfiguration_lock=scene_lock))
    models = [Model(), Model()]
    initial = environment(copy.deepcopy(base), "cpu")
    trainer = SimpleNamespace(env=initial, policy_model=models[0], ref_model=models[1])

    def release(env, **kwargs):
        assert trainer.env is None  # detach ownership before native teardown
        assert all(model.observation_history == [] and model.distribution is None
                   for model in models)
        env.released = True
        return {"released": True}

    scope = extracted("reset_rollout_for_scene", "_switch_scene", "_switch_scene_admitted",
        hashlib=hashlib, canonical_json=lambda value: json.dumps(value, sort_keys=True).encode(),
        time=SimpleNamespace(monotonic=lambda: 0), copy=copy,
        gc=SimpleNamespace(collect=lambda: None), compact_host_heap=lambda _: None,
        OmegaConf=SimpleNamespace(set_struct=lambda *_: None),
        release_native_scene=release, seeded=lambda seed: events.append(("seed", seed)),
        PanelResetEnvironment=environment, ResidentTrainingEnvironment=environment,
        ColdPhaseState=lambda env: ("new-cold-state", env), scenario=SimpleNamespace(),
        _training_table_metadata=object())
    kit = SimpleNamespace(update=lambda: None)
    backend = SimpleNamespace(profile="training", scene_seed=73, scene_initializations=1,
        trainer=trainer, kit=kit, native_device="cpu", base_env_config=base,
        _kit_unchanged=lambda: events.append("same-kit"), _require_collection_teacher=lambda: None)
    backend._switch_scene_admitted = lambda *a, **kw: scope["_switch_scene_admitted"](backend, *a, **kw)

    for profile, purpose in [("explicit_evaluation", "test"), ("training", "training"),
                             ("explicit_evaluation", "test")]:
        previous = trainer.env
        for name in ("histories", "actions", "goals", "contacts"):
            getattr(previous, name).append(object())
        for model in models:
            model.observation_history.append(object())
            model.distribution = object()
        scope["_switch_scene"](backend, profile, seed=73, panel=None,
            report=lambda *a, **kw: None, assignment={"purpose": purpose, "lanes": list(range(16))})
        assert previous.released and trainer.env is not previous
        assert trainer.env.num_envs == trainer.env.config.simulator.config.scene.num_envs == 16
        assert all(getattr(trainer.env, name) == []
                   for name in ("histories", "actions", "goals", "contacts"))
        assert trainer.env.config.domain_rand.image_augmentation.enabled is (purpose == "training")
        assert trainer.policy_model is models[0] and trainer.ref_model is models[1]
        assert backend.kit is kit

    assert len({id(env) for env in environments}) == backend.scene_initializations == 4
    assert [model.rollout_initializations for model in models] == [3, 3]
    assert events.count("same-kit") == 6
    assert [event for event in events if isinstance(event, tuple) and event[0] == "assignment"] == [
        ("assignment", "test"), ("assignment", "training"), ("assignment", "test")]
    assert base.num_envs == 16 and base.domain_rand.image_augmentation.enabled is True
