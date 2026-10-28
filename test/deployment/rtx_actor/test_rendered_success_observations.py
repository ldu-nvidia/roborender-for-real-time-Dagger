"""CPU-only observation-ablation tests: no Isaac, network, CUDA or allocation."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import hashlib
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import rendered_success_observations as module
RAVEN = HERE.parents[1] / "GR00T-VisualSim2Real" / "gr00t" / "rl"


def native_method(relative, class_name, method_name):
    """Execute the exact native CPU method, without importing Isaac/Kit."""
    path = RAVEN / relative
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    assert not method.decorator_list
    namespace = {"np": np, "torch": torch, "F": torch.nn.functional,
                 "os": os, "time": time, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


def fixture(tmp_path, count=2):
    anchors = []
    for index in range(3):
        path = tmp_path / f"anchor-{index}.npz"
        np.savez_compressed(path, frames=np.array([0, 40]), prompt=np.array(f"scene {index}"),
            **{"gen__" + view: np.full((2, 240, 416, 3), index + lane, np.uint8)
               for lane, view in enumerate(module.VIEWS)})
        anchors.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    calls = {"capture": 0, "normalize": 0, "depth": []}
    views = {view: {"depth": torch.full((count, 4, 6, 1), float(i + 1)),
                    "robot_rgb": torch.full((count, 4, 6, 3), i + 10, dtype=torch.uint8)}
             for i, view in enumerate(module.VIEWS)}
    def capture():
        calls["capture"] += 1
        return views
    def depth(value, view):
        calls["depth"].append((view, value.copy()))
        return np.full(value.shape, int(value[0, 0, 0, 0]), dtype=np.uint8)
    def resize(value, hw):
        return np.broadcast_to(value[:, :, :1, :1], (*value.shape[:2], *hw, *value.shape[4:])).copy()
    def normalize(rgb):
        calls["normalize"] += 1
        assert rgb.min() >= 0 and rgb.max() <= 1
        return (rgb - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    cameras = SimpleNamespace(policy_views=list(module.VIEWS), camera_resolutions=[108, 192], image_mean=[.485, .456, .406],
                              image_std=[.229, .224, .225])
    trainer = SimpleNamespace(env=SimpleNamespace(num_envs=count,
        config=SimpleNamespace(simulator=SimpleNamespace(config=SimpleNamespace(cameras=cameras)),
            domain_rand=SimpleNamespace(image_augmentation=SimpleNamespace(enabled=False))),
        simulator=SimpleNamespace(get_conditioning_views=capture)),
        vg_control_res=[240, 416], vg_depth_mode="h5_v4", camera_resolution=[324, 192, 3],
        _h5v4_control=depth, _resize_seq=resize, _normalize_like_env=normalize)
    clients = []
    class Client:
        def __init__(self, lane):
            self.lane, self.calls, self.closed = lane, [], False
        def render(self, **request):
            self.calls.append(deepcopy(request))
            return SimpleNamespace(views={view: np.full((240, 416, 3), 30 + i * 60 + self.lane, np.uint8)
                                         for i, view in enumerate(module.VIEWS)})
        def close(self): self.closed = True
    def factory(lane):
        client = Client(lane)
        clients.append(client)
        return client
    identity = {"renderer": "Image3F", "adapter_sha256": module.ADAPTER_SHA256,
                "engine_sha256": "f" * 64, "steps": 5, "cfg": 1.,
                "teacache_threshold": .3, "seed": 42, "server_rng_isolated": True}
    adapter = module.GeneratedSuccessObservations(trainer, client_factory=factory,
        anchor_files=anchors, anchor_seed=123, renderer_identity=identity)
    trainer._success_observation_transform = adapter
    obs = {"vision_obs": torch.zeros(count, 324, 192, 3), "actor_obs": torch.randn(count, 113),
           "teacher_obs": torch.randn(count, 226), "homie_obs": torch.randn(count, 12)}
    return trainer, adapter, obs, clients, calls, views


def test_context_free_and_explicit_teacher_observation_paths_remain_native(monkeypatch):
    monkeypatch.delenv("DIAGNOSTIC_ARM", raising=False)
    obs = {"vision_obs": torch.randn(2, 3)}
    trainer = SimpleNamespace()
    assert module.apply_observation_transform(trainer, obs, torch.zeros(2), 0) is obs
    assert module.observation_receipt(trainer, 1) == {"observation_mode": "raytraced_3view"}
    monkeypatch.setenv("DIAGNOSTIC_ARM", "teacher")
    assert module.apply_observation_transform(trainer, obs, torch.zeros(2), 0) is obs


@pytest.mark.parametrize("arm", ["bc", "dagger"])
def test_student_missing_callback_fails_before_action_and_result(monkeypatch, arm):
    monkeypatch.setenv("DIAGNOSTIC_ARM", arm)
    trainer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="no raytraced fallback"):
        module.apply_observation_transform(trainer, {"vision_obs": torch.zeros(2, 3)}, torch.zeros(2), 0)
    with pytest.raises(RuntimeError, match="no raytraced fallback"):
        module.observation_receipt(trainer, 1)


def test_student_callback_without_admitted_config_cannot_render(tmp_path, monkeypatch):
    trainer, adapter, obs, clients, *_ = fixture(tmp_path)
    monkeypatch.setenv("DIAGNOSTIC_ARM", "bc")
    with pytest.raises(RuntimeError, match="admitted configuration"):
        module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    assert adapter.steps == 0 and all(not client.calls for client in clients)
    adapter.close()


def test_generic_callback_binding_cannot_be_reused_for_another_cohort(tmp_path):
    trainer, old_adapter, obs, *_ = fixture(tmp_path)
    old_adapter.close()
    binding = {"schema": "roborender.rendered-student-success-cohort-binding.v1",
        "checkpoint_sha256": "a" * 64, "panel_sha256": "b" * 64,
        "config_sha256": "c" * 64, "cohort_index": 0, "motion_assignment_sha256": "d" * 64}
    adapter = module.GeneratedSuccessObservations(trainer,
        client_factory=lambda lane: SimpleNamespace(render=lambda **kw: None, close=lambda: None),
        anchor_files=old_adapter.anchor_files, anchor_seed=123,
        renderer_identity=old_adapter.renderer_identity, evaluation_binding=binding)
    trainer._success_observation_transform = adapter
    trainer._require_rendered_success_observations = True
    trainer._rendered_success_evaluation_binding = deepcopy(binding)
    assert module.require_observation_transform(trainer) is adapter
    # Returned receipts are copies, but a caller supplying an adapter for a
    # different cohort cannot reach a render request or policy action.
    adapter.receipt()["evaluation_binding"]["cohort_index"] = 100
    assert adapter.receipt()["evaluation_binding"] == binding
    trainer._rendered_success_evaluation_binding["cohort_index"] = 1
    with pytest.raises(RuntimeError, match="checkpoint/panel/cohort"):
        module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    assert adapter.steps == 0
    adapter.close()


def test_direct_student_cohort_requires_callback_before_scene_or_model_mutation(tmp_path):
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExplicitMotionSuccessEvaluator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_cohort")
    scope = {"DISTILL": "distill"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "student-cohort-admission", "exec"), scope)
    output = tmp_path / "not-created"
    with pytest.raises(RuntimeError, match="no raytraced fallback"):
        scope["evaluate_cohort"](SimpleNamespace(), {}, checkpoint_path="unused.pt",
            checkpoint_sha256="a" * 64, output=output, arm="bc")
    assert not output.exists()


def test_camera_guard_matches_real_native_hwc_assignment(tmp_path):
    trainer, adapter, *_ = fixture(tmp_path)
    adapter.close()
    for name in ("ppo_trainer.py", "distill_trainer_obj_pred_homie_api.py"):
        path = RAVEN / "trl/trainer" / name
        tree = ast.parse(path.read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "camera_resolution" for target in node.targets)
            and isinstance(node.value, ast.List) and len(node.value.elts) == 3)
        context = {"self": trainer, "_cams": trainer.env.config.simulator.config.cameras,
                   "_n_views": len(module.VIEWS), "num_channels": 3}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), context)
        assert trainer.camera_resolution == [324, 192, 3]
    for wrong_shape in ([324, 192], [108, 192, 3], [324, 192, 1]):
        trainer.camera_resolution = wrong_shape
        with pytest.raises(ValueError, match="geometry"):
            module.GeneratedSuccessObservations(trainer, client_factory=lambda lane: None,
                anchor_files=adapter.anchor_files, anchor_seed=123, renderer_identity=adapter.renderer_identity)


def test_actual_native_controls_helpers_framed_api_and_relabel_parity(tmp_path, monkeypatch):
    # opencv-python-headless is the native resize dependency; this test must be
    # included with it installed when qualifying a new source release.
    pytest.importorskip("cv2")
    import socket
    import yaml
    from pipeline.dagger.services.render_student import _Image3FLoopbackClient

    trainer, fixture_adapter, obs, *_ = fixture(tmp_path)
    fixture_adapter.close()
    def no_network(*args, **kwargs):
        pytest.fail("native CPU API qualification must not open a network connection")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.delenv("IMAGE_DAGGER_DUMP_DIR", raising=False)
    config = yaml.safe_load((RAVEN / "config/exp/loco_manip/wsdpt_image_dagger_3view.yaml").read_text())
    vg = config["algo"]["config"]["videogen"]
    trainer.vg_h5v4_per_view = vg["depth_h5v4_per_view"]
    trainer.vg_h5v4_default = {"raw_clip": [0., 10.], "inv_clip": [1., 20.]}
    trainer.vg_invalid_fill = "far"
    for name in ("_h5v4_control", "_resize_seq", "_normalize_like_env"):
        setattr(trainer, name, MethodType(native_method("trl/trainer/distill_trainer_videogen.py",
            "TRLDistillTrainerVideoGen", name), trainer))
    # Execute the actual simulator getters on CPU sensor buffers. Only the
    # render driver is replaced; no physics or simulator step is available.
    sensors = {}
    height, width = config["simulator"]["config"]["cameras"]["depth_hi_resolution"]
    for index, view in enumerate(module.VIEWS):
        depth = torch.linspace(.05, 3., 2 * height * width).reshape(2, height, width, 1)
        depth[0, 0, 0, 0] = float("inf")
        rgb = torch.full((2, height, width, 4), 40 + index, dtype=torch.uint8)
        segment = torch.ones((2, height, width, 1), dtype=torch.int64)
        segment[:, :, width // 2:, :] = 0
        sensors["cond_cam_" + view] = SimpleNamespace(data=SimpleNamespace(
            output={"depth": depth, "rgb": rgb, "semantic_segmentation": segment},
            info={"semantic_segmentation": {"idToLabels": {"1": {"class": "robot"}}}}))
    captures = []
    simulator = SimpleNamespace(conditioning_cameras=dict.fromkeys(module.VIEWS),
        scene=SimpleNamespace(sensors=sensors), refresh_conditioning_render=lambda: captures.append(1))
    for name in ("_cond_cam_depth", "_cond_cam_robot_rgb", "get_conditioning_views"):
        setattr(simulator, name, MethodType(native_method("simulator/isaacsim/isaacsim.py", "IsaacSim", name), simulator))
    trainer.env.simulator = simulator
    pixel_pattern = np.indices((240, 416, 3)).sum(axis=0)
    generated = {view: ((pixel_pattern + 30 + i * 60) % 256).astype(np.uint8)
                 for i, view in enumerate(module.VIEWS)}
    exchanges = []
    def factory(lane):
        client = _Image3FLoopbackClient("127.0.0.1", 18550)
        def exchange(payload, *, max_response_bytes):
            request = pickle.loads(payload)
            assert request["cmd"] == "render_frame" and request["seed"] == 42
            assert set(request) == {"cmd", "prompt", "seed", "depth", "mask", "prev"}
            exchanges.append(request)
            return {"ok": True, "views": {view: client._encode(value) for view, value in generated.items()},
                    "runtime_ms": 1., "total_ms": 2.}
        client._exchange = exchange
        return client
    adapter = module.GeneratedSuccessObservations(trainer, client_factory=factory,
        anchor_files=fixture_adapter.anchor_files, anchor_seed=123,
        renderer_identity=fixture_adapter.renderer_identity)
    trainer._success_observation_transform = adapter
    rendered = module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    assert len(captures) == 1 and len(exchanges) == 2
    for request in exchanges:
        for view in module.VIEWS:
            assert request["depth"][view]["shape"] == (240, 416)
            assert request["mask"][view]["shape"] == (240, 416, 3)
            assert request["depth"][view]["dtype"] == request["mask"][view]["dtype"] == "uint8"
    # Independent execution of Raven's entire online observation replacement
    # confirms the geometry and single normalization, not just a mock recipe.
    trainer._id_anchors = [anchor for anchor, _ in adapter.anchors]
    trainer._id_prompts = [prompt for _, prompt in adapter.anchors]
    trainer._id_use_raytraced = {}
    trainer._id_env_clients = [SimpleNamespace(render_frame=lambda *args, **kwargs: generated) for _ in range(2)]
    trainer._policy_views = trainer.vg_view_names = list(module.VIEWS)
    trainer._id_gen_ms = []
    trainer.vg_seed = 42
    reference = native_method("trl/trainer/distill_trainer_videogen.py", "TRLDistillTrainerVideoGen",
                              "_image_dagger_relabel")(trainer, obs, torch.zeros(2))
    assert torch.equal(rendered["vision_obs"], reference["vision_obs"])
    assert all(rendered[name] is obs[name] for name in obs if name != "vision_obs")
    adapter.close()


def test_real_framed_client_unconnected_cleanup_is_idempotent_and_offline(tmp_path, monkeypatch):
    from pipeline.dagger.services.render_student import _Image3FLoopbackClient
    import socket

    trainer, fixture_adapter, *_ = fixture(tmp_path)
    fixture_adapter.close()
    def no_network(*args, **kwargs):
        pytest.fail("unconnected renderer-client construction/cleanup must not open a socket")
    monkeypatch.setattr(socket, "create_connection", no_network)
    adapter = module.GeneratedSuccessObservations(trainer,
        client_factory=lambda lane: _Image3FLoopbackClient("127.0.0.1", 18550),
        anchor_files=fixture_adapter.anchor_files, anchor_seed=123,
        renderer_identity=fixture_adapter.renderer_identity)
    assert all(client._stream is None for client in adapter.clients)
    assert all(not hasattr(client, "close") for client in adapter.clients)
    adapter.close()
    adapter.close()
    assert adapter.closed and all(client._stream is None for client in adapter.clients)


@pytest.mark.parametrize("missing", ["render", "closer"])
def test_client_lifecycle_contract_is_checked_before_pool_creation(tmp_path, monkeypatch, missing):
    trainer, fixture_adapter, *_ = fixture(tmp_path)
    fixture_adapter.close()
    def no_pool(*args, **kwargs):
        pytest.fail("invalid client lifecycle must fail before creating a worker pool")
    monkeypatch.setattr(module, "ThreadPoolExecutor", no_pool)
    def factory(lane):
        if missing == "render":
            return SimpleNamespace(_close=lambda: None)
        return SimpleNamespace(render=lambda **kwargs: None, _close=None, close=None)
    with pytest.raises(ValueError, match="renderer client requires"):
        module.GeneratedSuccessObservations(trainer, client_factory=factory,
            anchor_files=fixture_adapter.anchor_files, anchor_seed=123,
            renderer_identity=fixture_adapter.renderer_identity)


def test_controls_current_three_views_and_only_vision_replaced_once(tmp_path):
    trainer, adapter, obs, clients, calls, views = fixture(tmp_path)
    original = obs["vision_obs"].clone()
    out = module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    assert calls["capture"] == calls["normalize"] == 1
    assert len(calls["depth"]) == 6
    assert all(out[key] is value for key, value in obs.items() if key != "vision_obs")
    assert torch.equal(obs["vision_obs"], original)
    assert [client.calls[0]["seed"] for client in clients] == [42, 42]
    for i, view in enumerate(module.VIEWS):
        assert clients[0].calls[0]["depth"][view][0, 0] == i + 1
        assert clients[1].calls[0]["mask"][view][0, 0, 0] == i + 10
        expected = (torch.tensor([30 + 60 * i] * 3) / 255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        assert torch.allclose(out["vision_obs"][0, i * 108 + 54, 96], expected)
    views["head_color"]["depth"].fill_(8)
    module.apply_observation_transform(trainer, obs, torch.zeros(2), 1)
    assert clients[0].calls[1]["depth"]["head_color"][0, 0] == 8
    receipt = module.observation_receipt(trainer, 2)
    assert receipt["rendered_observation_receipt"]["render_requests_completed"] == 4
    assert receipt["rendered_observation_receipt"]["offline_BC_teacher50_equivalent"] is False
    adapter.close()
    assert all(client.closed for client in clients)


def test_anchor_rng_isolated_fixed_per_episode_and_done_resamples(tmp_path):
    trainer, adapter, obs, clients, calls, views = fixture(tmp_path)
    random.seed(978)
    np.random.seed(978)
    torch.manual_seed(978)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    for step, done in enumerate((torch.zeros(2), torch.zeros(2), torch.tensor([1., 0.]))):
        module.apply_observation_transform(trainer, obs, done, step)
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1]) and after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert [(row["lane"], row["selected_at_vector_step"]) for row in adapter.anchor_receipts] == [(0, 0), (1, 0), (0, 2)]
    assert all(np.array_equal(clients[0].calls[0]["previous"][view], clients[0].calls[1]["previous"][view])
               for view in module.VIEWS)
    adapter.close()


@pytest.mark.parametrize("fault", ["missing_view", "missing_robot", "nonfinite_depth", "renderer_error", "wrong_rgb", "changed_anchor"])
def test_no_render_failure_falls_back_or_allows_retry(tmp_path, fault):
    trainer, adapter, obs, clients, calls, views = fixture(tmp_path)
    if fault == "missing_view": del views["hand_left"]
    elif fault == "missing_robot": views["head_color"]["robot_rgb"] = None
    elif fault == "nonfinite_depth": views["head_color"]["depth"].fill_(float("nan"))
    elif fault == "renderer_error":
        clients[0].render = lambda **kw: (_ for _ in ()).throw(RuntimeError("server unavailable"))
    elif fault == "wrong_rgb": clients[0].render = lambda **kw: SimpleNamespace(views={})
    elif fault == "changed_anchor":
        for row in adapter.anchor_files: Path(row["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError): module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    assert adapter.failed and adapter.steps == 0
    with pytest.raises(RuntimeError, match="retry/replay"):
        module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    with pytest.raises(RuntimeError): module.observation_receipt(trainer, 1)
    adapter.close()


@pytest.mark.parametrize("fault", ["actor_inplace", "actor_replace", "input_dict_mutation", "vision_inplace", "shape", "nan"])
def test_hook_rejects_changes_outside_fresh_compatible_vision(fault):
    obs = {"vision_obs": torch.zeros(2, 3), "actor_obs": torch.zeros(2, 4)}
    class Bad:
        def receipt(self): return {}
        def __call__(self, values, dones, step):
            out = {**values, "vision_obs": values["vision_obs"].clone()}
            if fault == "actor_inplace": values["actor_obs"].add_(1)
            if fault == "actor_replace": out["actor_obs"] = values["actor_obs"].clone()
            if fault == "input_dict_mutation": values["actor_obs"] = values["actor_obs"].clone()
            if fault == "vision_inplace": values["vision_obs"].add_(1)
            if fault == "shape": out["vision_obs"] = torch.zeros(2, 4)
            if fault == "nan": out["vision_obs"].fill_(float("nan"))
            return out
    with pytest.raises(RuntimeError):
        module.apply_observation_transform(SimpleNamespace(_success_observation_transform=Bad()), obs, torch.zeros(2), 0)


def test_receipt_cardinality_and_repeated_steps_rejected(tmp_path):
    trainer, adapter, obs, clients, calls, views = fixture(tmp_path)
    module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    with pytest.raises(RuntimeError): module.observation_receipt(trainer, 2)
    with pytest.raises(RuntimeError): module.apply_observation_transform(trainer, obs, torch.zeros(2), 0)
    adapter.close()


def test_native_hook_follows_initial_state_gate_and_precedes_policy_and_env_step():
    tree = ast.parse((SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text())
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    locate = lambda suffix: min(line for line, name in calls if name.endswith(suffix))
    assert locate("require_paired_states") < locate("apply_observation_transform") < locate("select_policy_actions") < locate("self.env.step")
    assert locate("observation_receipt") > locate("self.env.step")
