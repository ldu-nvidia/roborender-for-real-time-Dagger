"""CPU control/launcher contracts. These are not native scene-switch evidence."""
from test.deployment.source_paths import ACTOR_ROOT
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ACTOR = ACTOR_ROOT
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR))
from run_resident_isaac import build_resident_command
from resident_isaac_client import ResidentClient, canonical
from qualify_native_isaac6 import IMAGE


def test_launcher_pins_one_gpu_and_keeps_batch_profiles(tmp_path):
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    cmd = build_resident_command(name="standing-isaac-leo-test-g0", runtime=tmp_path,
        overlay=tmp_path, source=ROOT, jobs=tmp_path, assets=tmp_path,
        token=tmp_path / "token", source_sha="a"*64, gpu=0)
    assert cmd[cmd.index("--network")+1] == "host"
    assert "device=GPU-4d58bc2b-43c8-6ff8-e7fd-2348575f0cff" in cmd
    assert "num_envs=6" in cmd
    assert "trainer._target_=resident_isaac_trainer.ResidentIsaacTrainer" in cmd
    assert "env._target_=resident_isaac_trainer.ResidentTrainingEnvironment" in cmd
    assert "simulator._target_=resident_isaac_trainer.IsaacSim" in cmd
    # Raven selects AppLauncher with this exact final-name check, before
    # importing the configured environment. Renaming it skips Kit entirely.
    target = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("simulator._target_="))
    assert target.rsplit(".", 1)[-1] == "IsaacSim"
    assert "--restart" not in cmd
    assert "RESIDENT_PORT=18640" in cmd
    assert "--no-healthcheck" in cmd
    assert IMAGE in cmd
    assert f"type=bind,src={ROOT},dst=/roborender-source,readonly" in cmd
    assert any(arg.startswith("PYTHONPATH=/roborender-source:/resident-source:") for arg in cmd)


def test_full_source_mount_resolves_hardened_renderer_client_without_isaac(tmp_path):
    # Resolve the same mounted package root outside the repository cwd; no
    # native simulator or renderer is instantiated by this CPU import.
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-c",
        "import sys; from pipeline.dagger.services.render_student import _Image3FLoopbackClient; "
        "assert _Image3FLoopbackClient.__module__ == 'pipeline.dagger.services.render_student'; "
        "assert not any(k.startswith(('omni', 'isaacsim')) for k in sys.modules)"],
        env=environment, cwd=tmp_path, check=True)


def test_native_adapter_has_no_application_restart_or_close():
    source = (ACTOR / "resident_isaac_trainer.py").read_text()
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "AppLauncher" not in calls
    assert not any(name.endswith(".close") for name in calls)
    assert "SimulationContext.clear_instance" in calls
    assert "simulator.sim.stop" in source
    assert '5 if profile == "evaluation" else 6' in source
    assert "run_pair(request, panel, output" in source
    assert "NativeDaggerRTXServiceTrainer.train(self.trainer)" in source


def native_scene_cleanup():
    tree = ast.parse((ACTOR / "resident_isaac_trainer.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "release_native_scene")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "native-scene-cleanup", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("stop_invalidates", [True, False])
def test_scene_cleanup_destroys_only_owned_products_before_stage_close(stop_invalidates):
    namespace = native_scene_cleanup()
    events = []
    class Camera:
        def __init__(self, path):
            self._render_data = SimpleNamespace(render_product_paths=[path])
        def _invalidate_initialize_callback(self, event):
            assert event is None
            events.append("detach")
            self._render_data = None
    first, second = Camera("/Render/owned-first"), Camera("/Render/owned-second")
    scene = SimpleNamespace(sensors={"head": first, "head-alias": first, "hand": second,
                                     "contact": object()})
    def stop():
        events.append("stop")
        if stop_invalidates:
            first._invalidate_initialize_callback(None)
            second._invalidate_initialize_callback(None)
    simulator = SimpleNamespace(scene=scene, sim=SimpleNamespace(stop=stop),
        scene_creation_callback=object(), event_manager=object(), viewport_camera_controller=object(),
        conditioning_cameras={"head": first}, ego_camera=first, depth_hi_camera=second)
    env = SimpleNamespace(simulator=simulator)
    namespace["SimulationContext"] = SimpleNamespace(clear_instance=lambda: events.append("clear-stage"))
    def acquire(path):
        events.append("acquire:"+path)
        return SimpleNamespace(path=path, destroy=lambda: events.append("destroy:"+path))
    result = namespace["release_native_scene"](env, flush_app=lambda: events.append("kit-update"),
                                               acquire_render_product=acquire)
    assert result == {"destroyed_render_products": 2, "camera_sensors": 3,
                      "kit_updates": 2, "environment_step_calls": 0}
    assert events[:2] == ["acquire:/Render/owned-first", "acquire:/Render/owned-second"]
    assert events.count("detach") == 2
    assert events[events.index("stop")+1:events.index("clear-stage")] == [
        "detach", "detach", "destroy:/Render/owned-first", "destroy:/Render/owned-second", "kit-update"]
    assert events[-1] == "kit-update"
    assert env.simulator is None and simulator.scene is None
    assert simulator.conditioning_cameras == {}
    assert simulator.ego_camera is None and simulator.depth_hi_camera is None
    assert simulator.scene_creation_callback is None
    assert not hasattr(simulator, "event_manager") and not hasattr(simulator, "viewport_camera_controller")


def test_scene_cleanup_without_initialized_cameras_keeps_kit_and_no_product_lookup():
    namespace = native_scene_cleanup()
    events = []
    simulator = SimpleNamespace(scene=SimpleNamespace(sensors={"contact": object()}),
                                sim=SimpleNamespace(stop=lambda: events.append("stop")))
    namespace["SimulationContext"] = SimpleNamespace(clear_instance=lambda: events.append("clear-stage"))
    result = namespace["release_native_scene"](SimpleNamespace(simulator=simulator),
        flush_app=lambda: events.append("kit-update"),
        acquire_render_product=lambda _: pytest.fail("no camera product exists"))
    assert result["destroyed_render_products"] == 0
    assert events == ["stop", "kit-update", "clear-stage", "kit-update"]


@pytest.mark.parametrize("path,returned_path", [("/World/robot", "/World/robot"),
                                               ("/Render/owned", "/Render/other")])
def test_scene_cleanup_rejects_unknown_product_before_native_mutation(path, returned_path):
    namespace = native_scene_cleanup()
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product_paths=[path]))
    env = SimpleNamespace(simulator=SimpleNamespace(scene=SimpleNamespace(sensors={"camera": camera}),
        sim=SimpleNamespace(stop=lambda: pytest.fail("must fail before stopping"))))
    namespace["SimulationContext"] = SimpleNamespace(clear_instance=lambda: pytest.fail("must not clear stage"))
    with pytest.raises(RuntimeError, match="owned render.product"):
        namespace["release_native_scene"](env, flush_app=lambda: pytest.fail("must not update Kit"),
            acquire_render_product=lambda _: SimpleNamespace(path=returned_path, destroy=lambda: None))


def test_scene_cleanup_failure_does_not_clear_or_replace_old_stage():
    namespace = native_scene_cleanup()
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product_paths=["/Render/owned"]))
    camera._invalidate_initialize_callback = lambda _: setattr(camera, "_render_data", None)
    env = SimpleNamespace(simulator=SimpleNamespace(scene=SimpleNamespace(sensors={"camera": camera}),
        sim=SimpleNamespace(stop=lambda: None)))
    namespace["SimulationContext"] = SimpleNamespace(clear_instance=lambda: pytest.fail("cleanup failed"))
    def failed_destroy():
        raise RuntimeError("native product destruction failed")
    with pytest.raises(RuntimeError, match="native product destruction failed"):
        namespace["release_native_scene"](env, flush_app=lambda: pytest.fail("cleanup failed"),
            acquire_render_product=lambda path: SimpleNamespace(path=path, destroy=failed_destroy))
    assert env.simulator is not None


@pytest.mark.parametrize("context", ["default", "", "named-context"])
def test_scene_cleanup_registry_lookup_never_creates_or_attaches_texture(context, monkeypatch):
    namespace = native_scene_cleanup()
    events = []
    owned = SimpleNamespace(path="/Render/owned", destroy=lambda: events.append("destroy-owned"),
        hydra_texture=SimpleNamespace(get_render_product_path=lambda: "/Render/owned"))
    other = SimpleNamespace(path="/Render/unrelated", destroy=lambda: pytest.fail("not scene-owned"),
        hydra_texture=SimpleNamespace(get_render_product_path=lambda: "/Render/unrelated"))
    registry = SimpleNamespace(_hydra_textures={context: [owned], "other-context": [other]},
        get=lambda *_a: pytest.fail("do not use an implicit context lookup"),
        add_texture=lambda **_k: pytest.fail("must never create a replacement texture"))
    monkeypatch.setitem(sys.modules, "omni.replicator.core.scripts.utils.viewport_manager",
                        SimpleNamespace(ViewportManager=lambda: SimpleNamespace(_hydra_textures=registry)))
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product_paths=["/Render/owned"]))
    camera._invalidate_initialize_callback = lambda _: setattr(camera, "_render_data", None)
    env = SimpleNamespace(simulator=SimpleNamespace(scene=SimpleNamespace(sensors={"head": camera}),
        sim=SimpleNamespace(stop=lambda: events.append("stop"))))
    namespace["SimulationContext"] = SimpleNamespace(clear_instance=lambda: events.append("clear-stage"))
    result = namespace["release_native_scene"](env, flush_app=lambda: events.append("kit-update"))
    assert result["destroyed_render_products"] == 1
    assert events == ["stop", "destroy-owned", "kit-update", "clear-stage", "kit-update"]


@pytest.mark.parametrize("handles", [0, 2])
def test_scene_cleanup_registry_missing_or_ambiguous_handle_is_nonmutating(handles, monkeypatch):
    namespace = native_scene_cleanup()
    texture = SimpleNamespace(path="/Render/owned", destroy=lambda: pytest.fail("ambiguous owner"),
        hydra_texture=SimpleNamespace(get_render_product_path=lambda: "/Render/owned"))
    registry = SimpleNamespace(_hydra_textures={"default": [texture]*handles},
                               add_texture=lambda **_k: pytest.fail("must not create"))
    monkeypatch.setitem(sys.modules, "omni.replicator.core.scripts.utils.viewport_manager",
                        SimpleNamespace(ViewportManager=lambda: SimpleNamespace(_hydra_textures=registry)))
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product_paths=["/Render/owned"]))
    env = SimpleNamespace(simulator=SimpleNamespace(scene=SimpleNamespace(sensors={"head": camera}),
        sim=SimpleNamespace(stop=lambda: pytest.fail("must not stop before lookup succeeds"))))
    with pytest.raises(RuntimeError, match="unique existing registry handle"):
        namespace["release_native_scene"](env, flush_app=lambda: pytest.fail("must not update Kit"))


def test_scene_lookup_inspection_attests_actual_handles_with_zero_native_mutation(monkeypatch):
    namespace = native_scene_cleanup()
    owned = SimpleNamespace(path="/Render/owned", destroy=lambda: pytest.fail("inspection cannot destroy"),
        hydra_texture=SimpleNamespace(get_render_product_path=lambda: "/Render/owned"))
    registry = SimpleNamespace(_hydra_textures={"default": [owned]})
    monkeypatch.setitem(sys.modules, "omni.replicator.core.scripts.utils.viewport_manager",
                        SimpleNamespace(ViewportManager=lambda: SimpleNamespace(_hydra_textures=registry)))
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product_paths=["/Render/owned"]))
    env = SimpleNamespace(simulator=SimpleNamespace(scene=SimpleNamespace(sensors={"head": camera}),
        sim=SimpleNamespace(stop=lambda: pytest.fail("inspection cannot stop physics"))))
    namespace["SimulationContext"] = SimpleNamespace(clear_instance=lambda: pytest.fail("must not close stage"))
    before = env.simulator
    result = namespace["release_native_scene"](env,
        flush_app=lambda: pytest.fail("inspection cannot update Kit"), inspect_only=True)
    assert result == {"existing_render_products": 1, "render_product_paths": ["/Render/owned"],
        "camera_sensors": 1, "kit_updates": 0, "environment_step_calls": 0,
        "native_mutations": 0, "inspect_only": True}
    assert env.simulator is before and camera._render_data is not None


def test_explicit_teacher_reference_is_owned_completed_and_hash_bound(tmp_path, monkeypatch):
    tree = ast.parse((ACTOR / "resident_isaac_trainer.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ResidentRavenBackend")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_explicit_test_reference")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "paired-reference", "exec"), namespace)
    calls = []
    def validate(panel, result, sha, *, allowed_arms):
        calls.append((panel, result, sha, allowed_arms))
        assert result == {"arm": "teacher"}
    monkeypatch.setitem(sys.modules, "explicit_motion_panel", SimpleNamespace(validate_result=validate))
    prior = tmp_path / "phases" / "teacher-phase"
    prior.mkdir(parents=True)
    receipt = prior / "native-phase-receipt.json"
    receipt.write_text(json.dumps({"complete": True, "optimizer_steps": 0, "result": {"arm": "teacher"}}))
    backend = SimpleNamespace(runtime_root=tmp_path)
    payload = {"arm": "bc", "output_name": "bc-phase", "reference_output_name": "teacher-phase", "checkpoint_sha256": "a" * 64}
    fn = namespace["_explicit_test_reference"]
    assert fn(backend, payload, {"panel": "pin"}) == prior
    assert calls == [({"panel": "pin"}, {"arm": "teacher"}, "a" * 64, ("teacher",))]
    for change in ({"arm": "dagger"}, {"reference_output_name": "../escape"}, {"reference_output_name": "bc-phase"}):
        with pytest.raises(ValueError, match="previous teacher"):
            fn(backend, {**payload, **change}, {})
    receipt.write_text(json.dumps({"complete": False, "optimizer_steps": 0}))
    with pytest.raises(ValueError, match="incomplete"):
        fn(backend, payload, {})


def test_completed_success_prefix_is_explicitly_bound_before_execution(tmp_path, monkeypatch):
    tree = ast.parse((ACTOR / "resident_isaac_trainer.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ResidentRavenBackend")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_explicit_completed_receipts")
    checked = []
    def identity(path, expected, maximum):
        assert path.stat().st_size < maximum
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("phase artifact content identity changed")
    namespace = {"json": json, "file_identity": identity}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "success-prefix", "exec"), namespace)
    def progress(panel, receipts, **kw):
        checked.append((panel, receipts, kw))
        if receipts != [{"validated": "whole-native-cohort"}]:
            raise ValueError("not a completed prefix")
    monkeypatch.setitem(sys.modules, "explicit_motion_panel", SimpleNamespace(
        completed_progress=progress, panel_sha256=lambda _: "b" * 64))
    path = tmp_path / "resume.json"
    document = {"schema": "roborender.explicit-motion-success-resume.v1",
        "panel_sha256": "b" * 64, "checkpoint_sha256": "c" * 64, "arm": "bc",
        "cohort_receipts": [{"validated": "whole-native-cohort"}]}
    path.write_bytes(canonical(document))
    payload = {"completed_receipts_file": "resume.json",
        "completed_receipts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": "c" * 64, "arm": "bc"}
    backend = SimpleNamespace(artifact=lambda _: path)
    fn = namespace["_explicit_completed_receipts"]
    assert fn(backend, payload, {"full": "original"}) == document["cohort_receipts"]
    assert len(checked) == 1
    for change in ({"panel_sha256": "d" * 64}, {"checkpoint_sha256": "d" * 64},
                   {"arm": "teacher"}, {"extra": True}):
        path.write_bytes(canonical({**document, **change}))
        altered = {**payload, "completed_receipts_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with pytest.raises(ValueError, match="panel/checkpoint/arm differs"):
            fn(backend, altered, {})
    assert len(checked) == 1
    path.write_bytes(canonical(document))
    with pytest.raises(ValueError, match="content identity changed"):
        fn(backend, {**payload, "completed_receipts_sha256": "e" * 64}, {})


def test_training_material_normalization_preserves_paths_order_and_metadata():
    # Import only the pure helper, not Kit, in this CPU regression.
    tree = ast.parse((ACTOR / "resident_isaac_trainer.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "normalize_material_paths")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "material-normalization", "exec"), namespace)
    class VtArray:
        def __iter__(self):
            return iter(("/World/Looks/Z", "/World/Looks/A"))
        def __deepcopy__(self, memo):
            raise RuntimeError("Pickling of pxr.Vt.StringArray is not enabled")
    metadata = object()
    module = SimpleNamespace(table_top_material_prim_paths=VtArray(),
                             leg_material_prim_paths=VtArray(), get_table_grid_meta_data=metadata)
    namespace["normalize_material_paths"](module)
    import copy
    assert copy.deepcopy(module.table_top_material_prim_paths) == ["/World/Looks/Z", "/World/Looks/A"]
    assert module.leg_material_prim_paths == module.table_top_material_prim_paths
    assert module.get_table_grid_meta_data is metadata


def test_client_owner_restart_rejected(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-token")
    token.chmod(0o600)
    client = ResidentClient(18640, token, source_sha256="a"*64, gpu_uuid="GPU-test")
    row = {"ok": True, "role": "isaac_resident", "schema": "roborender.isaac-resident.v1",
           "gpu_uuid": "GPU-test", "source_sha256": "a"*64, "native_initializations": 1,
           "worker_instance_id": "a"*32}
    client._request = lambda *args: row
    assert client.ready()["worker_instance_id"] == "a"*32
    row["worker_instance_id"] = "b"*32
    with pytest.raises(RuntimeError, match="restarted"):
        client.ready()


def test_phase_rollout_reset_is_safe_after_window_cleanup():
    tree = ast.parse((ACTOR / "resident_isaac_trainer.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "reset_rollout_for_scene")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "phase-rollout-reset", "exec"), namespace)
    class NativeActor:
        def init_rollout(self):
            self.steps = 0
            self.obs_dict_buffer = {}
        def clear_rollout(self):
            raise AssertionError("a completed window already called native clear_rollout")
    actor = NativeActor()
    actor.steps, actor.obs_dict_buffer = 42, {"old_scene": object()}
    reset = namespace["reset_rollout_for_scene"]
    reset(actor)
    assert actor.steps == 0 and actor.obs_dict_buffer == {}
    actor.distribution = object()
    reset(actor)
    reset(actor)
    assert actor.distribution is None


@pytest.mark.parametrize("device,index,count,expected", [
    ("cuda", 0, 1, "cuda:0"), ("cuda:0", 0, 1, "cuda:0"),
    ("cpu", 0, 1, ValueError), ("cuda:1", 0, 1, RuntimeError),
    ("cuda", 1, 2, RuntimeError), ("cuda:0", 0, 2, RuntimeError),
])
def test_scene_device_is_indexed_and_keeps_single_gpu_isolation(device,index,count,expected):
    import torch
    tree = ast.parse((ACTOR / "resident_isaac_trainer.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "native_device_string")
    namespace = {"torch": SimpleNamespace(device=torch.device, cuda=SimpleNamespace(
        current_device=lambda: index, device_count=lambda: count))}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "native-device", "exec"), namespace)
    if isinstance(expected, type):
        with pytest.raises(expected):
            namespace["native_device_string"](device)
    else:
        assert namespace["native_device_string"](device) == expected
    source = (ACTOR / "resident_isaac_trainer.py").read_text()
    assert "PanelResetEnvironment(cfg, self.native_device)" in source
    assert "ResidentTrainingEnvironment(cfg, self.native_device)" in source


def test_client_exact_phase_binding(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-token")
    token.chmod(0o600)
    client = ResidentClient(18640, token, source_sha256="a"*64, gpu_uuid="GPU-test")
    def request(route, body=None):
        if route == "/ready":
            return {"ok": True, "role": "isaac_resident", "schema": "roborender.isaac-resident.v1",
                    "gpu_uuid": "GPU-test", "source_sha256": "a"*64, "native_initializations": 1,
                    "worker_instance_id": "a"*32}
        return {"command_id": "r1", "worker_instance_id": "a"*32, "status": "accepted",
                "payload_sha256": hashlib.sha256(canonical(body)).hexdigest()}
    client._request = request
    assert client.submit("r1", "collect_round", {"config": "r1.json"})["status"] == "accepted"
