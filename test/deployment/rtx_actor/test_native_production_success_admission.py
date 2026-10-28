"""Pure admission checks from the real resident method; no Isaac/GPU import."""
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

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
from explicit_motion_panel import canonical
from initial_robot_refresh_diagnostic import CLEAR_ONCE, ORIGINAL, STUDENT_CLEAR_ONCE
from test.deployment.rtx_actor.test_explicit_motion_panel import panel as small_panel
from test.deployment.rtx_actor.test_production_success_contract import accepted


def admission(tmp_path, monkeypatch, panel, *, production=True, arm="bc", options=None, rendered=True):
    """Extract only validate(), retaining its real imports and ordered gates."""
    source = HERE / "resident_isaac_trainer.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "ResidentRavenBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
    path = tmp_path / "panel.json"
    path.write_bytes(canonical(panel))
    checks = []

    def identity(value, expected, maximum=None):
        checks.append(Path(value))
        if Path(value) == path:
            assert maximum and path.stat().st_size <= maximum
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("panel raw identity changed")
        else:
            assert Path(value).parent == Path("/diagnostic-assets")

    scope = {"os": os, "json": json, "Path": Path, "file_identity": identity}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    monkeypatch.setenv("RESIDENT_PRODUCTION_FULL295", "1" if production else "0")
    if os.environ.get("RESIDENT_PIPELINE_MODE") != "smoke":
        monkeypatch.setenv("RESIDENT_PIPELINE_MODE", "production" if production else "test")
    paths = {"panel.json": path}
    owner = SimpleNamespace(runtime_root=tmp_path, _consumed_outputs=set(), artifact=lambda name: paths[name],
                            motion_inventory_sha256=panel["motion_inventory_sha256"])
    payload = dict(panel_file="panel.json", panel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256="b" * 64, arm=arm, output_name="cpu-admission-only")
    payload.update(options or {})
    if rendered and arm in {"bc", "dagger"}:
        from rendered_student_success_contract import CONFIG_SCHEMA
        from rendered_success_observations import ADAPTER_SHA256
        config = {"schema": CONFIG_SCHEMA, "checkpoint_sha256": "b" * 64,
            "panel_sha256": hashlib.sha256(canonical(panel)).hexdigest(),
            "renderer": {"host": "127.0.0.1", "port": 18550, "identity": {
                "renderer": "Image3F", "adapter_sha256": ADAPTER_SHA256, "engine_sha256": "e" * 64,
                "steps": 5, "cfg": 1., "teacache_threshold": .3, "seed": 42, "server_rng_isolated": True}},
            "anchors": {"seed": 123, "files": [{"path": f"/anchors/{i}.npz", "sha256": "d" * 64}
                                                 for i in range(2048)]}}
        paths["config.json"] = tmp_path / "config.json"
        paths["config.json"].write_bytes(canonical(config))
        payload.update(rendered_observation_config_file="config.json",
                       rendered_observation_config_sha256=hashlib.sha256(canonical(config)).hexdigest())
    return lambda: scope["validate"](owner, "evaluate_explicit_panel", payload), checks, payload, path


@pytest.mark.parametrize("arm", ["bc", "dagger"])
def test_production_student_corrected_request_is_admitted_before_checkpoint_check(accepted, tmp_path, monkeypatch, arm):
    call, checks, _, path = admission(tmp_path, monkeypatch, accepted, arm=arm, options={
        "teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE})
    call()
    assert checks == [path, Path("/diagnostic-assets") / ("b" * 64 + ".pt")]


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("fault", ["missing", "missing_hash", "changed_config", "checkpoint", "panel", "renderer", "anchors"])
def test_rendering_config_faults_fail_before_checkpoint_or_action(accepted, tmp_path, monkeypatch, smoke, fault):
    if smoke:
        from smoke_success_contract import make_panel, evaluation_options
        value = make_panel(accepted)
        monkeypatch.setenv("RESIDENT_PIPELINE_MODE", "smoke")
    else:
        from production_success_contract import evaluation_options
        value = accepted
    call, checks, payload, _ = admission(tmp_path, monkeypatch, value, options=evaluation_options(value))
    path = tmp_path / "config.json"
    if fault == "missing":
        payload.pop("rendered_observation_config_file")
        payload.pop("rendered_observation_config_sha256")
    elif fault == "missing_hash":
        payload.pop("rendered_observation_config_sha256")
    elif fault == "changed_config":
        path.write_text("{}")
    else:
        config = json.loads(path.read_bytes())
        if fault == "checkpoint": config["checkpoint_sha256"] = "f" * 64
        elif fault == "panel": config["panel_sha256"] = "f" * 64
        elif fault == "renderer": config["renderer"]["identity"]["steps"] = 50
        else: config["anchors"]["files"].pop()
        path.write_bytes(canonical(config))
        payload["rendered_observation_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        call()
    assert not any(p.parent == Path("/diagnostic-assets") for p in checks)


@pytest.mark.parametrize("options", [
    {}, {"teacher_action_mode": "distill"},
    {"teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": ORIGINAL},
    {"diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE},
    {"teacher_action_mode": "raven_ppo", "diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE},
    {"teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": CLEAR_ONCE},
])
def test_production_student_cannot_omit_or_downgrade_corrected_mode(accepted, tmp_path, monkeypatch, options):
    call, checks, _, _ = admission(tmp_path, monkeypatch, accepted, options=options)
    with pytest.raises(ValueError):
        call()
    assert not any(path.parent == Path("/diagnostic-assets") for path in checks)


@pytest.mark.parametrize("fault", ["seed", "case", "reduced", "raw", "resume"])
def test_production_student_substitute_panel_or_partial_resume_fails_before_model_check(accepted, tmp_path, monkeypatch, fault):
    value = deepcopy(accepted)
    if fault == "seed":
        value["cohorts"][0]["assignment"]["simulator_seed"] += 100_000
    elif fault == "case":
        value["cohorts"][0]["case_ids"][0] = "replacement"
    elif fault == "reduced":
        value["cohorts"] = value["cohorts"][:1]
        value["episode_count"] = 2
    options = {"teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE}
    if fault == "resume":
        options.update(completed_receipts_file="old.json", completed_receipts_sha256="a" * 64)
    call, checks, payload, path = admission(tmp_path, monkeypatch, value, options=options)
    if fault == "raw":
        path.write_text(json.dumps(value, indent=2))
        payload["panel_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        call()
    assert not any(path.parent == Path("/diagnostic-assets") for path in checks)


@pytest.mark.parametrize("production", [False, True])
def test_teacher_corrected_only_can_execute(tmp_path, monkeypatch, production):
    for mode in (ORIGINAL, CLEAR_ONCE):
        call, checks, _, path = admission(tmp_path, monkeypatch, small_panel(), production=production,
            arm="teacher", options={"teacher_action_mode": "raven_ppo", "diagnostic_initial_robot_refresh": mode})
        if mode == ORIGINAL:
            with pytest.raises(ValueError,match='retired'):call()
            assert not any(p.parent == Path('/diagnostic-assets') for p in checks)
        else:
            call()
            assert checks[0] == path and checks[-1].parent == Path("/diagnostic-assets")


def test_legacy_nonproduction_student_smoke_cannot_execute(tmp_path, monkeypatch):
    call, checks, payload, _ = admission(tmp_path, monkeypatch, small_panel(), production=False)
    with pytest.raises(ValueError,match='retired'):call()
    assert "diagnostic_initial_robot_refresh" not in payload
    assert not any(p.parent == Path('/diagnostic-assets') for p in checks)


def test_explicit_corrected_mode_requires_fixed_panel_even_without_production_flag(tmp_path, monkeypatch):
    call, checks, _, _ = admission(tmp_path, monkeypatch, small_panel(), production=False,
        options={"teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE})
    with pytest.raises(ValueError):
        call()
    assert not any(path.parent == Path("/diagnostic-assets") for path in checks)


def test_native_smoke_owner_admits_only_corrected_smoke_panel(accepted,tmp_path,monkeypatch):
    from smoke_success_contract import make_panel,evaluation_options
    monkeypatch.setenv('RESIDENT_PIPELINE_MODE','smoke')
    panel=make_panel(accepted)
    call,checks,_,path=admission(tmp_path,monkeypatch,panel,options=evaluation_options(panel))
    call()
    assert checks==[path,Path('/diagnostic-assets')/('b'*64+'.pt')]
    call,checks,_,_=admission(tmp_path,monkeypatch,accepted,options={
        'teacher_action_mode':'distill','diagnostic_initial_robot_refresh':STUDENT_CLEAR_ONCE})
    with pytest.raises(ValueError,match='production owner|smoke owner'):call()
    assert not any(p.parent==Path('/diagnostic-assets') for p in checks)
