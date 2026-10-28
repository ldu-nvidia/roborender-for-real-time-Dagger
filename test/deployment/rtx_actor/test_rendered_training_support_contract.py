"""CPU-only paired Image3F diagnostic gates; no renderer or Isaac is launched."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import training_support_success_contract as baseline
import rendered_training_support_contract as contract
import initial_robot_refresh_diagnostic as refresh
from explicit_motion_panel import canonical, run_panel, summarize, validate_panel, validate_result
from native_artifact_io import publish_json
from teacher_action_diagnostic import DISTILL, RAVEN_PPO, validate_mode
from test.deployment.rtx_actor.test_training_support_success_contract import accepted, cohort_result, resident_admission


def raw_digest(value):
    return hashlib.sha256(canonical(value) + b"\n").hexdigest()


@pytest.fixture
def paired(accepted, monkeypatch):
    selection, ray = accepted
    panel = deepcopy(ray)
    panel.update(observation_mode=contract.MODE, panel_id=contract.PANEL_ID)
    monkeypatch.setattr(contract, "PANEL_SHA256", hashlib.sha256(canonical(panel)).hexdigest())
    monkeypatch.setattr(contract, "PANEL_RAW_SHA256", raw_digest(panel))
    physics = {"schema": "test.actual-physics.v1", "physics_unchanged": True, "control_hz": 50}
    state = {name: {"dtype": "float32", "shape": [2, 1], "sha256": "a" * 64}
             for name in sorted(contract.STATE_FIELDS)}
    native = []
    for cohort in ray["cohorts"]:
        result = cohort_result(cohort)
        result.update(vector_steps=123, runtime_physics_sha256=contract.physics_sha256(physics))
        native.append(result)
    final = summarize(ray, native)
    final.update(training_support_success_protocol=baseline.protocol_identity(ray),
                 diagnostic_only=True, exact_historical_replay=False)
    monkeypatch.setattr(contract, "BASELINE_RESULT_RAW_SHA256", raw_digest(final))
    reference = {"schema": contract.BASELINE_SCHEMA, "baseline_panel_sha256": baseline.PANEL_SHA256,
        "baseline_result_raw_sha256": contract.BASELINE_RESULT_RAW_SHA256,
        "baseline_reference_manifest_raw_sha256": contract.BASELINE_MANIFEST_RAW_SHA256,
        "baseline_result": final, "cohorts": []}
    for index, result in enumerate(native):
        raws = {"initial_state": json.dumps(state, indent=2).encode() + b"\n",
            "runtime_physics": json.dumps(physics, sort_keys=True, allow_nan=False).encode() + b"\n",
            "result": canonical(result) + b"\n"}
        reference["cohorts"].append({"cohort_index": index, "initial_state": deepcopy(state),
            "runtime_physics": deepcopy(physics), "result": result,
            "source_files": {key: {"bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest()}
                             for key, raw in raws.items()}})
    monkeypatch.setattr(contract, "PAIRED_BASELINE_RAW_SHA256", raw_digest(reference))
    renderer = {"renderer": "Image3F", "adapter_sha256": contract.ADAPTER_SHA256,
        "engine_sha256": "e" * 64, "steps": 5, "cfg": 1., "teacache_threshold": .3,
        "seed": 42, "server_rng_isolated": True}
    config = {"schema": contract.CONFIG_SCHEMA, "checkpoint_sha256": baseline.CHECKPOINT_SHA256,
        "source_selection_sha256": baseline.SOURCE_SELECTION_RAW_SHA256, "panel_sha256": contract.PANEL_SHA256,
        "renderer": {"host": "127.0.0.1", "port": 18500, "identity": renderer},
        "anchors": {"seed": 20261023, "files": [{"path": f"/anchors/{i}.npz", "sha256": "d" * 64}
                                               for i in range(2048)]},
        "paired_baseline_file": "reference.json", "paired_baseline_sha256": raw_digest(reference)}
    return SimpleNamespace(selection=selection, ray=ray, panel=panel, reference=reference, config=config)


def rendered_cohort(data, index):
    value = deepcopy(data.reference["cohorts"][index]["result"])
    value["observation_mode"] = contract.MODE
    value["rendered_observation_receipt"] = {
        "schema": "roborender.rendered-success-observation-receipt.v1", "observation_mode": contract.MODE,
        "renderer_identity": deepcopy(data.config["renderer"]["identity"]),
        "vector_steps_rendered": 123, "render_requests_completed": 246, "raytraced_fallbacks": 0,
        "failed": False, "view_order": ["head_color", "hand_left", "hand_right"],
        "normalizations_per_vector_step": 1, "anchor_policy": "fixed_frame0_per_episode",
        "offline_BC_teacher50_equivalent": False, "controls_from_current_simulator_state": True,
        "simulator_steps_added": 0, "optimizer_steps": 0, "trace_sha256": "f" * 64,
        "anchor_seed": data.config["anchors"]["seed"] + index,
        "anchor_selections": [{"lane": lane, "selected_at_vector_step": 0, "file_sha256": "d" * 64,
                               "frame_index": 0, "prompt_sha256": "c" * 64} for lane in (0, 1)]}
    ref = data.reference["cohorts"][index]
    value["paired_initial_state_attestation"] = contract.compare_initial_state(data.reference,
        cohort_index=index, initial_state=ref["initial_state"], runtime_physics=ref["runtime_physics"])
    return value


def rendered_result(data):
    value = summarize(data.panel, [rendered_cohort(data, i) for i in range(100)])
    value.update(rendered_training_support_protocol=contract.protocol_identity(data.panel, data.config),
                 diagnostic_only=True, exact_historical_replay=False)
    return value


def test_only_observation_and_panel_label_change(paired):
    assert contract.build_panel(paired.ray) == paired.panel
    assert {key for key in paired.ray if paired.ray[key] != paired.panel[key]} == {"panel_id", "observation_mode"}
    assert contract.raytraced_shape(paired.panel) == paired.ray
    assert contract.require_panel(paired.panel, raw_sha256=contract.PANEL_RAW_SHA256) == paired.panel
    from production_success_contract import require_panel as production
    from smoke_success_contract import require_panel_smoke
    for gate in (production, require_panel_smoke, baseline.require_panel):
        with pytest.raises(ValueError): gate(paired.panel)
    protocol = contract.protocol_identity(paired.panel, paired.config)
    assert protocol["optimizer_steps"] == protocol["teacher_action_probability"] == 0
    assert not protocol["fixed295_benchmark"] and not protocol["production_promotion_allowed"]
    assert not protocol["offline_BC_teacher50_equivalent"]


def test_actual_baseline_and_complete_new_result_validate_independently(paired):
    assert contract.validate_baseline(paired.reference, paired.panel) == paired.reference
    contract.validate_execution(paired.panel, paired.selection, baseline.CHECKPOINT_SHA256,
                                paired.config, paired.reference)
    result = rendered_result(paired)
    assert contract.validate_rendered_result(paired.panel, result, baseline.CHECKPOINT_SHA256,
                                             paired.config, paired.reference) == result
    assert result["completed_episodes"] == 200


@pytest.mark.parametrize("fault", ["host", "port", "model", "engine", "rng", "steps", "anchors", "duplicate",
    "anchor_path", "anchor_sha", "seed", "checkpoint", "source", "baseline_path", "panel"])
def test_changed_rendering_config_is_not_admitted(paired, fault):
    value = deepcopy(paired.config)
    if fault == "host": value["renderer"]["host"] = "0.0.0.0"
    elif fault == "port": value["renderer"]["port"] = True
    elif fault == "model": value["renderer"]["identity"]["renderer"] = "fake"
    elif fault == "engine": value["renderer"]["identity"]["engine_sha256"] = "bad"
    elif fault == "rng": value["renderer"]["identity"]["server_rng_isolated"] = False
    elif fault == "steps": value["renderer"]["identity"]["steps"] = 50
    elif fault == "anchors": value["anchors"]["files"].pop()
    elif fault == "duplicate": value["anchors"]["files"][1] = value["anchors"]["files"][0]
    elif fault == "anchor_path": value["anchors"]["files"][0]["path"] = "../anchor.npz"
    elif fault == "anchor_sha": value["anchors"]["files"][0]["sha256"] = "bad"
    elif fault == "seed": value["anchors"]["seed"] = -1
    elif fault == "checkpoint": value["checkpoint_sha256"] = "f" * 64
    elif fault == "source": value["source_selection_sha256"] = "f" * 64
    elif fault == "baseline_path": value["paired_baseline_file"] = "/unowned/baseline.json"
    else: value["panel_sha256"] = "f" * 64
    with pytest.raises(ValueError): contract.validate_config(value, paired.panel)


@pytest.mark.parametrize("fault", ["missing_cohort", "cohort_order", "state", "physics", "source_hash", "result", "final_result"])
def test_baseline_reference_cannot_be_rewritten(paired, fault):
    value = deepcopy(paired.reference)
    if fault == "missing_cohort": value["cohorts"].pop()
    elif fault == "cohort_order": value["cohorts"].reverse()
    elif fault == "state": value["cohorts"][0]["initial_state"]["robot_root"]["sha256"] = "b" * 64
    elif fault == "physics": value["cohorts"][0]["runtime_physics"]["control_hz"] = 25
    elif fault == "source_hash": value["cohorts"][0]["source_files"]["initial_state"]["raw_sha256"] = "b" * 64
    elif fault == "result": value["cohorts"][0]["result"]["runtime_physics_sha256"] = "b" * 64
    else: value["baseline_result"]["successes"] += 1
    with pytest.raises(ValueError): contract.validate_baseline(value, paired.panel)


@pytest.mark.parametrize("field", sorted(contract.STATE_FIELDS) + ["physics"])
def test_every_physical_and_policy_state_mismatch_prevents_first_action(paired, field):
    reference = paired.reference["cohorts"][0]
    state, physics = deepcopy(reference["initial_state"]), deepcopy(reference["runtime_physics"])
    if field == "physics": physics["control_hz"] = 25
    else: state[field]["sha256"] = "b" * 64
    actions = []
    with pytest.raises(RuntimeError, match="no first action"):
        contract.compare_initial_state(paired.reference, cohort_index=0, initial_state=state, runtime_physics=physics)
        actions.append("not permitted")
    assert not actions


def test_actual_native_after_reset_hook_checks_pair_before_any_first_action(paired, tmp_path, monkeypatch):
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExplicitMotionSuccessEvaluator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_diagnostic_after_reset")
    assignment = paired.panel["cohorts"][0]["assignment"]
    state = deepcopy(paired.reference["cohorts"][0]["initial_state"])
    physics = deepcopy(paired.reference["cohorts"][0]["runtime_physics"])
    monkeypatch.setitem(sys.modules, "isaac6_success_diagnostic", SimpleNamespace(
        initial_state=lambda env, obs: state, digest=lambda value: value,
        runtime_physics_evidence=lambda env: physics))
    events = []
    scope = {"hashlib": hashlib, "canonical": canonical,
        "assert_motion_environment": lambda config, assigned: assigned,
        "torch": SimpleNamespace(initial_seed=lambda: assignment["simulator_seed"]),
        "publish_json": lambda path, value: events.append(path.name)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "native-paired-reset", "exec"), scope)
    env = SimpleNamespace(config=SimpleNamespace(domain_rand=SimpleNamespace(image_augmentation=SimpleNamespace(enabled=False))),
        simulator=SimpleNamespace(native_motion_assignment_sha256=hashlib.sha256(canonical(assignment)).hexdigest()),
        reset_dataset=SimpleNamespace(motion_files=[r["path"] for r in assignment["lanes"]],
            sample_idx_based_on_demo_clip_number=[(0, 1), (1, 2)]))
    owner = SimpleNamespace(env=env, motion_assignment=assignment, _first_native_resets={
        0: {"sample_idx": 0}, 1: {"sample_idx": 1}},
        _rendered_paired_baseline=paired.reference, _rendered_cohort_index=0)
    scope["_diagnostic_after_reset"](owner, {}, tmp_path)
    assert events == ["native-motion-reset.json", "paired-initial-state.json"]
    assert owner._paired_initial_state_attestation["checked_before_first_action"]
    events.clear()
    state["robot_root"]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="no first action"):
        scope["_diagnostic_after_reset"](owner, {}, tmp_path)
        events.append("first-policy-action")
    assert events == ["native-motion-reset.json"]
    # The real native train function invokes this synchronous hook before both
    # the observation transform and the teacher/student action dispatch.
    native = (SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text()
    assert native.index("self._diagnostic_after_reset(obs, output)") < native.index("apply_observation_transform(self, obs, dones, step)")
    assert native.index("self._diagnostic_after_reset(obs, output)") < native.index("actions, raw_teacher = select_policy_actions")


@pytest.mark.parametrize("fault", ["fallback", "missing_render", "wrong_count", "extra_step", "optimizer", "bad_state",
    "after_action", "wrong_baseline", "normalization", "view_order", "teacher50", "missing_pair"])
def test_native_cohort_receipt_requires_generation_and_before_action_pairing(paired, fault):
    result = rendered_cohort(paired, 0)
    receipt, pair = result["rendered_observation_receipt"], result["paired_initial_state_attestation"]
    if fault == "fallback": receipt["raytraced_fallbacks"] = 1
    elif fault == "missing_render": receipt["vector_steps_rendered"] -= 1
    elif fault == "wrong_count": receipt["render_requests_completed"] -= 1
    elif fault == "extra_step": receipt["simulator_steps_added"] = 1
    elif fault == "optimizer": receipt["optimizer_steps"] = 1
    elif fault == "bad_state": pair["exact_initial_state_match"] = False
    elif fault == "after_action": pair["physical_steps_before_check"] = 1
    elif fault == "wrong_baseline": pair["baseline_result_raw_sha256"] = "e" * 64
    elif fault == "normalization": receipt["normalizations_per_vector_step"] = 2
    elif fault == "view_order": receipt["view_order"].reverse()
    elif fault == "teacher50": receipt["offline_BC_teacher50_equivalent"] = True
    else: result.pop("paired_initial_state_attestation")
    with pytest.raises(ValueError): contract.validate_cohort_observations(result, count=2)


@pytest.mark.parametrize("fault", ["anchor_seed", "anchor_hash", "frame", "lane", "pair_order", "renderer", "raytraced", "reuse"])
def test_final_receipt_binds_each_cohort_to_admitted_renderer_and_reference(paired, fault):
    result = rendered_result(paired)
    receipt = result["rendered_observation_receipts"][0]
    if fault == "anchor_seed": receipt["anchor_seed"] += 1
    elif fault == "anchor_hash": receipt["anchor_selections"][0]["file_sha256"] = "e" * 64
    elif fault == "frame": receipt["anchor_selections"][0]["frame_index"] = 40
    elif fault == "lane": receipt["anchor_selections"][0]["lane"] = 1
    elif fault == "pair_order": result["paired_initial_state_attestations"].reverse()
    elif fault == "renderer": receipt["renderer_identity"]["engine_sha256"] = "f" * 64
    elif fault == "raytraced": result["observation_mode"] = "raytraced_3view"
    else: result["training_support_success_protocol"] = baseline.protocol_identity(paired.ray)
    with pytest.raises(ValueError):
        contract.validate_rendered_result(paired.panel, result, baseline.CHECKPOINT_SHA256,
                                           paired.config, paired.reference)


def rendered_admission(paired, tmp_path, monkeypatch):
    call, payload, checked, paths, owner = resident_admission(
        (paired.selection, paired.ray), tmp_path, monkeypatch)
    paths.update({"config.json": tmp_path / "config.json", "reference.json": tmp_path / "reference.json"})
    paths["panel.json"].write_bytes(canonical(paired.panel) + b"\n")
    paths["config.json"].write_bytes(canonical(paired.config) + b"\n")
    paths["reference.json"].write_bytes(canonical(paired.reference) + b"\n")
    payload.update(panel_sha256=contract.PANEL_RAW_SHA256, diagnostic_protocol=contract.SCHEMA,
        rendered_observation_config_file="config.json", rendered_observation_config_sha256=raw_digest(paired.config))
    return call, payload, checked, paths, owner


def test_real_resident_admission_reads_all_hash_bound_inputs_before_checkpoint(paired, tmp_path, monkeypatch):
    call, _, checked, paths, _ = rendered_admission(paired, tmp_path, monkeypatch)
    call()
    assert checked[-1] == Path("/diagnostic-assets") / (baseline.CHECKPOINT_SHA256 + ".pt")


@pytest.mark.parametrize("fault", ["protocol", "missing_config", "missing_config_hash", "config_bytes", "baseline_bytes",
    "reference", "resume", "teacher", "source", "reset"])
def test_resident_partial_or_changed_rendered_request_never_loads_checkpoint(paired, tmp_path, monkeypatch, fault):
    call, payload, checked, paths, _ = rendered_admission(paired, tmp_path, monkeypatch)
    if fault == "protocol": payload["diagnostic_protocol"] = baseline.SCHEMA
    elif fault == "missing_config": payload.pop("rendered_observation_config_file")
    elif fault == "missing_config_hash": payload.pop("rendered_observation_config_sha256")
    elif fault == "config_bytes": paths["config.json"].write_bytes(b"{}\n")
    elif fault == "baseline_bytes": paths["reference.json"].write_bytes(b"{}\n")
    elif fault == "reference": payload["reference_output_name"] = "other"
    elif fault == "resume": payload.update(completed_receipts_file="prior", completed_receipts_sha256="f" * 64)
    elif fault == "teacher": payload["arm"] = "teacher"
    elif fault == "source": payload["training_support_selection_sha256"] = "f" * 64
    else: payload["diagnostic_initial_robot_refresh"] = "raven_original"
    with pytest.raises(ValueError): call()
    assert not any(path.parent == Path("/diagnostic-assets") for path in checked)


@pytest.mark.parametrize("native_failure", [False, True])
def test_real_evaluator_uses_original_trainer_independent_preview_paths_and_closes_clients(paired, tmp_path, monkeypatch, native_failure):
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_explicit_panel")
    prepared, adapters = [], []
    trainer = SimpleNamespace(env=object())
    class OwnedPath(type(Path())):
        def is_relative_to(self, *others):
            return super().is_relative_to(tmp_path if others == ("/dagger-runtime",) else others[0])
    class Adapter:
        def __init__(self, owner, **kwargs):
            assert owner is trainer
            assert kwargs["preview_directory"].parent.name == "render-previews"
            assert kwargs["telemetry_path"].parent.name == "render-timings"
            assert kwargs["client_factory"](0) != kwargs["client_factory"](1)
            self.closed = False
            adapters.append(self)
        def close(self): self.closed = True
    class Evaluator:
        def evaluate_cohort(self, assignment, **kwargs):
            index = self._rendered_cohort_index
            assert self.env is trainer.env and self._rendered_paired_baseline == paired.reference
            assert self._success_observation_transform is adapters[-1]
            assert not kwargs["output"].exists()
            if native_failure:
                raise RuntimeError("native failed before completion")
            return rendered_cohort(paired, index)
    def pooled_adapter(*args, **kwargs):
        raise AssertionError("historical paired diagnostic must not select the pooled benchmark adapter")
    monkeypatch.setitem(sys.modules, "rendered_success_observations", SimpleNamespace(
        GeneratedSuccessObservations=Adapter, GeneratedPooledSuccessObservations=pooled_adapter))
    monkeypatch.setitem(sys.modules, "pipeline.dagger.services.render_student",
                        SimpleNamespace(_Image3FLoopbackClient=lambda *args: object()))
    scope = {"validate_panel": validate_panel, "validate_mode": validate_mode,
        "require_corrected_execution": refresh.require_corrected_execution,
        "validate_initial_refresh_panel": refresh.validate_panel_mode,
        "STUDENT_CLEAR_ONCE": refresh.STUDENT_CLEAR_ONCE, "SMOKE_STUDENT_CLEAR_ONCE": refresh.SMOKE_STUDENT_CLEAR_ONCE,
        "DISTILL": DISTILL, "RAVEN_PPO": RAVEN_PPO, "Path": OwnedPath,
        "ExplicitMotionSuccessEvaluator": Evaluator, "publish_json": publish_json, "run_panel": run_panel}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "rendered-evaluator-routing", "exec"), scope)
    kwargs = dict(root=tmp_path / "fresh",
        checkpoint_path="/diagnostic-assets/frozen.pt", checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        prepare_cohort=prepared.append, report=lambda row: None, arm="bc", training_support_selection=paired.selection,
        rendered_observation_config=paired.config, rendered_paired_baseline=paired.reference,
        **contract.evaluation_options(paired.panel))
    if native_failure:
        with pytest.raises(RuntimeError, match="native failed"):
            scope["evaluate_explicit_panel"](trainer, paired.panel, **kwargs)
        assert len(adapters) == len(prepared) == 1 and adapters[0].closed
        assert not (tmp_path / "fresh" / "result.json").exists()
        assert not list((tmp_path / "fresh" / "cohort-receipts").iterdir())
        assert not hasattr(trainer, "_success_observation_transform")
        return
    result = scope["evaluate_explicit_panel"](trainer, paired.panel, **kwargs)
    assert len(adapters) == len(prepared) == 100 and all(adapter.closed for adapter in adapters)
    assert not hasattr(trainer, "_success_observation_transform")
    assert contract.validate_rendered_result(paired.panel, result, baseline.CHECKPOINT_SHA256,
                                             paired.config, paired.reference) == result
