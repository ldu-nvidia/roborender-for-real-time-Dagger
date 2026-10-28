"""CPU tests of test-only native input recording/replay, never native proof."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.bc_collection import run_continuous as runner
from pipeline.deployment.rtx_actor import isaac6_bc_production as bc
from test.deployment.rtx_actor import bc_native_same_input as helper
from test.deployment.source_paths import REPO_ROOT
from test.pipeline.test_continuous_bc_launcher import fixture_capacity, request


REFERENCE = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_videogen.py"


def owner(output):
    value = helper.reference_writer(REFERENCE)()
    value.env = SimpleNamespace(num_envs=2)
    value._ep_buf = value._ep_seq_base = None
    value._vg_saved_chunks = 0
    value._vg_run_tag = "test-fixture"
    value.vg_collect_dir = str(output)
    value.vg_prompt = "CPU fixture, not native qualification"
    value.vg_save_raw_depth_native = value.vg_save_raw_depth = True
    value.vg_control_res = [2, 3]
    value.vg_depth_mode, value.vg_invalid_fill = "h5_v4", "far"
    value.vg_depth_near, value.vg_depth_far, value.vg_invert_depth = .25, 6., True
    value.vg_depth_near_per_view = value.vg_depth_far_per_view = {}
    value.vg_h5v4_default = dict(raw_clip=[0., 10.], inv_clip=[1., 20.])
    value.vg_h5v4_per_view = {"hand_left": dict(raw_pct=[1, 99]), "hand_right": dict(raw_pct=[1, 99])}
    value.algo_obs_dim_dict = {"actor_obs": 113, "teacher_obs": 226, "vision_obs": 6}
    return value


def populate(value, window, view_order=bc.VIEWS):
    rows = torch.arange((window - 1) * 81, window * 81).float()
    value._depth_chunk = [(torch.arange(48).reshape(2, 4, 6, 1) / 40 + row / 1000).float() for row in rows]
    value._robot_rgb_chunk = [torch.full((2, 4, 6, 3), int(row) % 256, dtype=torch.uint8) for row in rows]
    value._views_chunk = [{view: dict(depth=depth + i / 9, robot_rgb=rgb.clone(), rgb=None)
        for i, view in enumerate(view_order)} for depth, rgb in zip(value._depth_chunk, value._robot_rgb_chunk)]
    value.storage = SimpleNamespace(
        actor_obs=rows[:, None, None].expand(81, 2, 113).clone(),
        teacher_obs=rows[:, None, None].expand(81, 2, 226).clone(),
        gt_actions=rows[:, None, None].expand(81, 2, 26).clone(),
        dones=torch.zeros(81, 2, 1, dtype=torch.bool))
    if window == 1:
        value.storage.dones[39, 0] = True
        value.storage.dones[80, 1] = True


def write(path, value):
    helper.publish(path, value)


def recorded_fixture(tmp_path, *, view_order=bc.VIEWS, legacy=False):
    """Construct explicit CPU fixture evidence; never returned as native proof."""
    runtime = tmp_path / "runtime"
    chunks = runtime / "bc_isaac"
    chunks.mkdir(parents=True)
    writer = owner(chunks)
    values = fixture_capacity()
    raw = request(2)
    values["launch"]["request"] = values["native"]["request"] = raw
    values["launch"]["capacity_contract"]["num_envs"] = 2
    values["launch"]["command"] = ["trainer._target_=bc_native_same_input.CapturedContinuousBCTrainer"]
    capture_sha = helper.LEGACY_CAPTURE_PIN if legacy else helper.sha(helper.__file__)
    if legacy:
        legacy_proof_fixture(runtime, values)
    native = values["native"]
    native["initial_state_attestation"]["initial_reset_completion"].update(num_envs=2, cleared_lanes=2)
    refs, emitted = [], [0, 0]
    for window in range(1, 5):
        populate(writer, window, view_order)
        destination = runtime / "bc-parity-inputs" / f"window-{window:06d}"
        before_rng = torch.get_rng_state().clone()
        before_depth = writer._depth_chunk[0].clone()
        helper.capture_window(writer, destination, window)
        if legacy:
            path = destination / "inputs.json"
            metadata = helper.read(path)
            metadata.pop("view_order")  # Reproduce the already-frozen recorder omission.
            path.write_bytes(bc.canonical(metadata))
        torch.testing.assert_close(torch.get_rng_state(), before_rng)
        torch.testing.assert_close(writer._depth_chunk[0], before_depth)
        first = writer._vg_saved_chunks
        writer._save_collected_chunks()
        paths = [chunks / f"chunk_test-fixture_{i:07d}.npz" for i in range(first, writer._vg_saved_chunks)]
        for path in paths:
            with np.load(path) as data:
                emitted[int(data["env_id"])] += 1
        write(destination / "outputs.json", dict(schema=helper.SCHEMA, window=window,
            inputs=helper.record(destination / "inputs.json", runtime),
            chunks=[helper.record(path, runtime) for path in paths],
            native_tail_rows=[len(buf.get("gt", [])) for buf in writer._ep_buf],
            native_sequence=list(writer._ep_seq_base)))
        refs.append(helper.record(destination / "outputs.json", runtime))
        for event in values["save_events"][(window - 1) * 2:window * 2]:
            event["num_envs"] = 2
            event["emitted_chunks_per_lane"] = list(emitted)
    native.update(emitted_per_lane=emitted, emitted_chunks=sum(emitted), selected_chunks=sum(emitted))
    execution = values["execution_receipt"]
    execution.update(actual_num_envs=2, request_canonical_sha256=runner.digest(bc.canonical(raw)))
    evidence = {}
    for key, name in (("launch", "launch.json"), ("host", "host-result.json"),
                      ("native", "bc_isaac/cohort-manifest.json")):
        write(runtime / name, values[key])
        evidence[key] = helper.sha(runtime / name)
    execution.update(launch_sha256=evidence["launch"], host_result_sha256=evidence["host"],
                     native_manifest_sha256=evidence["native"])
    write(runtime / "continuous-execution.json", execution)
    evidence["execution"] = helper.sha(runtime / "continuous-execution.json")
    for key, name in (("save_events", "capacity-save-events.jsonl"), ("memory_samples", "memory-samples.jsonl")):
        (runtime / name).write_bytes(b"".join(bc.canonical(row) + b"\n" for row in values[key]))
        evidence[key] = helper.sha(runtime / name)
    values["evidence"] = evidence
    write(runtime / "capacity-receipt.json", helper.instrument_capacity(
        runner.build_capacity(**values), harness_sha256=capture_sha))
    write(runtime / "bc-parity-inputs/capture.json", dict(schema=helper.SCHEMA, request=raw,
        request_sha256=values["launch"]["request_sha256"], harness_sha256=capture_sha,
        original_writer_sha256=helper.PIN, windows=refs,
        native=helper.record(runtime / "bc_isaac/cohort-manifest.json", runtime),
        qualification_only=True, production_admitted=False, physics_replayed=False))
    return runtime


def legacy_proof_fixture(runtime, values):
    """Independent source/config evidence for a deliberately synthetic run."""
    old_helper = (REPO_ROOT / "runtime/raven-stage-certification-20261025-kBFQ3q/"
                  "source-bc-native-v1/code" / helper.SELF)
    assert helper.sha(old_helper) == helper.LEGACY_CAPTURE_PIN
    config = REFERENCE.parents[2] / "config/exp/loco_manip/wsdpt_videogen_collect.yaml"
    assert helper.sha(config) == helper.COLLECT_CONFIG_PIN
    rows = [dict(path=helper.SELF, sha256=helper.LEGACY_CAPTURE_PIN,
                 bytes=old_helper.stat().st_size, executable=False)]
    manifest = runtime / "fixture-source-manifest.json"
    source_sha = runner.digest(bc.canonical(rows))
    write(manifest, dict(schema="roborender.source-snapshot.v1", files=rows,
                         file_count=len(rows), source_sha256=source_sha))
    runtime_config = runtime / "hydra/.hydra/config.yaml"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_bytes(config.read_bytes())
    simulator = REFERENCE.parents[2] / "simulator/isaacsim/isaacsim.py"
    assert helper.sha(simulator) == helper.SIMULATOR_PIN
    relative = "pipeline/GR00T-VisualSim2Real/gr00t/rl/simulator/isaacsim/isaacsim.py"
    overlay = runtime / "overlay"
    derived = overlay / relative
    derived.parent.mkdir(parents=True)
    derived.write_bytes(simulator.read_bytes())
    port = overlay / "port-manifest.json"
    write(port, dict(files=[dict(path=relative, source_sha256=helper.SIMULATOR_PIN,
        derived_sha256=helper.sha(derived), bytes=derived.stat().st_size)]))
    for execution in (values["launch"]["collector_execution"],
                      values["launch"]["capacity_contract"]["collector_execution"],
                      values["execution_receipt"]["collector_execution"]):
        execution.update(source_sha256=source_sha, source_manifest_sha256=helper.sha(manifest),
                         port_manifest_sha256=helper.sha(port))
    values["launch"]["command"].extend(["ROBORENDER_BC_SAME_INPUT_SHA256=" + helper.LEGACY_CAPTURE_PIN,
                                         "+exp=loco_manip/wsdpt_videogen_collect"])
    def external(path):
        return dict(path=str(path), sha256=helper.sha(path), bytes=path.stat().st_size)
    write(runtime / "legacy-proof.json", dict(schema="roborender.bc-legacy-capture-view-order.v1",
        capture_helper=external(old_helper), source_manifest=external(manifest), reference_config=external(config),
        runtime_config=helper.record(runtime_config, runtime)))


@pytest.mark.parametrize("view_order", [bc.VIEWS, ("hand_left", "hand_right", "head_color")])
def test_exact_recorded_input_replay_all_arrays_bytes_and_cross_window_tails(tmp_path, view_order):
    runtime = recorded_fixture(tmp_path, view_order=view_order)
    result = helper.compare(runtime, REFERENCE, tmp_path / "reference-output")
    assert result["verdict"] == "PASS" and result["chunks"] == 7 and result["windows"] == 4
    assert result["comparison_scope"] == "exact_native_writer_inputs_only"
    assert result["independent_native_trajectory_tested"] is False
    assert result["physics_replayed"] is result["production_admitted"] is False
    assert result["comparator_sha256"] == result["harness_sha256"] == helper.sha(helper.__file__)
    with pytest.raises(ValueError, match="fresh"):
        helper.compare(runtime, REFERENCE, tmp_path / "reference-output")


def test_legacy_capture_uses_independent_native_config_not_npz_order(tmp_path):
    runtime = recorded_fixture(tmp_path, view_order=("hand_left", "hand_right", "head_color"), legacy=True)
    with pytest.raises(ValueError, match="capture identity"):
        helper.compare(runtime, REFERENCE, tmp_path / "must-not-start")
    result = helper.compare(runtime, REFERENCE, tmp_path / "reference-output",
                            legacy_order_proof=runtime / "legacy-proof.json")
    assert result["chunks"] == 7 and result["harness_sha256"] == helper.LEGACY_CAPTURE_PIN
    assert result["comparator_sha256"] == helper.sha(helper.__file__) != result["harness_sha256"]
    assert result["legacy_order_proof"]["view_order"] == ["hand_left", "hand_right", "head_color"]
    assert result["legacy_order_proof"]["inferred_from_output"] is False
    assert result["legacy_order_proof"]["config_captured_in_original_capture_receipt"] is False


@pytest.mark.parametrize("fault", ["helper", "config", "manifest", "native_config", "port", "override"])
def test_legacy_view_order_proof_rejects_missing_changed_or_unbound_inputs(tmp_path, fault):
    runtime = recorded_fixture(tmp_path, view_order=("hand_left", "hand_right", "head_color"), legacy=True)
    proof_path = runtime / "legacy-proof.json"
    proof = helper.read(proof_path)
    if fault in ("helper", "config", "manifest"):
        key = {"helper": "capture_helper", "config": "reference_config", "manifest": "source_manifest"}[fault]
        proof[key]["sha256"] = "0" * 64
        proof_path.write_bytes(bc.canonical(proof))
    elif fault == "native_config":
        path = runtime / "hydra/.hydra/config.yaml"
        path.write_bytes(path.read_bytes() + b"\n# changed independently saved config\n")
    elif fault == "port":
        (runtime / "overlay/port-manifest.json").write_bytes(b"{}")
    else:
        launch = helper.read(runtime / "launch.json")
        launch["command"].append("simulator.config.cameras.conditioning_views=[]")
        with pytest.raises(ValueError, match="binding differs"):
            helper.legacy_view_order(runtime, proof_path, helper.read(runtime / "bc-parity-inputs/capture.json"), launch)
        return
    with pytest.raises(ValueError):
        helper.compare(runtime, REFERENCE, tmp_path / "replay", legacy_order_proof=proof_path)


def test_capture_rejects_in_window_view_order_change(tmp_path):
    value = owner(tmp_path)
    populate(value, 1)
    value._views_chunk[1] = dict(reversed(list(value._views_chunk[1].items())))
    with pytest.raises(ValueError, match="insertion order"):
        helper.capture_window(value, tmp_path / "capture", 1)


@pytest.mark.parametrize("fault", ["input", "output", "window", "host", "resource", "extra_chunk"])
def test_comparison_rejects_changed_or_incomplete_bound_evidence(tmp_path, fault):
    runtime = recorded_fixture(tmp_path)
    if fault == "input":
        with (runtime / "bc-parity-inputs/window-000001/gt_actions.npy").open("ab") as stream:
            stream.write(b"tamper")
    elif fault == "output":
        with next((runtime / "bc_isaac").glob("chunk_*.npz")).open("ab") as stream:
            stream.write(b"tamper")
    elif fault == "window":
        path = runtime / "bc-parity-inputs/capture.json"
        value = helper.read(path)
        value["windows"] = value["windows"][::-1]
        path.write_bytes(bc.canonical(value))
    elif fault in ("host", "resource"):
        path = runtime / ("host-result.json" if fault == "host" else "capacity-receipt.json")
        value = helper.read(path)
        value["exit_code" if fault == "host" else "minimum_full_emissions_per_lane"] = 999
        path.write_bytes(bc.canonical(value))
    else:
        (runtime / "bc_isaac/chunk_extra.npz").write_bytes(b"not admitted")
    with pytest.raises((ValueError, AssertionError)):
        helper.compare(runtime, REFERENCE, tmp_path / "replay")


def test_instrumentation_keeps_actual_command_guard_source_and_resource_bounds(tmp_path):
    image = "sha256:" + "a" * 64
    def base(*_a, **_k):
        return ["docker", "run", "--memory", "40g", "--memory-swap", "40g", "--env",
                "PYTHONPATH=/isaac6-port", image, "python", "train.py", "trainer._target_=old"]
    _, command = runner.build_command(request=request(2), request_path="/request.json", request_sha256="b" * 64,
        source=REPO_ROOT, runtime=tmp_path, overlay=tmp_path / "overlay", physical_gpu=0, memory_gib=43,
        guard_sha256="c" * 64, image_id=image, base_builder=base)
    before = list(command)
    actual = helper.instrument_command(command, source=REPO_ROOT, image_id=image, request=request(2))
    assert command == before  # No mutation of an already-built command.
    assert actual.count("trainer._target_=bc_native_same_input.CapturedContinuousBCTrainer") == 1
    assert "ROBORENDER_CONTINUOUS_BC_GUARD_SHA256=" + "c" * 64 in actual
    assert "ROBORENDER_BC_SAME_INPUT_SHA256=" + helper.sha(helper.__file__) in actual
    assert any("dst=/bc-parity-source,readonly" in item for item in actual)
    for flag in ("--memory", "--memory-swap", "--cidfile"):
        assert actual[actual.index(flag) + 1] == command[command.index(flag) + 1]
    assert "algo.config.num_learning_epochs=0" in actual and "algo.trl.num_total_batches=4" in actual
    assert not any("restart" in item for item in actual)


def test_run_delegates_guards_adds_disk_budget_and_restores_injection(monkeypatch):
    originals = runner.build_command, runner.storage_budget, runner.build_capacity
    policy = dict(schema=runner.STORAGE_SCHEMA, root="/data", reserve_bytes=64 * helper.GIB,
        bytes_per_emission_budget=512 << 20, session_metadata_budget_bytes=helper.GIB, poll_seconds=1)
    def main(argv):
        budget = runner.storage_budget(policy, request(2))
        assert budget["same_input_capture_budget_bytes"] == 8 * helper.GIB + (256 << 20)
        assert budget["required_available_bytes"] == 64 * helper.GIB + 5 * helper.GIB + budget["same_input_capture_budget_bytes"]
        receipt = runner.build_capacity(**fixture_capacity())
        assert receipt["schema"] == helper.CAPACITY_SCHEMA
        with pytest.raises(ValueError):
            runner.validate_capacity(receipt, receipt["contract"])
        raise RuntimeError("fixture fails before any native launch")
    monkeypatch.setattr(runner, "main", main)
    with pytest.raises(RuntimeError, match="fixture fails"):
        helper.run(["--source-root", str(REPO_ROOT), "--qualification", "--num-envs", "2"])
    assert (runner.build_command, runner.storage_budget, runner.build_capacity) == originals


@pytest.mark.parametrize("width,windows", [(3, 4), (32, 4), (2, 5), (1, 100)])
def test_test_harness_cannot_expand_to_production_profile(width, windows):
    with pytest.raises(ValueError):
        helper.validate_request(request(width, windows=windows))


def test_capture_guard_fails_before_allocating_or_writing_payload(tmp_path):
    value = owner(tmp_path)
    populate(value, 1)
    def fail():
        raise RuntimeError("memory headroom")
    with pytest.raises(RuntimeError, match="headroom"):
        helper.capture_window(value, tmp_path / "capture", 1, guard=fail)
    assert list((tmp_path / "capture").iterdir()) == []
    assert value._vg_saved_chunks == 0 and len(value._depth_chunk) == 81
