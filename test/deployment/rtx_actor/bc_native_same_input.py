"""Explicit, bounded BC qualification instrumentation; never a production entry.

`run` delegates to the guarded continuous launcher, recording an explicit test
trainer before Docker acceptance. `compare` is CPU-only: replay the actual
captured writer inputs through hash-pinned Raven, never replay Isaac physics.
All outputs are qualification-only and cannot admit a production collection.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace


GIB = 2**30
SELF = "test/deployment/rtx_actor/bc_native_same_input.py"
PIN = "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b"
SCHEMA = "roborender.bc-native-same-input.v1"
CAPACITY_SCHEMA = "roborender.bc-instrumented-test-capacity.v1"
ACK = "CAPTURE_WRITER_INPUTS_ONLY_NO_PHYSICS_REPLAY_V1"
LEGACY_CAPTURE_PIN = "fa764e28a23c8eb051c3f77ce0c40a9f511315a324699d3c4c353b545113ae49"
COLLECT_CONFIG_PIN = "c483a32b5a1810dd7abcedfb425b86c939d45e02a0ea43d448f8b0b226f21e5c"
SIMULATOR_PIN = "1d03ebca26f8c5f64d65d2146e79bbf8ceab6af2b83dfb65afa690b784b1073f"
VIEWS = ("head_color", "hand_left", "hand_right")
ATTRS = ("vg_prompt", "vg_save_raw_depth_native", "vg_save_raw_depth", "vg_control_res",
         "vg_depth_mode", "vg_depth_near", "vg_depth_far", "vg_depth_near_per_view",
         "vg_depth_far_per_view", "vg_invert_depth", "vg_h5v4_per_view",
         "vg_h5v4_default", "vg_invalid_fill")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size)


def record(path, root):
    path, root = Path(path), Path(root)
    return dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)


def checked(root, item):
    root = Path(root).resolve(strict=True)
    path = root / item["path"]
    if (Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts
            or path.is_symlink() or path.resolve(strict=True) != path or not path.is_file()
            or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]):
        raise ValueError("same-input evidence path/bytes/hash changed")
    return path


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 << 20:
        raise ValueError("bounded regular JSON required")
    return json.loads(path.read_bytes())


def validate_request(request):
    if (request.get("schema") != "roborender.bc-native-continuous-request.v1"
            or len(request.get("motions", [])) not in (1, 2)
            or request.get("max_windows") not in (4, 8)):
        raise ValueError("same-input native test supports only B1/B2 and4/8windows")
    return request


def capture_window(trainer, root, window, *, guard=lambda: None):
    """Write one tensor at a time; no extra model, rendering or physics call."""
    import numpy as np
    import torch

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    b = int(trainer.env.num_envs)
    if b not in (1, 2) or len(trainer._depth_chunk) != 81:
        raise ValueError("bounded full writer input required")
    view_order = list(trainer._views_chunk[0])
    if (len(view_order) != 3 or set(view_order) != set(VIEWS)
            or any(list(row) != view_order for row in trainer._views_chunk)):
        raise ValueError("original conditioning-view insertion order must remain constant")
    # OmegaConf values must become ordinary metadata, never pickle payloads.
    def plain(value):
        if value is None or type(value) in (str, int, float, bool):
            return value
        if hasattr(value, "items"):
            return {str(k): plain(v) for k, v in value.items()}
        return [plain(v) for v in value]
    metadata = dict(schema=SCHEMA, window=window, num_envs=b,
        attrs={key: plain(getattr(trainer, key)) for key in ATTRS},
        observation_keys=list(trainer.algo_obs_dim_dict), view_order=view_order, run_tag=trainer._vg_run_tag,
        original_writer_sha256=PIN, arrays={}, capture_added_physics_steps=0,
        capture_added_policy_calls=0, capture_added_renderer_calls=0)
    total = 0
    def save(key, value):
        nonlocal total
        guard()
        array = value.detach().cpu().numpy()
        if array.dtype.kind not in "biuf" or array.shape[:2] != (81, b):
            raise ValueError("writer input must be numeric full T/B array")
        total += array.nbytes
        if total > b * GIB:
            raise RuntimeError("writer input exceeds declared1GiB/lane/window capture budget")
        path = root / (key + ".npy")
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        metadata["arrays"][key] = dict(**record(path, root), shape=list(array.shape), dtype=array.dtype.str)
    def sequence(key, values):
        guard()  # Before allocating the single stacked CPU buffer.
        save(key, torch.stack(values))
    sequence("primary_depth", trainer._depth_chunk)
    sequence("primary_robot", trainer._robot_rgb_chunk)
    for view in view_order:
        for kind in ("depth", "robot_rgb"):
            sequence(view + "__" + kind, [row[view][kind] for row in trainer._views_chunk])
        if any(row[view].get("rgb") is not None for row in trainer._views_chunk):
            raise ValueError("sim-RGB debug recipe is outside this bounded test")
    for key in trainer.algo_obs_dim_dict:
        if key != "vision_obs":
            save("obs__" + key, getattr(trainer.storage, key))
    save("gt_actions", trainer.storage.gt_actions)
    save("dones", trainer.storage.dones)
    metadata["array_bytes"] = total
    publish(root / "inputs.json", metadata)
    return metadata


def __getattr__(name):
    if name != "CapturedContinuousBCTrainer":
        raise AttributeError(name)
    # Hydra reaches this only after the normal AppLauncher/Kit startup.
    from continuous_bc_guard import GuardedContinuousBCTrainer, memory_gate, validate_full_buffers
    from isaac6_bc_production import load_request
    from isaac6_bc_session import memory_snapshot
    request = validate_request(load_request())
    if (os.environ.get("ROBORENDER_BC_SAME_INPUT_ACK") != ACK
            or os.environ.get("ROBORENDER_BC_SAME_INPUT_SHA256") != sha(__file__)):
        raise RuntimeError("explicit frozen test harness acknowledgement/hash required")
    b = len(request["motions"])
    cap = int(os.environ["ROBORENDER_CONTINUOUS_BC_MEMORY_BYTES"])
    runtime = Path("/dagger-runtime")
    def guard():
        rows = dict(line.split(":", 1) for line in Path("/run/host-meminfo").read_text().splitlines())
        available = int(rows["MemAvailable"].split()[0]) * 1024
        return memory_gate(memory_snapshot(), available, num_envs=b, cap_bytes=cap, saving=True)

    class CapturedContinuousBCTrainer(GuardedContinuousBCTrainer):
        def _save_collected_chunks(self):
            validate_full_buffers(self, b)
            guard()
            window = self._bc_ledger.vector_steps // 81
            first = self._vg_saved_chunks
            root = runtime / "bc-parity-inputs" / f"window-{window:06d}"
            capture_window(self, root, window, guard=guard)
            result = super()._save_collected_chunks()
            # The ordinary writer/guard is unchanged. Bind every actual output,
            # including the legitimate zero-emission-window case.
            publish(root / "outputs.json", dict(schema=SCHEMA, window=window,
                inputs=record(root / "inputs.json", runtime),
                chunks=[record(Path(self.vg_collect_dir) / f"chunk_{self._vg_run_tag}_{i:07d}.npz", runtime)
                        for i in range(first, self._vg_saved_chunks)],
                native_tail_rows=[len(buf.get("gt", [])) for buf in self._ep_buf],
                native_sequence=list(self._ep_seq_base)))
            return result

        def train(self):
            result = super().train()
            refs = [record(runtime / "bc-parity-inputs" / f"window-{i:06d}" / "outputs.json", runtime)
                    for i in range(1, request["max_windows"] + 1)]
            publish(runtime / "bc-parity-inputs" / "capture.json", dict(schema=SCHEMA,
                request=request, request_sha256=os.environ["ROBORENDER_BC_COLLECTION_REQUEST_SHA256"],
                harness_sha256=sha(__file__), original_writer_sha256=PIN, windows=refs,
                native=record(runtime / "bc_isaac/cohort-manifest.json", runtime),
                qualification_only=True, production_admitted=False, physics_replayed=False))
            return result
    globals()[name] = CapturedContinuousBCTrainer
    return CapturedContinuousBCTrainer


def instrument_command(command, *, source, image_id, request):
    validate_request(request)
    source = Path(source)
    helper = source / SELF
    cmd = list(command)
    target = "trainer._target_=continuous_bc_guard.GuardedContinuousBCTrainer"
    if cmd.count(target) != 1 or cmd.count(image_id) != 1:
        raise ValueError("unchanged continuous guard command required before instrumentation")
    cmd[cmd.index(target)] = "trainer._target_=bc_native_same_input.CapturedContinuousBCTrainer"
    paths = [i for i, value in enumerate(cmd) if value.startswith("PYTHONPATH=")]
    if len(paths) != 1:
        raise ValueError("one explicit native PYTHONPATH required")
    cmd[paths[0]] = cmd[paths[0]].replace("PYTHONPATH=", "PYTHONPATH=/bc-parity-source:", 1)
    extra = ["--mount", f"type=bind,src={helper.parent},dst=/bc-parity-source,readonly",
        "--env", "ROBORENDER_BC_SAME_INPUT_ACK=" + ACK,
        "--env", "ROBORENDER_BC_SAME_INPUT_SHA256=" + sha(helper)]
    cmd[cmd.index(image_id):cmd.index(image_id)] = extra
    return cmd


def run(argv):
    # No source mutation: explicit test-scoped dependency injection, before any
    # immutable launch/acceptance. The original runtime guard owns execution.
    from pipeline.bc_collection import run_continuous as runner
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--num-envs", type=int, required=True)
    args, _ = parser.parse_known_args(argv)
    if not args.qualification or args.num_envs not in (1, 2):
        raise ValueError("explicit small-batch qualification required; production forbidden")
    if Path(__file__).resolve() != (args.source_root / SELF).resolve():
        raise ValueError("run this helper from the same frozen source release")
    original_build, original_budget, original_capacity = runner.build_command, runner.storage_budget, runner.build_capacity
    def build(**kwargs):
        name, cmd = original_build(**kwargs)
        image = kwargs.get("image_id")
        if image is None:
            from qualify_native_isaac6 import IMAGE
            image = IMAGE
        return name, instrument_command(cmd, source=kwargs["source"], image_id=image, request=kwargs["request"])
    def budget(policy, request):
        validate_request(request)
        value = original_budget(policy, request)
        # Arrays<=1GiB/lane/window, plus64MiB/window for NPY/JSON metadata.
        capture = request["max_windows"] * (len(request["motions"]) * GIB + (64 << 20))
        return dict(value, budget_bytes=value["budget_bytes"] + capture,
            required_available_bytes=value["required_available_bytes"] + capture,
            same_input_capture_budget_bytes=capture, instrumented_qualification_only=True)
    def capacity(**kwargs):
        return instrument_capacity(original_capacity(**kwargs))
    runner.build_command, runner.storage_budget, runner.build_capacity = build, budget, capacity
    try:
        return runner.main(argv)
    finally:
        runner.build_command, runner.storage_budget, runner.build_capacity = original_build, original_budget, original_capacity


def instrument_capacity(value, *, harness_sha256=None):
    # Instrumentation changes RAM/I/O; this schema never admits production.
    return dict(value, schema=CAPACITY_SCHEMA, instrumented_qualification_only=True,
                production_capacity_admitted=False, harness_sha256=harness_sha256 or sha(__file__))


def verify_resources(runtime, *, harness_sha256=None):
    from pipeline.bc_collection import run_continuous as runner
    original = runner.build_capacity
    runner.build_capacity = lambda **kwargs: instrument_capacity(original(**kwargs), harness_sha256=harness_sha256)
    try:
        path = Path(runtime) / "capacity-receipt.json"
        return runner.read_capacity(path, sha(path))
    finally:
        runner.build_capacity = original


def legacy_view_order(runtime, proof_path, capture, launch):
    """Recover omitted v1 dict order from independent config/source, NEVER NPZ.

    The only admitted legacy capture helper is immutable and hash-pinned. Its
    source manifest, native launch, port, independently saved Hydra config and
    pinned original experiment must agree. No output array influences order.
    """
    import yaml
    from pipeline.bc_collection.run_continuous import canonical, digest
    proof = read(proof_path)
    if (set(proof) != {"schema", "capture_helper", "source_manifest", "reference_config", "runtime_config"}
            or proof["schema"] != "roborender.bc-legacy-capture-view-order.v1"
            or capture["harness_sha256"] != LEGACY_CAPTURE_PIN):
        raise ValueError("explicit known legacy capture/view-order proof required")
    def external(item):
        path = Path(item["path"])
        if not path.is_absolute() or item["bytes"] > 32 << 20:
            raise ValueError("bounded absolute independent source proof required")
        return checked(path.parent, dict(item, path=path.name))
    helper = external(proof["capture_helper"])
    manifest_path = external(proof["source_manifest"])
    config_path = external(proof["reference_config"])
    manifest = read(manifest_path)
    rows = manifest.get("files", [])
    execution = launch["collector_execution"]
    command = launch["command"]
    if (sha(helper) != LEGACY_CAPTURE_PIN or sha(config_path) != COLLECT_CONFIG_PIN
            or sha(manifest_path) != execution["source_manifest_sha256"]
            or manifest.get("schema") != "roborender.source-snapshot.v1"
            or manifest.get("file_count") != len(rows)
            or digest(canonical(rows)) != manifest.get("source_sha256")
            or manifest["source_sha256"] != execution["source_sha256"]
            or len([r for r in rows if r["path"] == SELF and r["sha256"] == LEGACY_CAPTURE_PIN]) != 1
            or command.count("ROBORENDER_BC_SAME_INPUT_SHA256=" + LEGACY_CAPTURE_PIN) != 1
            or command.count("+exp=loco_manip/wsdpt_videogen_collect") != 1
            or any("conditioning_views" in token for token in command)):
        raise ValueError("legacy source/config/launch binding differs")
    if proof["runtime_config"]["path"] != "hydra/.hydra/config.yaml":
        raise ValueError("actual saved native Hydra configuration required")
    runtime_config = checked(runtime, proof["runtime_config"])
    port_path = runtime / "overlay/port-manifest.json"
    if sha(port_path) != execution["port_manifest_sha256"]:
        raise ValueError("native port source binding differs")
    port = read(port_path)
    simulator = next(r for r in port["files"] if r["path"] ==
        "pipeline/GR00T-VisualSim2Real/gr00t/rl/simulator/isaacsim/isaacsim.py")
    checked(runtime / "overlay", dict(path=simulator["path"],
        sha256=simulator["derived_sha256"], bytes=simulator["bytes"]))
    if simulator["source_sha256"] != SIMULATOR_PIN:
        raise ValueError("unknown native conditioning dictionary construction")
    source_views = yaml.safe_load(config_path.read_bytes())["simulator"]["config"]["cameras"]["conditioning_views"]
    actual_views = yaml.safe_load(runtime_config.read_bytes())["simulator"]["config"]["cameras"]["conditioning_views"]
    order = [row["name"] for row in source_views]
    if actual_views != source_views or len(order) != 3 or set(order) != set(VIEWS):
        raise ValueError("native saved configuration and pinned conditioning order differ")
    return order, dict(schema=proof["schema"], proof_sha256=sha(proof_path),
        evidence=proof, view_order=order, inferred_from_output=False,
        config_captured_in_original_capture_receipt=False)


def reference_writer(path):
    import numpy as np
    import torch
    path = Path(path)
    if sha(path) != PIN:
        raise ValueError("original reference writer hash differs")
    cls = next(n for n in ast.parse(path.read_bytes()).body
               if isinstance(n, ast.ClassDef) and n.name == "TRLDistillTrainerVideoGen")
    names = {"_save_collected_chunks", "_depth_to_control_video", "_h5v4_control", "_resize_seq"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise ValueError("original conversion/writer methods incomplete")
    node = ast.ClassDef(name="Reference", bases=[], keywords=[], decorator_list=[], body=methods)
    namespace = dict(np=np, torch=torch, os=os, VIDEO_MODEL_CHUNK=81, logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Reference"]


def compare(runtime, reference_path, output, *, legacy_order_proof=None):
    """Independent CPU replay of exact recorded tensors; no simulator imports."""
    import numpy as np
    import torch

    runtime, output = Path(runtime).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.resolve() != output or output.is_relative_to(runtime / "bc_isaac"):
        raise ValueError("fresh separate reference-output directory required")
    capture = read(runtime / "bc-parity-inputs/capture.json")
    request = validate_request(capture["request"])
    capture_sha = LEGACY_CAPTURE_PIN if legacy_order_proof is not None else sha(__file__)
    if (capture.get("schema") != SCHEMA or capture.get("qualification_only") is not True
            or capture.get("production_admitted") is not False or capture.get("physics_replayed") is not False
            or capture.get("harness_sha256") != capture_sha or capture.get("original_writer_sha256") != PIN
            or len(capture["windows"]) != request["max_windows"]):
        raise ValueError("capture identity or exact bounded window count differs")
    native_path = checked(runtime, capture["native"])
    native = read(native_path)
    from pipeline.bc_collection.run_continuous import validate_completion
    validate_completion(native, request, capture["request_sha256"])
    execution, launch, host = (read(runtime / name) for name in (
        "continuous-execution.json", "launch.json", "host-result.json"))
    if (execution.get("verdict") != "PASS" or execution.get("qualification_only") is not True
            or execution.get("native_manifest_sha256") != sha(native_path)
            or execution.get("launch_sha256") != sha(runtime / "launch.json")
            or execution.get("host_result_sha256") != sha(runtime / "host-result.json")
            or execution.get("request_sha256") != capture["request_sha256"]
            or launch.get("request") != request or host.get("passed") is not True
            or host.get("exit_code") != 0 or host.get("failure") is not None
            or "trainer._target_=bc_native_same_input.CapturedContinuousBCTrainer" not in launch.get("command", [])):
        raise ValueError("actual completed native launch/host/execution chain required")
    recovered_order, order_proof = (legacy_view_order(runtime, legacy_order_proof, capture, launch)
        if legacy_order_proof is not None else (None, None))
    resource = verify_resources(runtime, harness_sha256=capture_sha)
    if (resource.get("schema") != CAPACITY_SCHEMA or resource.get("verdict") != "PASS"
            or resource.get("harness_sha256") != capture_sha
            or resource.get("production_capacity_admitted") is not False):
        raise ValueError("actual instrumented resource qualification required")
    owner = reference_writer(reference_path)()
    output.mkdir(parents=True, exist_ok=False)
    owner._ep_buf, owner._ep_seq_base, owner._vg_saved_chunks = None, None, 0
    owner.vg_collect_dir = str(output)
    compared, signature = [], None
    for index, item in enumerate(capture["windows"], 1):
        emitted = read(checked(runtime, item))
        metadata_path = checked(runtime, emitted["inputs"])
        metadata = read(metadata_path)
        if (metadata.get("window") != index or metadata.get("num_envs") != len(request["motions"])
                or metadata.get("original_writer_sha256") != PIN):
            raise ValueError("recorded input sequence/source/topology differs")
        view_order = metadata.get("view_order", recovered_order)
        if (not isinstance(view_order, list) or len(view_order) != 3 or set(view_order) != set(VIEWS)
                or (recovered_order is not None and view_order != recovered_order)):
            raise ValueError("captured or independently proven conditioning-view order required")
        current = (metadata["attrs"], metadata["observation_keys"], metadata["run_tag"], view_order)
        if signature is not None and signature != current:
            raise ValueError("writer configuration changed between windows")
        signature = current
        arrays = {}
        for key, entry in metadata["arrays"].items():
            array = np.load(checked(metadata_path.parent, entry), allow_pickle=False)
            if list(array.shape) != entry["shape"] or array.dtype.str != entry["dtype"]:
                raise ValueError("captured tensor shape/dtype differs")
            arrays[key] = torch.from_numpy(array)
        owner.__dict__.update(metadata["attrs"])
        owner._vg_run_tag = metadata["run_tag"]
        owner.algo_obs_dim_dict = dict.fromkeys(metadata["observation_keys"])
        owner._depth_chunk = list(arrays["primary_depth"])
        owner._robot_rgb_chunk = list(arrays["primary_robot"])
        owner._views_chunk = [{view: dict(depth=arrays[view + "__depth"][t],
            robot_rgb=arrays[view + "__robot_rgb"][t], rgb=None)
            for view in view_order} for t in range(81)]
        owner.storage = SimpleNamespace(**{key[5:]: value for key, value in arrays.items() if key.startswith("obs__")},
            gt_actions=arrays["gt_actions"], dones=arrays["dones"])
        first = owner._vg_saved_chunks
        owner._save_collected_chunks()
        if (owner._vg_saved_chunks - first != len(emitted["chunks"])
                or [len(buf.get("gt", [])) for buf in owner._ep_buf] != emitted["native_tail_rows"]
                or owner._ep_seq_base != emitted["native_sequence"]):
            raise AssertionError("reference emission/tail/sequence differs from native adapter")
        for offset, expected in enumerate(emitted["chunks"], first):
            original = checked(runtime, expected)
            replayed = output / f"chunk_{owner._vg_run_tag}_{offset:07d}.npz"
            if replayed.stat().st_size != original.stat().st_size or sha(replayed) != expected["sha256"]:
                raise AssertionError("same-input reference NPZ bytes differ from actual native adapter")
            with np.load(original, allow_pickle=False) as before, np.load(replayed, allow_pickle=False) as after:
                if before.files != after.files:
                    raise AssertionError("same-input payload field order differs")
                for key in before.files:
                    if before[key].dtype != after[key].dtype or not np.array_equal(before[key], after[key]):
                        raise AssertionError("same-input payload field/dtype differs: " + key)
            compared.append(dict(actual=expected, reference=record(replayed, output)))
        del arrays
    actual_files = sorted(p.name for p in (runtime / "bc_isaac").glob("chunk_*.npz"))
    if actual_files != sorted(Path(row["actual"]["path"]).name for row in compared) or not compared:
        raise AssertionError("missing/extra native chunks or no emitted payload to compare")
    result = dict(schema=SCHEMA, verdict="PASS", comparison_scope="exact_native_writer_inputs_only",
        windows=request["max_windows"], num_envs=len(request["motions"]), chunks=len(compared),
        capture=record(runtime / "bc-parity-inputs/capture.json", runtime),
        execution=record(runtime / "continuous-execution.json", runtime),
        resources=record(runtime / "capacity-receipt.json", runtime),
        original_writer_sha256=PIN, harness_sha256=capture_sha, comparator_sha256=sha(__file__),
        comparison_method_version="conditioning_insertion_order_v2", legacy_order_proof=order_proof,
        outputs=compared,
        independent_native_trajectory_tested=False, physics_replayed=False, production_admitted=False)
    publish(output / "comparison.json", result)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "run":
        return run(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["compare"])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reference-writer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-order-proof", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.runtime, args.reference_writer, args.output,
                             legacy_order_proof=args.legacy_order_proof)))


if __name__ == "__main__":
    # Direct-file invocation from a frozen source tree, not a mutable overlay.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    main()
