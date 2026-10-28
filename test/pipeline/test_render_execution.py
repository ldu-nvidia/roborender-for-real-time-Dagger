"""CPU-only worker/activation receipt contract using small fake files."""
from test.deployment.source_paths import PIPELINE_ROOT
import json
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline.bc_training import render_execution as execution
from pipeline.bc_training.render_asset_identity import produce, sha


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(execution.canonical(doc))
    return {"path": str(path), "raw_sha256": sha(path.read_bytes())}


def setup(tmp_path):
    renderer = tmp_path / "renderer"
    source = renderer / "source"
    worker = tmp_path / "code/pipeline/bc_rendering/render_bc_teacher_worker.py"
    paths = [renderer / "base/weights", renderer / "adapter", source / "diffsynth/__init__.py",
        source / "sim2realgen/inference_long_video.py", renderer / "tokenizer/vocab", worker,
        tmp_path / "prompt"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw")
    registry = {"schema": "roborender.renderer_registry.v1", "artifacts": [
        dict(id="wan21_fun_control_1p3b_foundation", relative_root="base",
             files=[dict(path="weights", sha256=sha(b"raw"), bytes=3)]),
        dict(id="roborender_offline_bc_video_teacher_50step_lora_step05294", relative_path="adapter", sha256=sha(b"raw"), bytes=3)]}
    registry_binding = put(tmp_path / "registry.json", registry)
    bindings = produce(registry_path=registry_binding["path"], renderer_root=renderer, source_files={
        "diffsynth/__init__.py": paths[2], "sim2realgen/inference_long_video.py": paths[3],
        "tokenizer/vocab": paths[4], "pipeline/bc_rendering/render_bc_teacher_worker.py": worker},
        prompt_path=paths[6], output_dir=tmp_path / "assets")
    source_pool = put(tmp_path / "source-pool.json", {})
    protocol = put(tmp_path / "protocol.json", {**bindings, "source_pool": source_pool})
    config = dict(execution.RECIPE, execution_identity=protocol, source_path=str(source),
        adapter_path=str(paths[1]), prompt_file=str(paths[6]))
    identity = execution.verify_worker(config, worker, loaded_paths=[paths[0]], loaded_modules={
        "diffsynth": paths[2], "sim2realgen.inference_long_video": paths[3]}, tokenizer_path=renderer / "tokenizer")
    output = tmp_path / "rendered/chunk.npz"
    output.parent.mkdir()
    output.write_bytes(b"not-a-real-npz")
    lane = dict(config=config, physical_gpu=0, lane=0, progress_path=str(tmp_path / "progress/lane.json"),
        jobs=[dict(output_npz=str(output), source_payload_sha256="a" * 64, seed=42, frame_count=81, chain_length=1, prompt="raw")])
    lane_binding = put(tmp_path / "lane-plan.json", lane)
    receipt = execution.complete_lane(lane, identity, rendered_count=1, skipped_count=0)
    receipt_path = Path(lane["progress_path"]).with_suffix(".execution.json")
    put(receipt_path, receipt)
    plan = put(tmp_path / "render-plan.json", dict(status="complete", config=config, lane_plan_paths=[lane_binding["path"]]))
    manifest = put(tmp_path / "activated.json", dict(parent_manifest_sha256=source_pool["raw_sha256"],
        validation={"verdict": "PASS"}, rendered_pool=str(output.parent),
        chunks=[dict(chunk_id="id-0", rendered_npz="rendered/chunk.npz", source_sha256="a" * 64,
                     rendered_sha256=sha(output.read_bytes()))]))
    return dict(protocol=protocol, plan=plan, manifest=manifest, lane=lane, receipt=receipt, receipt_path=receipt_path,
                output=output, identity=identity, config=config, worker=worker, paths=paths)


def test_actual_worker_identity_to_activated_receipt_without_npz_reads(tmp_path, monkeypatch):
    data = setup(tmp_path)
    original = Path.open
    def bounded(path, *args, **kwargs):
        assert path.suffix != ".npz", "execution binding reread a payload"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", bounded)
    receipt = execution.finish(data["protocol"], data["plan"]["path"], data["manifest"]["path"])
    assert receipt["verdict"] == "PASS" and receipt["num_inference_steps"] == 50
    assert set(receipt["chunks"]) == {"id-0"}


def test_actual_loaded_paths_not_merely_declared_foundation(tmp_path):
    data = setup(tmp_path)
    with pytest.raises(ValueError, match="actual loaded"):
        execution.verify_worker(data["config"], data["worker"], loaded_paths=[data["paths"][1]])


def test_streamed_asset_bound_receipt_preserves_original_worker_identity(tmp_path, monkeypatch):
    data = setup(tmp_path)
    final = execution.bound(data['protocol'])
    final['final_pool_only'] = True
    final_binding = put(tmp_path / 'final-protocol.json', final)
    original = Path.open
    def bounded(path, *args, **kwargs):
        assert path.suffix != '.npz', 'stream aggregation reread a payload'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', bounded)
    result = execution.finish_streamed(final_binding, [data['plan']], data['manifest']['path'])
    assert result['verdict'] == 'PASS'
    assert result['streamed_worker_plans'][0]['execution_identity'] == data['protocol']


def test_streamed_two_actual_workers_require_exact_full_population(tmp_path):
    data = setup(tmp_path)
    second = data['output'].with_name('second.npz')
    second.write_bytes(b'second-small-test-payload')
    lane = deepcopy(data['lane'])
    lane.update(lane=1, progress_path=str(tmp_path/'progress/second.json'))
    lane['jobs'][0]['output_npz'] = str(second)
    lane_binding=put(tmp_path/'second-lane.json',lane)
    receipt=execution.complete_lane(lane,data['identity'],rendered_count=1,skipped_count=0)
    put(Path(lane['progress_path']).with_suffix('.execution.json'),receipt)
    second_plan=put(tmp_path/'second-plan.json',dict(status='complete',config=data['config'],lane_plan_paths=[lane_binding['path']]))
    manifest=execution.bound(data['manifest'])
    manifest['chunks'].append(dict(chunk_id='id-1',rendered_npz='rendered/second.npz',
        source_sha256='a'*64,rendered_sha256=sha(second.read_bytes())))
    complete=put(tmp_path/'complete.json',manifest)
    result=execution.finish_streamed(data['protocol'],[data['plan'],second_plan],complete['path'])
    assert set(result['chunks']) == {'id-0','id-1'}
    with pytest.raises(ValueError,match='population'):
        execution.finish_streamed(data['protocol'],[data['plan']],complete['path'])


@pytest.mark.parametrize('mutation', ['empty', 'duplicate', 'asset', 'changed_plan', 'changed_output'])
def test_streamed_rejects_incomplete_duplicate_or_changed_evidence(tmp_path, mutation):
    data = setup(tmp_path)
    plans = [data['plan']]
    final = execution.bound(data['protocol'])
    if mutation == 'empty': plans = []
    elif mutation == 'duplicate': plans *= 2
    elif mutation == 'asset': final['renderer_source_manifest'] = {'path': '/missing', 'raw_sha256': '0'*64}
    elif mutation == 'changed_plan': Path(data['plan']['path']).write_text('{}')
    elif mutation == 'changed_output': data['output'].write_bytes(b'changed')
    final_binding = put(tmp_path / 'final-protocol.json', final)
    with pytest.raises((ValueError, FileNotFoundError)):
        execution.finish_streamed(final_binding, plans, data['manifest']['path'])


@pytest.mark.parametrize("mutation", ["skipped", "not_loaded", "changed_output", "wrong_recipe", "missing_worker", "wrong_gpu", "wrong_lane", "wrong_source"])
def test_missing_or_changed_execution_fails_closed(tmp_path, mutation):
    data = setup(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        if mutation == "skipped":
            execution.complete_lane(data["lane"], data["identity"], rendered_count=0, skipped_count=1)
        elif mutation == "not_loaded":
            execution.complete_lane(data["lane"], {**data["identity"], "loaded_paths_verified": False}, rendered_count=1, skipped_count=0)
        else:
            if mutation == "changed_output": data["output"].write_bytes(b"changed")
            elif mutation == "missing_worker": data["receipt_path"].unlink()
            else:
                if mutation == "wrong_recipe": data["receipt"]["recipe"] = {**data["receipt"]["recipe"], "cfg_scale": 1.0}
                elif mutation == "wrong_gpu": data["receipt"]["physical_gpu"] = 1
                elif mutation == "wrong_lane": data["receipt"]["lane"] = 1
                else: data["receipt"]["outputs"][0]["source_sha256"] = "b" * 64
                put(data["receipt_path"], data["receipt"])
            execution.finish(data["protocol"], data["plan"]["path"], data["manifest"]["path"])


def test_worker_main_imports_completion_producer_in_its_own_scope():
    path = PIPELINE_ROOT / "bc_rendering/render_bc_teacher_worker.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    imports = [n for n in ast.walk(main) if isinstance(n, ast.ImportFrom)
               and n.module == "pipeline.bc_training.render_execution"]
    assert any(any(a.name == "complete_lane" for a in n.names) for n in imports)
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If) and any(
        isinstance(s, ast.ImportFrom) and any(a.name == "complete_lane" for a in s.names) for s in n.body))
    imported = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.ImportFrom))
    called = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                  and n.value.func.id == "complete_lane")
    assert imported < called
