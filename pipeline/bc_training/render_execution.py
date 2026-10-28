"""Bind genuine worker execution to existing activation hashes; no NPZ rescans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.bc_training.production_protocol import bound, canonical, read_metadata, sha, verify_render_asset_identity
from pipeline.bc_training.render_asset_identity import file_identity

RECIPE = {"render_recipe": "raven_3view_50step_independent_v2", "num_inference_steps": 50,
          "cfg_scale": 5.0, "seed": 42, "height": 240, "width": 416,
          "student_height": 108, "student_width": 192, "max_chain_chunks": 1}


def verify_worker(config, worker_path, *, loaded_paths=None, loaded_modules=None, tokenizer_path=None):
    binding = config["execution_identity"]
    protocol = bound(binding)
    verify_render_asset_identity(protocol)
    receipt = bound(protocol["render_asset_identity"])
    records = {row["role"]: row for row in receipt["files"]}
    if any(config.get(key) != value for key, value in RECIPE.items()):
        raise ValueError("executed renderer recipe differs from the pinned BC recipe")
    def same(path, role):
        if str(Path(path).resolve(strict=True)) != records[role]["path"]:
            raise ValueError("loaded renderer asset/source path differs: " + role)
    same(config["adapter_path"], "adapter")
    same(config["prompt_file"], "prompt")
    same(worker_path, "source/pipeline/bc_rendering/render_bc_teacher_worker.py")
    for role in records:
        if role.startswith("source/diffsynth/") or role.startswith("source/sim2realgen/"):
            same(Path(config["source_path"]) / role.removeprefix("source/"), role)
        elif role.startswith("source/pipeline/bc_rendering/"):
            same(Path(worker_path).parent / role.removeprefix("source/pipeline/bc_rendering/"), role)
    if loaded_paths is not None:
        actual = {str(Path(p).resolve(strict=True)) for p in loaded_paths}
        expected = {row["path"] for role, row in records.items() if role.startswith("foundation/")}
        if actual != expected:
            raise ValueError("actual loaded ModelConfig files differ from attested foundation")
        if not loaded_modules or not tokenizer_path:
            raise ValueError("loaded source/tokenizer evidence is missing")
        same(loaded_modules["diffsynth"], "source/diffsynth/__init__.py")
        same(loaded_modules["sim2realgen.inference_long_video"], "source/sim2realgen/inference_long_video.py")
        token_records = {role: row for role, row in records.items() if role.startswith("source/tokenizer/")}
        if not token_records:
            raise ValueError("tokenizer asset identity missing")
        actual_token_files = {str(p.resolve()) for p in Path(tokenizer_path).rglob("*") if p.is_file()}
        if actual_token_files != {row["path"] for row in token_records.values()}:
            raise ValueError("actual tokenizer files differ from attested tokenizer")
    return {"protocol_raw_sha256": binding["raw_sha256"], "asset_receipt_raw_sha256": protocol["render_asset_identity"]["raw_sha256"],
            "model_sha256": protocol["render_model_sha256"], "loaded_paths_verified": loaded_paths is not None}


def complete_lane(plan, identity, *, rendered_count, skipped_count):
    if not identity.get("loaded_paths_verified") or skipped_count or rendered_count != len(plan["jobs"]):
        raise ValueError("only actually rendered, fully completed fresh lanes can emit execution evidence")
    outputs = []
    prompt = read_metadata(plan["config"]["prompt_file"]).decode().strip()
    for job in plan["jobs"]:
        if job["seed"] != 42 or job["frame_count"] != 81 or job["chain_length"] != 1 or job.get("prompt") != prompt:
            raise ValueError("worker job recipe/sequence length changed")
        path = Path(job["output_npz"]).resolve(strict=True)
        outputs.append(dict(path=str(path), source_sha256=job["source_payload_sha256"],
                            file_identity=file_identity(path.stat())))
    return dict(schema="roborender.bc-render-lane-execution.v1", verdict="PASS", identity=identity,
        lane_plan_canonical_sha256=sha(canonical(plan)), physical_gpu=plan["physical_gpu"], lane=plan["lane"],
        recipe=RECIPE, outputs=outputs)


def _verified_outputs(protocol_binding, protocol, render_plan_path):
    """Read original worker plans/receipts without rewriting their identities."""
    plan = json.loads(read_metadata(render_plan_path))
    if plan.get("status") != "complete" or plan["config"].get("execution_identity") != protocol_binding:
        raise ValueError("genuine complete identity-bound renderer plan required")
    produced = {}
    for path in plan["lane_plan_paths"]:
        lane_plan = json.loads(read_metadata(path))
        if not lane_plan["jobs"]:
            continue
        receipt_path = Path(lane_plan["progress_path"]).with_suffix(".execution.json")
        receipt = json.loads(read_metadata(receipt_path))
        expected_identity = dict(protocol_raw_sha256=protocol_binding["raw_sha256"],
            asset_receipt_raw_sha256=protocol["render_asset_identity"]["raw_sha256"],
            model_sha256=protocol["render_model_sha256"], loaded_paths_verified=True)
        if (receipt.get("schema") != "roborender.bc-render-lane-execution.v1" or receipt.get("verdict") != "PASS"
                or receipt.get("identity") != expected_identity or receipt.get("recipe") != RECIPE
                or receipt.get("physical_gpu") != lane_plan["physical_gpu"] or receipt.get("lane") != lane_plan["lane"]
                or receipt.get("lane_plan_canonical_sha256") != sha(canonical(lane_plan))
                or len(receipt.get("outputs", [])) != len(lane_plan["jobs"])):
            raise ValueError("worker execution receipt differs from the actual pinned lane")
        expected_outputs = [(str(Path(job["output_npz"]).resolve(strict=True)), job["source_payload_sha256"])
                            for job in lane_plan["jobs"]]
        if [(row["path"], row["source_sha256"]) for row in receipt["outputs"]] != expected_outputs:
            raise ValueError("worker ordered output/source mapping differs from its actual lane plan")
        for row in receipt["outputs"]:
            if row["path"] in produced or file_identity(Path(row["path"]).stat()) != row["file_identity"]:
                raise ValueError("rendered output was replaced/changed after worker completion")
            produced[row["path"]] = row
    return produced


def _finish_receipt(protocol, rendered_manifest_path, produced):
    raw = read_metadata(rendered_manifest_path)
    rendered = json.loads(raw)
    if (rendered.get("parent_manifest_sha256") != protocol["source_pool"]["raw_sha256"]
            or rendered.get("validation", {}).get("verdict") != "PASS"):
        raise ValueError("activated manifest is not validated against the pinned source pool")
    rows = rendered["chunks"]
    if len(produced) != len(rows) or len({r["chunk_id"] for r in rows}) != len(rows):
        raise ValueError("worker output population differs from activated pool")
    for row in rows:
        output = str((Path(rendered["rendered_pool"]) / Path(row["rendered_npz"]).name).resolve())
        if output not in produced or produced[output]["source_sha256"] != row["source_sha256"]:
            raise ValueError("worker source/output identity differs from activation")
    return dict(schema="roborender.bc-independent-render-execution.v1", verdict="PASS",
        rendered_manifest_raw_sha256=sha(raw), source_pool_raw_sha256=protocol["source_pool"]["raw_sha256"],
        recipe=RECIPE["render_recipe"], num_inference_steps=50, cfg_scale=5.0, seed=42,
        model_sha256=protocol["render_model_sha256"],
        chunks={row["chunk_id"]: {k: row[k] for k in ("source_sha256", "rendered_sha256")} for row in rows})


def finish(protocol_binding, render_plan_path, rendered_manifest_path):
    protocol = bound(protocol_binding)
    verify_render_asset_identity(protocol)
    return _finish_receipt(protocol, rendered_manifest_path,
        _verified_outputs(protocol_binding, protocol, render_plan_path))


def finish_streamed(protocol_binding, render_plan_bindings, rendered_manifest_path):
    """Union genuine cohort workers; final source pool did not exist at render time.

    Each worker keeps its original asset-only execution identity. All must match
    the final protocol's exact renderer bindings; no simulated merged worker is
    created. Exact output coverage and unchanged file identities remain required.
    """
    protocol = bound(protocol_binding)
    verify_render_asset_identity(protocol)
    keys = ('renderer_registry', 'renderer_source_manifest', 'render_asset_identity', 'render_model_sha256')
    produced, seen, evidence = {}, set(), []
    if not render_plan_bindings:
        raise ValueError('streamed render requires completed worker plans')
    for plan_binding in render_plan_bindings:
        plan = bound(plan_binding)
        if plan_binding['raw_sha256'] in seen:
            raise ValueError('duplicate streamed render plan')
        seen.add(plan_binding['raw_sha256'])
        original_binding = plan['config']['execution_identity']
        original = bound(original_binding)
        if any(original.get(k) != protocol.get(k) for k in keys):
            raise ValueError('streamed renderer asset/source identity differs')
        rows = _verified_outputs(original_binding, original, plan_binding['path'])
        if produced.keys() & rows.keys():
            raise ValueError('duplicate output across streamed render workers')
        produced.update(rows)
        evidence.append({'plan': plan_binding, 'execution_identity': original_binding})
    result = _finish_receipt(protocol, rendered_manifest_path, produced)
    result['streamed_worker_plans'] = evidence
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--render-plan", type=Path, required=True)
    parser.add_argument("--rendered-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finish({"path": str(args.protocol), "raw_sha256": args.protocol_sha256}, args.render_plan, args.rendered_manifest)
    with args.output.open("xb") as stream:
        stream.write(canonical(result))
    print(json.dumps({"path": str(args.output), "raw_sha256": sha(canonical(result)), "payloads_reread": False}))


if __name__ == "__main__":
    main()
