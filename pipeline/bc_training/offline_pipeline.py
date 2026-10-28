"""Bounded offline render -> split admission -> BC training on one owned H200 job.

The native source pool must already exist. No Isaac, collection, test-policy
execution, allocation, or credential copying occurs here. The default is a
plan; --execute is explicit. A failed stage is retained, never silently replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def plan(*, dataset_root, asset_root, runtime_root, run_name, gpus=(0,),
         max_epochs=100, min_epochs=10, patience=8, qualification=False,
         source_root=ROOT, production_protocol=None, production_protocol_sha256=None,
         render_execution_receipt=None, render_execution_receipt_sha256=None, mode=None):
    if mode is not None:
        from pipeline.deployment.run_mode import recipe
        selected = recipe(mode)['bc']
        qualification = mode == 'smoke'
        max_epochs, min_epochs, patience = (selected[k] for k in ('max_epochs', 'min_epochs', 'patience'))
    dataset, assets, runtime, source = map(Path, (dataset_root, asset_root, runtime_root, source_root))
    if any(not p.is_absolute() or ".." in p.parts for p in (dataset, assets, runtime, source)):
        raise ValueError("all roots must be absolute and normalized")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,95}", run_name)
            or tuple(gpus) not in ((0,), (0, 1, 2, 3))
            or any(type(n) is not int or n < 1 for n in (max_epochs, min_epochs, patience))
            or max_epochs < min_epochs):
        raise ValueError("invalid run identity, bounded GPU set, or stopping configuration")
    if qualification and not run_name.startswith("smoke-"):
        raise ValueError("bounded qualification must be visibly named smoke")
    if not qualification and (min_epochs < 10 or patience < 8):
        raise ValueError("production stopping must not use diagnostic patience")
    if not qualification and (production_protocol is None or production_protocol_sha256 is None):
        raise ValueError("production requires an explicit pinned full-population protocol")
    if qualification and any(x is not None for x in (production_protocol, production_protocol_sha256,
                                                     render_execution_receipt, render_execution_receipt_sha256)):
        raise ValueError("qualification cannot borrow production evidence")
    manifest = dataset / "manifests/independent_v2.json"
    splits = dataset / "splits_independent_v2"
    admission = splits / "data-admission.json"
    renderer_python = assets / "renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python"
    def script(relative):
        return str(source / relative)
    stages = [
        {"name": "render_50step_teacher", "argv": [str(renderer_python),
            script("pipeline/bc_rendering/render_bc_teacher_8gpu.py"),
            "--gpus", ",".join(map(str, gpus)), "--asset-root", str(assets),
            "--source-root", str(dataset / "isaac"), "--pool-manifest", str(dataset / "manifest.json"),
            "--pool-seal", str(dataset / "SEALED.json"),
            "--output-root", str(dataset / "rendered_independent_v2"), "--run-dir", str(runtime / "render")]},
        {"name": "activate_disjoint_splits", "argv": [str(renderer_python),
            script("pipeline/bc_rendering/activate_corrected_render_pool.py"),
            "--dataset-root", str(dataset), "--pool-seal", str(dataset / "SEALED.json"),
            "--report", str(runtime / "activation.json")]},
        {"name": "metadata_data_admission", "argv": [str(renderer_python),
            script("pipeline/bc_training/data_admission.py"),
            "--source-manifest", str(manifest), "--split-dir", str(splits),
            "--output", str(admission), "--require-independent-render",
            "--purpose", "qualification" if qualification else "production", "--run-name", run_name]},
        {"name": "bc_train_and_validation_mse", "argv": ["/bin/bash",
            script("pipeline/bc_training/train_raven_bc_h200.sh"),
            "--run-name", run_name, "--wandb-run-name", run_name,
            "--max-epochs", str(max_epochs), "--min-epochs", str(min_epochs),
            "--patience", str(patience), "--seed", "171015"]},
    ]
    if not qualification:
        if (render_execution_receipt is None) != (render_execution_receipt_sha256 is None):
            raise ValueError("an existing render receipt requires its raw SHA256")
        stages[0]["argv"] += ["--execution-identity", str(production_protocol),
                              "--execution-identity-sha256", production_protocol_sha256]
        admission_stage = stages[2]
        if render_execution_receipt is None:
            render_execution_receipt = runtime / "render-execution.json"
            render_execution_receipt_sha256 = "RESOLVE_ACTUAL_RENDER_EXECUTION_SHA256"
            stages.insert(2, {"name": "bind_render_execution", "argv": [str(renderer_python),
                script("pipeline/bc_training/render_execution.py"), "--protocol", str(production_protocol),
                "--protocol-sha256", production_protocol_sha256, "--render-plan", str(runtime / "render/plan.json"),
                "--rendered-manifest", str(manifest), "--output", str(render_execution_receipt)]})
        admission_stage["argv"] += ["--protocol", str(production_protocol), "--protocol-sha256", production_protocol_sha256,
            "--render-execution-receipt", str(render_execution_receipt),
            "--render-execution-receipt-sha256", render_execution_receipt_sha256]
    return {"schema": "roborender.bc-offline-pipeline.v1", "run_name": run_name,
        "qualification": qualification, "source_root": str(source),
        "production_protocol": None if production_protocol is None else str(production_protocol),
        "production_protocol_sha256": production_protocol_sha256,
        "dataset_root": str(dataset), "runtime_root": str(runtime), "gpus": list(gpus),
        "training_checkpoint_dir": str(assets / "checkpoints/students/BC" / run_name),
        "test_policy_executed": False,
        "env": {"ROBORENDER_ASSET_ROOT": str(assets), "DATASET_ROOT": str(dataset),
                "TRAIN_SPLIT": str(splits / "train.json"),
                "VALIDATION_SPLIT": str(splits / "validation.json"),
                "DATA_ADMISSION_JSON": str(admission), "REQUIRE_PRODUCTION_DATA_ADMISSION": "1",
                "ALLOW_QUALIFICATION_DATA": "1" if qualification else "0",
                "CHUNK_BATCH_SIZE": "16", "WANDB_MODE": "online", "PYTHONUNBUFFERED": "1"},
        "stages": stages}


def publish(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def execute(job, *, runner=subprocess.run):
    runtime = Path(job["runtime_root"])
    dataset = Path(job["dataset_root"])
    # Reject incomplete or historical inputs before any GPU process is started.
    source_pool = json.loads((dataset / "manifest.json").read_bytes())
    seal = json.loads((dataset / "SEALED.json").read_bytes())
    if (source_pool.get("schema") != "roborender.bc_pool.native-cohorts.v1"
            or source_pool.get("status") != "sealed" or seal.get("verdict") != "PASS"
            or source_pool.get("partition_counts", {}).get("train", 0) % 16
            or source_pool.get("partition_counts", {}).get("train", 0) < 16):
        raise ValueError("fresh complete native pool with full B16 training batches is required")
    if not job["qualification"]:
        from pipeline.bc_training.production_protocol import validate_protocol_inputs, bound
        protocol, _, _, _ = validate_protocol_inputs(job["production_protocol"], job["production_protocol_sha256"])
        if bound(protocol["source_pool"]) != source_pool:
            raise ValueError("production pool differs from frozen expected population")
    if runtime.exists() or Path(job["training_checkpoint_dir"]).exists():
        raise FileExistsError("offline stage identity exists; inspect or use explicit checkpoint resume")
    runtime.mkdir(parents=True)
    publish(runtime / "plan.json", job)
    started = time.monotonic()
    for index, stage in enumerate(job["stages"]):
        now = time.monotonic()
        event = {"stage": stage["name"], "index": index, "status": "running",
                 "elapsed_seconds": now - started}
        publish(runtime / "progress.json", event)
        print(json.dumps(event), flush=True)
        try:
            argv = list(stage["argv"])
            if "RESOLVE_ACTUAL_RENDER_EXECUTION_SHA256" in argv:
                from pipeline.bc_training.production_protocol import read_metadata
                receipt_path = Path(argv[argv.index("--render-execution-receipt") + 1])
                receipt_sha = hashlib.sha256(read_metadata(receipt_path)).hexdigest()
                argv[argv.index("RESOLVE_ACTUAL_RENDER_EXECUTION_SHA256")] = receipt_sha
                event["render_execution_raw_sha256"] = receipt_sha
            result = runner(argv, cwd=job["source_root"],
                            env=dict(os.environ, **job["env"]), check=False)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, stage["argv"])
        except BaseException as exc:
            event.update(status="failed", seconds=time.monotonic() - now, error_type=type(exc).__name__)
            publish(runtime / "progress.json", event)
            print(json.dumps(event), flush=True)
            raise
        event.update(status="complete", seconds=time.monotonic() - now)
        publish(runtime / f"stage-{index:02d}.json", event)
        print(json.dumps(event), flush=True)
    final = {"status": "complete", "seconds": time.monotonic() - started,
             "checkpoint_dir": job["training_checkpoint_dir"], "test_policy_executed": False}
    publish(runtime / "progress.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "production"),
                        help="Shared pipeline recipe; overrides epoch/patience defaults")
    parser.add_argument("--production-protocol", type=Path)
    parser.add_argument("--production-protocol-sha256")
    parser.add_argument("--render-execution-receipt", type=Path)
    parser.add_argument("--render-execution-receipt-sha256")
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    run = args.pop("execute")
    args["gpus"] = tuple(int(n) for n in args["gpus"].split(","))
    job = plan(**args)
    print(json.dumps(execute(job) if run else job, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
