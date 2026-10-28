"""Real, disposable pre-BC -> BC -> four-round DAgger smoke, CPU orchestrated.

No synthetic observations, no promoted checkpoints, and no automatic replay of
a lost Isaac process. Production's 50-update/256-case contract is untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.deployment.beta_site import Site, validate_site, site_host
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.source_snapshot import verify
from pipeline.deployment.prepare_beta_campaign import prepare
from pipeline.dagger.beta_campaign import make_first_round_config
from pipeline.dagger.services.contracts import file_sha256
from pipeline.evaluation.beta_small_batch_panel import make_panel, canonical, validate_comparison

BC = ROOT / "checkpoints/students/BC/bc_3view_autoregressive_v1_20260914T044707Z/best_validation.pt"
BC_SHA = "053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718"
IMAGE = "sha256:7aa2c5aaf53e55e749c9f63b6430b9554b4ca49928cf35f376a646669da7b509"


class Smoke:
    def __init__(self, site, root, run):
        self.site, self.root, self.run = validate_site(site), root, run
        self.role = "leo" if site_host(site) == "leo-rtx-6k" else "beta"
        self.named_prebc = False
        self.bc_logger = None
        self.started = time.monotonic()
        self.state = {"schema": f"roborender.{self.role}-e2e-smoke.v1", "run_id": run,
                      "production_promoted": False, "phase": "STATIC_ADMISSION"}
        self.phase("STATIC_ADMISSION")

    def phase(self, phase, **values):
        self.state.update(phase=phase, elapsed_seconds=time.monotonic()-self.started, **values)
        atomic(self.root / "status.json", self.state)
        print(json.dumps({"phase": phase, "elapsed_seconds": self.state["elapsed_seconds"], **values}), flush=True)
        if self.bc_logger is not None:
            self.bc_logger.log({"pipeline/phase": phase, "pipeline/elapsed_seconds": self.state["elapsed_seconds"]})

    def allocation(self, site, action, keys=()):
        return site.json([self.site["broker_python"], "-", action, "--run-id", site.run,
            "--round-id", "1", "--gpus", "1",
            *(["--existing-name", "isaac-sim-render-leo"] if self.named_prebc else []),
            *[item for key in keys for item in ("--public-key", key)]], "broker",
            data=(ROOT / "pipeline/deployment/h200/beta_pod.py").read_bytes())

    def release(self, site):
        # Detached work must have stopped before releasing this exact pod.
        for label in ("bc-render", "bc-train"):
            if site.gpu_ssh.exists():
                site.stop_process(label, "gpu")
        if self.allocation(site, "status").get("present"):
            if site.gpu_ssh.exists():
                for _ in range(3):
                    if site.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], "gpu").strip():
                        raise RuntimeError("pre-BC pod still has GPU clients; release not certified")
                    time.sleep(1)
            self.allocation(site, "delete")
            deadline = time.monotonic()+90
            while self.allocation(site, "status").get("present"):
                if time.monotonic()>deadline:
                    raise TimeoutError("pre-BC allocation release unconfirmed")
                time.sleep(3)

    def execute(self, previous_collection=None):
        from pipeline.deployment.smoke_asset_audit import audit
        manifest = json.loads(Path(self.site["source_manifest"]).read_bytes())
        verify(ROOT, manifest)
        atomic(self.root / "static-assets.json", audit(ROOT, "cpu"))
        panel = make_panel(file_sha256(ROOT / "pipeline/dagger/rtx/config/fixed_panel_motion_inventory.v1.json"), smoke=True)
        panel_path = self.root / "panel.json"
        panel_path.write_bytes(canonical(panel))
        config = make_first_round_config(self.run, seed=610121, checkpoint=BC, checkpoint_sha=BC_SHA,
            panel=panel_path, panel_sha=file_sha256(panel_path), source_sha=manifest["source_sha256"],
            gateway_pids=[], smoke=True, deployment_host=site_host(self.site))
        suffix = hashlib.sha256(self.run.encode()).hexdigest()[:16]
        pre_run = ("dagger-leo-smoke-prebc-named-20261013" if self.named_prebc else
                   f"dagger-{self.role}-smoke-prebc-"+suffix)
        pre_config = make_first_round_config(pre_run, seed=610121, checkpoint=BC, checkpoint_sha=BC_SHA,
            panel=panel_path, panel_sha=file_sha256(panel_path), source_sha=manifest["source_sha256"],
            gateway_pids=[], smoke=True, deployment_host=site_host(self.site))
        request = {"run_id": pre_run, "round_id": 1, "config": pre_config,
                   "campaign_root": str(self.root / "prebc"),
                   "baseline_checkpoint_path": str(BC), "baseline_checkpoint_sha256": BC_SHA}
        site = Site(self.site, request)
        if self.named_prebc:
            if self.role != "leo":
                raise ValueError("the named pre-BC allocation is Leo-only")
            site.pod = "isaac-sim-render-leo"
        # Source verification is independent of asset verification.
        code = """import json,sys,hashlib
from pathlib import Path
r=Path(sys.argv[1]);m=json.loads(Path(sys.argv[2]).read_bytes());sys.path.insert(0,str(r))
from pipeline.deployment.source_snapshot import file_record,canonical
assert [file_record(r,x['path']) for x in m['files']]==m['files']
assert hashlib.sha256(canonical(m['files'])).hexdigest()==sys.argv[3]
from pipeline.deployment.smoke_asset_audit import audit
print(json.dumps(audit(r,'beta')))
"""
        beta_audit = site.json([self.site["rtx_python"], "-c", code, site.source,
            self.site["rtx_source_manifest"], manifest["source_sha256"]], "beta", timeout=90)
        atomic(self.root / "static-beta-assets.json", beta_audit)
        if previous_collection is not None:
            receipt, previous_sources = self.reuse_collected_bc(previous_collection, site)
            self.phase("BC_NATIVE_COLLECTION_REUSED", source_sha256=manifest["source_sha256"],
                       previous_run=str(previous_collection), native_replay_allowed=False)
        else:
            self.phase("BC_NATIVE_COLLECTION", source_sha256=manifest["source_sha256"])
        if self.role == "leo":
            collection_args = [site.source / "pipeline/deployment/rtx_actor/qualify_leo.py",
                "--source-root", site.source, "--source-manifest", self.site["rtx_source_manifest"],
                "--physical-gpu", "0", "--memory-gib", "32"]
        else:
            collection_args = [site.source / "pipeline/deployment/rtx_actor/qualify_native_isaac6.py", "--code-root",
                site.source / "pipeline/dagger/rtx/code", "--image-id", IMAGE]
        if previous_collection is None:
            site.process("bc-collection", "start", host="beta", ttl=1300, argv=[self.site["rtx_python"],
                *collection_args, "--mode", "bc-collect", "--num-envs", "1", "--timeout", "1200"])
            site.await_process("bc-collection", host="beta", timeout=1350)
            log = site.remote / "processes/bc-collection/console.log"
            host_result = site.json(["python3", "-c",
                "import json,pathlib,sys; rows=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if x.startswith('{')]; print(json.dumps(rows[-1]))", log], "beta")
            if host_result.get("passed") is not True:
                raise RuntimeError("fresh native BC collection did not pass")
            remote_output = Path(host_result["output"])
            receipt = site.read(remote_output / "bc-collection-smoke.json", "beta")
        if receipt["vector_steps"] != 243 or receipt["optimizer_steps"] != 0:
            raise RuntimeError("BC collector count changed")
        atomic(self.root / "bc-collection.json", receipt)
        # Copy only up to three newly collected chunks; no full archive scan.
        work = ROOT / ("runtime/migrations/beta-to-leo-20261013/h200" if self.role == "leo" else "runtime/migrations/alpha-to-beta-20261008/h200") / self.run
        source_dir = work / "fresh-isaac"
        source_dir.mkdir(parents=True, exist_ok=False)
        sources = []
        for row in receipt["chunks"][:3]:
            if previous_collection is not None:
                original = previous_sources[Path(row["path"]).name]
                local = source_dir / original.name
                shutil.copyfile(original, local)
                if file_sha256(local) != file_sha256(original):
                    raise RuntimeError("reused completed BC chunk copy differs")
                sources.append(local)
                continue
            relative = Path(row["path"]).relative_to("/dagger-runtime")
            remote = remote_output / relative
            local = source_dir / remote.name
            digest = site.command(["sha256sum", remote], "beta").decode().split()[0]
            subprocess.run(["scp", "-F", self.site["ssh_config"], "-o", "BatchMode=yes",
                f"{self.site['rtx_ssh_host']}:{remote}", str(local)+".part"], check=True, timeout=300)
            part = local.with_suffix(local.suffix+".part")
            if file_sha256(part) != digest:
                raise RuntimeError("fresh BC transfer digest differs")
            part.replace(local)
            sources.append(local)
        if not sources:
            raise RuntimeError("no complete fresh BC chunk")
        self.begin_bc_logging(manifest["source_sha256"])
        allocated = False
        try:
            self.phase("BC_RENDER_ALLOCATING_ONE_H200", fresh_chunks=len(sources))
            cpu_key = Path(self.site["cpu_identity_file"]+".pub").read_text().strip()
            beta_key = site.command(["python3", "-c", "from pathlib import Path;import sys;print(Path(sys.argv[1]).read_text().strip())",
                                     self.site["rtx_identity_file"]+".pub"], "beta").decode().strip()
            # Mark ownership before the request: a lost create ACK still requires cleanup.
            allocated = True
            self.allocation(site, "create", [cpu_key, beta_key])
            host_key_path = ROOT / f"runtime/{self.role}-pods" / pre_run / "round-1/host-key.pub"
            site.await_allocation(host_key_path, inspect=lambda: self.allocation(site, "inspect"))
            key = host_key_path.read_text().strip()
            if not key.startswith("ssh-ed25519 ") or "\n" in key:
                raise RuntimeError("workload host key is invalid")
            site.publish(site.round / "known_hosts", (site.pod+" "+key+"\n").encode())
            site.publish(site.gpu_ssh, (f"Host {site.pod}\n HostName {site.pod}\n HostKeyAlias {site.pod}\n User root\n"
                f" IdentityFile {self.site['cpu_identity_file']}\n IdentitiesOnly yes\n StrictHostKeyChecking yes\n"
                f" UserKnownHostsFile {site.round / 'known_hosts'}\n ProxyCommand {shlex.join(site.gpu_proxy('local'))}\n").encode())
            renderer_python = ROOT / "renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python"
            render_output = work / "render"
            self.phase("BC_TEACHER50_RENDER", h200_pod=site.pod)
            site.process("bc-render", "start", host="gpu", ttl=3600, argv=[renderer_python, "-m",
                "pipeline.deployment.h200.qualify_fresh_bc", "render", "--source", sources[0], "--output", render_output,
                *[item for p in sources[1:] for item in ("--chain-source", p)]], env={
                    "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": f"{ROOT}:{ROOT / 'renderer/offline_teacher/source'}",
                    "DIFFSYNTH_MODEL_BASE_PATH": str(ROOT / "renderer/models"), "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                    "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"})
            site.await_process("bc-render", host="gpu", timeout=3630)
            self.phase("BC_MSE_OPTIMIZATION", batch_chunks=16, frames_per_chunk=81)
            train_output = work / "train"
            script = f"source {ROOT}/pipeline/common/setup_h200_isaac_runtime.sh\nexport PYTHONPATH={ROOT}:$PYTHONPATH\nexport CUDA_VISIBLE_DEVICES=0\nexec \"$GR00T_PYTHON\" -m pipeline.deployment.h200.qualify_fresh_bc train --source {render_output}/rendered.npz --output {train_output}"
            site.process("bc-train", "start", host="gpu", ttl=900, argv=["/bin/bash", "-c", script])
            site.await_process("bc-train", host="gpu", timeout=930)
            training = json.loads((train_output / "train-result.json").read_bytes())
            candidate = train_output / "fresh-bc-disposable-candidate.pt"
            if training["result"] != "PASS" or file_sha256(candidate) != training["candidate_sha256"]:
                raise RuntimeError("BC optimization/handoff failed")
            atomic(self.root / "bc-training.json", training)
            self.bc_logger.log({"train/iteration_mse": training["train_mse"],
                                "train/grad_norm": training["grad_norm"], "train/optimizer_steps": 1})
        finally:
            if allocated:
                self.release(site)
        self.phase("BC_SUCCESS_EVALUATION", completed_episode_budget=10, h200_gpus=0)
        site.request["final_checkpoint"] = {"path": str(candidate), "sha256": file_sha256(candidate)}
        evaluation = site.evaluate()
        comparison = json.loads(Path(evaluation["comparison_path"]).read_bytes())
        validate_comparison(panel, comparison["arms"], {"bc": BC_SHA, "dagger": file_sha256(candidate)})
        self.finish_bc_and_run_dagger(training, comparison, candidate, panel_path, manifest["source_sha256"])

    def reuse_collected_bc(self, previous, site):
        """Read completed native output only; never revive simulator state."""
        previous = Path(previous).resolve(strict=True)
        if self.role != "leo" or not previous.is_relative_to(ROOT / "runtime") or previous == self.root:
            raise ValueError("completed-collection reuse requires a distinct Leo runtime directory")
        state = json.loads((previous / "status.json").read_bytes())
        old_run = state.get("run_id", "")
        if (state.get("schema") != "roborender.leo-e2e-smoke.v1" or state.get("phase") != "FAILED"
                or state.get("production_promoted") is not False or state.get("native_replay_allowed") is not False
                or not re.fullmatch(r"dagger-leo-smoke-[a-z0-9-]{1,45}", old_run)
                or (previous / "bc-training.json").exists() or (previous / "campaign").exists()):
            raise RuntimeError("only completed collection from a failed pre-learning Leo smoke may be reused")
        receipt = json.loads((previous / "bc-collection.json").read_bytes())
        if (receipt.get("result") != "PASS" or receipt.get("native_teacher_only") is not True
                or receipt.get("num_envs") != 1 or receipt.get("vector_steps") != 243
                or receipt.get("teacher_action_equalities") != 243 or receipt.get("optimizer_steps") != 0):
            raise RuntimeError("native collection provenance/counts differ")
        old_pre_run = "dagger-leo-smoke-prebc-"+hashlib.sha256(old_run.encode()).hexdigest()[:16]
        old_remote = Path(self.site["rtx_repo"]) / "runtime/leo-campaigns" / old_pre_run / "round-1"
        completed = site.read(old_remote / "processes/bc-collection/result.json", "beta")
        if completed.get("state") != "COMPLETE" or completed.get("exit_code") != 0:
            raise RuntimeError("old native collector has not completed")
        host_result = site.json(["python3", "-c",
            "import json,pathlib,sys;r=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if x.startswith('{')];print(json.dumps(r[-1]))",
            old_remote / "processes/bc-collection/console.log"], "beta")
        if host_result.get("passed") is not True:
            raise RuntimeError("old native completion gate differs")
        remote_root = Path(host_result["output"])
        if not remote_root.is_relative_to(Path(self.site["rtx_repo"]) / "runtime/leo-native-qualification"):
            raise RuntimeError("old native output is not Leo-scoped")
        if site.read(remote_root / "bc-collection-smoke.json", "beta") != receipt:
            raise RuntimeError("local and remote native collection receipts differ")
        local_root = ROOT / "runtime/migrations/beta-to-leo-20261013/h200" / old_run / "fresh-isaac"
        sources, records = {}, []
        for row in receipt["chunks"][:3]:
            relative = Path(row["path"]).relative_to("/dagger-runtime")
            path = local_root / relative.name
            digest = site.command(["sha256sum", remote_root / relative], "beta").decode().split()[0]
            if path.stat().st_size != row["bytes"] or file_sha256(path) != digest:
                raise RuntimeError("completed collection artifact differs across hosts")
            sources[path.name] = path
            records.append({"path": str(path), "sha256": digest, "bytes": row["bytes"]})
        if not records:
            raise RuntimeError("completed collection contains no chunks")
        atomic(self.root / "collection-reuse.json", {"previous_run_id": old_run,
            "previous_source_sha256": state["source_sha256"], "native_work_repeated": False,
            "files": records, "production_promoted": False})
        return receipt, sources

    def recover_completed_bc(self, previous):
        """Recover only completed BC/evaluation; never restart a native process.

        A new smoke identity/source release owns the not-yet-started campaign.
        The failed run and its simulator receipts remain unchanged.
        """
        if self.role != "beta":
            raise ValueError("Leo certification requires fresh native BC; Beta recovery cannot be relabeled")
        previous = previous.resolve(strict=True)
        if not previous.is_relative_to(ROOT / "runtime") or previous == self.root:
            raise ValueError("recovery requires a distinct shared-runtime smoke directory")
        state = json.loads((previous / "status.json").read_bytes())
        prior_run = state.get("run_id", "")
        if (state.get("schema") != "roborender.beta-e2e-smoke.v1" or state.get("phase") != "FAILED"
                or state.get("production_promoted") is not False or state.get("native_replay_allowed") is not False
                or not re.fullmatch(r"dagger-beta-smoke-[a-z0-9-]{1,45}", prior_run) or prior_run == self.run
                or (previous / "campaign").exists()):
            raise RuntimeError("only a failed pre-DAgger smoke with no campaign may be recovered")
        training = json.loads((previous / "bc-training.json").read_bytes())
        collection = json.loads((previous / "bc-collection.json").read_bytes())
        candidate = ROOT / "runtime/migrations/alpha-to-beta-20261008/h200" / prior_run / "train/fresh-bc-disposable-candidate.pt"
        if (training.get("result") != "PASS" or training.get("optimizer_steps") != 1
                or training.get("checkpoint_sha256") != BC_SHA or training.get("sample_chunks") != 16
                or training.get("frames_per_chunk") != 81 or training.get("production_promoted") is not False
                or collection.get("vector_steps") != 243 or collection.get("optimizer_steps") != 0
                or file_sha256(candidate) != training.get("candidate_sha256")):
            raise RuntimeError("completed BC evidence/checkpoint differs")
        panel = make_panel(file_sha256(ROOT / "pipeline/dagger/rtx/config/fixed_panel_motion_inventory.v1.json"), smoke=True)
        if json.loads((previous / "panel.json").read_bytes()) != panel:
            raise RuntimeError("recovery panel differs from the completed evaluation")
        panel_path = self.root / "panel.json"
        panel_path.write_bytes(canonical(panel))
        manifest = json.loads(Path(self.site["source_manifest"]).read_bytes())
        verify(ROOT, manifest)
        pre_run = "dagger-beta-smoke-prebc-"+hashlib.sha256(prior_run.encode()).hexdigest()[:16]
        pre_config = make_first_round_config(pre_run, seed=610121, checkpoint=BC, checkpoint_sha=BC_SHA,
            panel=previous / "panel.json", panel_sha=file_sha256(previous / "panel.json"),
            source_sha=state["source_sha256"], gateway_pids=[], smoke=True)
        site = Site(self.site, {"run_id": pre_run, "round_id": 1, "config": pre_config,
                              "campaign_root": str(previous / "prebc")})
        self.phase("RECOVERING_COMPLETED_BC", previous_run_id=prior_run, source_sha256=manifest["source_sha256"],
                   reused_bc_source_sha256=state["source_sha256"], native_replay_allowed=False)
        # The new source is checked independently; old native work is read-only.
        code = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);m=json.loads(Path(sys.argv[2]).read_bytes());sys.path.insert(0,str(r))
from pipeline.deployment.source_snapshot import file_record,canonical
rows=[file_record(r,x['path']) for x in m['files']]
assert rows==m['files'] and hashlib.sha256(canonical(rows)).hexdigest()==sys.argv[3]
print(json.dumps({'ok':True,'source_sha256':sys.argv[3]}))
"""
        site.json([self.site["rtx_python"], "-c", code, site.source,
                   self.site["rtx_source_manifest"], manifest["source_sha256"]], "beta")
        process = site.process("evaluation", "status", host="beta")
        if process.get("state") != "COMPLETE" or process.get("exit_code") != 0:
            raise RuntimeError("prior evaluation is not complete; recovery never starts it")
        if self.allocation(site, "status").get("present"):
            raise RuntimeError("prior pre-BC allocation has not been released")
        evaluation = "success-beta-panel-"+hashlib.sha256(pre_run.encode()).hexdigest()[:16]+"-r1"
        site.verify_containers_stopped([f"roborender-{evaluation}-{arm}" for arm in ("bc", "dagger")])
        comparison = site.read(Path(self.site["rtx_repo"]) / "runtime/success-validations" / evaluation / "comparison.json", "beta")
        if comparison.get("run_id") != evaluation or comparison.get("complete") is not True:
            raise RuntimeError("prior evaluation identity/completion differs")
        validate_comparison(panel, comparison["arms"], {"bc": BC_SHA, "dagger": file_sha256(candidate)})
        atomic(self.root / "bc-collection.json", collection)
        atomic(self.root / "bc-training.json", training)
        atomic(self.root / "bc-success-comparison.json", comparison)
        atomic(self.root / "recovery.json", {"schema": "roborender.completed-bc-recovery.v1",
            "previous_run_id": prior_run, "previous_output": str(previous), "new_run_id": self.run,
            "previous_source_sha256": state["source_sha256"], "source_sha256": manifest["source_sha256"],
            "candidate_sha256": file_sha256(candidate), "evaluation_id": evaluation,
            "comparison_sha256": file_sha256(self.root / "bc-success-comparison.json"),
            "native_work_repeated": False, "production_promoted": False})
        self.finish_bc_and_run_dagger(training, comparison, candidate, panel_path, manifest["source_sha256"])

    def resume_after_bc_training(self, previous):
        """Start new evaluation/campaign from completed Leo BC, not native replay."""
        previous = Path(previous).resolve(strict=True)
        if self.role != "leo" or not previous.is_relative_to(ROOT / "runtime") or previous == self.root:
            raise ValueError("post-training recovery requires a distinct Leo runtime directory")
        state = json.loads((previous / "status.json").read_bytes())
        prior_run = state.get("run_id", "")
        if (state.get("schema") != "roborender.leo-e2e-smoke.v1" or state.get("phase") != "FAILED"
                or state.get("production_promoted") is not False or state.get("native_replay_allowed") is not False
                or not re.fullmatch(r"dagger-leo-smoke-[a-z0-9-]{1,45}", prior_run)
                or prior_run == self.run or (previous / "campaign").exists()
                or (previous / "prebc/round-1/success-comparison.json").exists()):
            raise RuntimeError("only a failed Leo smoke before success evaluation/DAgger may resume here")
        work = ROOT / "runtime/migrations/beta-to-leo-20261013/h200" / prior_run
        collection = json.loads((previous / "bc-collection.json").read_bytes())
        training = json.loads((previous / "bc-training.json").read_bytes())
        candidate = work / "train/fresh-bc-disposable-candidate.pt"
        render = json.loads((work / "render/render-result.json").read_bytes())
        if (collection.get("result") != "PASS" or collection.get("native_teacher_only") is not True
                or collection.get("vector_steps") != 243 or collection.get("teacher_action_equalities") != 243
                or collection.get("num_envs") != 1 or collection.get("optimizer_steps") != 0
                or training.get("result") != "PASS" or training.get("optimizer_steps") != 1
                or training.get("checkpoint_sha256") != BC_SHA or training.get("sample_chunks") != 16
                or training.get("frames_per_chunk") != 81 or training.get("production_promoted") is not False
                or json.loads((work / "train/train-result.json").read_bytes()) != training
                or file_sha256(candidate) != training.get("candidate_sha256")
                or render.get("result") != "PASS" or render.get("all_source_arrays_preserved") is not True
                or render.get("production_promoted") is not False
                or file_sha256(work / "render/rendered.npz") != training.get("source_sha256")
                or training.get("source_sha256") != render.get("output_sha256")):
            raise RuntimeError("completed Leo BC checkpoint/render/teacher evidence differs")
        processes = previous / "prebc/round-1/processes"
        pre_run = json.loads((processes / "bc-train.json").read_bytes())["owner"]
        if pre_run not in {"dagger-leo-smoke-prebc-named-20261013",
                           "dagger-leo-smoke-prebc-"+hashlib.sha256(prior_run.encode()).hexdigest()[:16]}:
            raise RuntimeError("prior pre-BC allocation owner differs")
        for label in ("bc-render", "bc-train"):
            result = json.loads((processes / label / "result.json").read_bytes())
            if result.get("state") != "COMPLETE" or result.get("exit_code") != 0 or result.get("owner") != pre_run:
                raise RuntimeError("prior BC GPU process is not complete")
        panel = make_panel(file_sha256(ROOT / "pipeline/dagger/rtx/config/fixed_panel_motion_inventory.v1.json"), smoke=True)
        if json.loads((previous / "panel.json").read_bytes()) != panel:
            raise RuntimeError("recovery panel changed")
        panel_path = self.root / "panel.json"
        panel_path.write_bytes(canonical(panel))
        manifest = json.loads(Path(self.site["source_manifest"]).read_bytes())
        verify(ROOT, manifest)
        config = make_first_round_config(pre_run, seed=610121, checkpoint=BC, checkpoint_sha=BC_SHA,
            panel=panel_path, panel_sha=file_sha256(panel_path), source_sha=manifest["source_sha256"],
            gateway_pids=[], smoke=True, deployment_host=site_host(self.site))
        old_site = Site(self.site, {"run_id": pre_run, "round_id": 1, "config": config,
                                  "campaign_root": str(previous / "prebc")})
        prior_allocation = old_site.json([self.site["broker_python"], "-", "status", "--run-id", pre_run,
            "--round-id", "1", "--gpus", "1",
            *(["--existing-name", "isaac-sim-render-leo"] if pre_run.endswith("named-20261013") else [])],
            "broker", data=(ROOT / "pipeline/deployment/h200/beta_pod.py").read_bytes())
        if prior_allocation.get("present") is not False:
            raise RuntimeError("prior H200 allocation has not been released")
        # If native evaluation was even started, do not replay it under a new name.
        if old_site.process("evaluation", "status", host="beta").get("state") != "ABSENT":
            raise RuntimeError("prior evaluation exists; refusing native replay")
        old_eval = "success-leo-panel-"+hashlib.sha256(pre_run.encode()).hexdigest()[:16]+"-r1"
        old_site.verify_containers_stopped([f"roborender-{old_eval}-{arm}" for arm in ("bc", "dagger")])
        code = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);m=json.loads(Path(sys.argv[2]).read_bytes());sys.path.insert(0,str(r))
from pipeline.deployment.source_snapshot import file_record,canonical
rows=[file_record(r,x['path']) for x in m['files']]
assert rows==m['files'] and hashlib.sha256(canonical(rows)).hexdigest()==sys.argv[3]
print(json.dumps({'ok':True,'source_sha256':sys.argv[3]}))
"""
        old_site.json([self.site["rtx_python"], "-c", code, old_site.source,
                       self.site["rtx_source_manifest"], manifest["source_sha256"]], "beta")
        atomic(self.root / "bc-collection.json", collection)
        atomic(self.root / "bc-training.json", training)
        atomic(self.root / "recovery.json", {"schema": "roborender.completed-bc-training-recovery.v1",
            "previous_run_id": prior_run, "previous_output": str(previous), "new_run_id": self.run,
            "previous_source_sha256": state["source_sha256"], "source_sha256": manifest["source_sha256"],
            "candidate_sha256": file_sha256(candidate), "native_work_repeated": False,
            "bc_optimizer_updates_repeated": 0, "production_promoted": False})
        self.phase("RECOVERING_COMPLETED_BC_TRAINING", previous_run_id=prior_run,
                   source_sha256=manifest["source_sha256"], reused_bc_source_sha256=state["source_sha256"],
                   native_replay_allowed=False)
        logger = self.begin_bc_logging(manifest["source_sha256"])
        logger.config.update({"recovered_training_run": prior_run, "new_bc_optimizer_updates": 0})
        logger.log({"train/iteration_mse": training["train_mse"], "train/grad_norm": training["grad_norm"],
                    "recovery/reused_bc_update": True})
        new_pre = "dagger-leo-smoke-prebc-"+hashlib.sha256(self.run.encode()).hexdigest()[:16]
        config = make_first_round_config(new_pre, seed=610121, checkpoint=BC, checkpoint_sha=BC_SHA,
            panel=panel_path, panel_sha=file_sha256(panel_path), source_sha=manifest["source_sha256"],
            gateway_pids=[], smoke=True, deployment_host=site_host(self.site))
        site = Site(self.site, {"run_id": new_pre, "round_id": 1, "config": config,
            "campaign_root": str(self.root / "prebc"), "baseline_checkpoint_path": str(BC),
            "baseline_checkpoint_sha256": BC_SHA,
            "final_checkpoint": {"path": str(candidate), "sha256": file_sha256(candidate)}})
        self.phase("BC_SUCCESS_EVALUATION", completed_episode_budget=10, h200_gpus=0)
        evaluation = site.evaluate()
        comparison = json.loads(Path(evaluation["comparison_path"]).read_bytes())
        validate_comparison(panel, comparison["arms"], {"bc": BC_SHA, "dagger": file_sha256(candidate)})
        self.finish_bc_and_run_dagger(training, comparison, candidate, panel_path, manifest["source_sha256"])

    def finish_bc_and_run_dagger(self, training, comparison, candidate, panel_path, source_sha):
        self.log_bc(training, comparison["arms"]["dagger"], source_sha)
        self.phase("DAGGER_FOUR_ROUND_SMOKE", bc_candidate_sha256=file_sha256(candidate))
        plan = prepare(self.site, self.run, 610121, candidate, panel_path, self.root / "campaign", smoke=True)
        subprocess.run([self.site["cpu_python"], "-m", "pipeline.dagger.beta_campaign",
            "--first-round-config", plan["config"], "--root", plan["state_root"],
            "--operations", plan["operations"], "--wandb-entity", "nvidia", "--execute"], check=True, cwd=ROOT)
        final = json.loads((Path(plan["state_root"])/"state.json").read_bytes())
        if final["phase"] != "COMPLETE" or len(final["completed_rounds"]) != 4:
            raise RuntimeError("all four DAgger rounds did not complete")
        self.phase("COMPLETE", completed_rounds=final["completed_rounds"])

    def begin_bc_logging(self, source_sha):
        if self.bc_logger is not None:
            return self.bc_logger
        from pipeline.deployment.wandb_preflight import admit_logging
        admit_logging("online", "nvidia")
        import wandb
        run_id = "smoke-bc-"+hashlib.sha256(self.run.encode()).hexdigest()[:20]
        self.bc_logger = wandb.init(entity="nvidia", project="dagger", id=run_id, resume="never",
            name=self.run+"-BC", config={"source_sha256": source_sha, "purpose": "real_e2e_smoke",
            "bc_updates": 1, "chunk_population": 16, "frames_per_chunk": 81, "success_cases": 10,
            "initial_checkpoint_sha256": BC_SHA, "production_promoted": False})
        self.phase("BC_LOGGING_READY", bc_wandb_url=self.bc_logger.url)
        return self.bc_logger

    def log_bc(self, training, evaluation, source_sha):
        from pipeline.deployment.wandb_preflight import admit_logging, verify_remote_summary
        api, _ = admit_logging("online", "nvidia")
        run = self.begin_bc_logging(source_sha)
        run_id = "smoke-bc-"+hashlib.sha256(self.run.encode()).hexdigest()[:20]
        metrics = {"train/iteration_mse": training["train_mse"], "train/grad_norm": training["grad_norm"],
            "validation/goal_reached_rate": evaluation["goal_reached_rate"],
            "validation/successes": evaluation["successes"], "validation/completed_episodes": 10}
        try:
            run.log(metrics)
            run.summary.update(metrics)
            url = run.url
        finally:
            run.finish()
            self.bc_logger = None
        verify_remote_summary(api, "nvidia", run_id, metrics)
        self.phase("BC_LOGGED", bc_wandb_url=url)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("smoke stop requested; release owned resources")
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--recover-completed-bc-from", type=Path,
                   help="Read completed BC/evaluation from a failed pre-DAgger smoke; new identity/output required")
    p.add_argument("--reuse-completed-collection-from", type=Path,
                   help="Reuse only verified completed Leo native collection; no simulator replay")
    p.add_argument("--resume-after-bc-training-from", type=Path,
                   help="Reuse completed Leo BC and start its not-yet-started success evaluation under a new identity")
    p.add_argument("--use-existing-named-prebc", action="store_true",
                   help="Use the user-owned isaac-sim-render-leo one-H200 allocation for BC, then release it")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not re.fullmatch(r"dagger-(?:beta|leo)-smoke-[a-z0-9-]{1,45}", args.run_id):
        p.error("fresh dagger-beta-smoke-* identity required")
    if not args.output.resolve().is_relative_to(ROOT / "runtime"):
        p.error("smoke output must be on shared runtime storage")
    if sum(x is not None for x in (args.recover_completed_bc_from, args.reuse_completed_collection_from,
                                  args.resume_after_bc_training_from)) > 1:
        p.error("choose one completed-stage recovery mode")
    if not args.execute:
        print(json.dumps({"launch": False, "BC": "fresh native collection, teacher50 render, one update, 10-case success",
                          "DAgger": "four rounds, three 81-step updates each, 10-case success each"}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    site = json.loads(args.site.read_bytes())
    os.environ["NETRC"] = site["netrc_file"]
    from pipeline.deployment.wandb_preflight import admit_logging
    admit_logging("online", "nvidia")
    smoke = Smoke(site, args.output.resolve(), args.run_id)
    smoke.named_prebc = args.use_existing_named_prebc
    try:
        if args.recover_completed_bc_from is not None:
            smoke.recover_completed_bc(args.recover_completed_bc_from)
        elif args.resume_after_bc_training_from is not None:
            smoke.resume_after_bc_training(args.resume_after_bc_training_from)
        else:
            smoke.execute(args.reuse_completed_collection_from)
    except BaseException as exc:
        smoke.phase("FAILED", error_type=type(exc).__name__, native_replay_allowed=False)
        if smoke.bc_logger is not None:
            smoke.bc_logger.finish(exit_code=1)
            smoke.bc_logger = None
        raise


if __name__ == "__main__":
    main()
