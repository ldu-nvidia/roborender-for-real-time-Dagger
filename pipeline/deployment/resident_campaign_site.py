"""Opt-in concrete campaign phases using an already qualified resident pool.

The existing CPU/H200 controller and learner are unchanged. Cold Site remains
the default. A phase barrier means both resident native owners are IDLE, not
that their containers were destroyed. No current experiment is auto-migrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.deployment.beta_site import Site, ROOT, site_host, canonical, actor_ids, file_sha256
from pipeline.deployment.beta_operations import atomic, round_runtime_path


class ResidentSite(Site):
    def __init__(self, site, request, pool_file, pool_sha256, *, management_release=None, management_release_sha256=None):
        super().__init__(site, request)
        if site_host(site) != "leo-rtx-6k":
            raise ValueError("resident campaign is explicitly Leo-only")
        self.pool_file, self.pool_sha = Path(pool_file), pool_sha256
        self.pool = self.read(self.pool_file, "beta")
        if hashlib.sha256(canonical(self.pool)).hexdigest() != self.pool_sha:
            raise ValueError("frozen resident pool binding differs")
        if self.pool["source_sha256"] != self.config["source_sha256"]:
            raise ValueError("resident workers and round source differ")
        self.jobs = Path(self.pool["jobs_root"])
        self.remote = round_runtime_path(self.jobs / self.run, self.rid, self.attempts)
        if (management_release is None) != (management_release_sha256 is None):
            raise ValueError("control-only release requires both path and hash")
        self.management_release, self.management_release_sha256 = management_release, management_release_sha256
        self.management_identity = {}

    def verify_sources(self):
        if self.management_release is None:
            return super().verify_sources()
        from pipeline.deployment.infrastructure_overlay import verify_overlay
        native = json.loads(Path(self.site["source_manifest"]).read_bytes())
        if native["source_sha256"] != self.config["source_sha256"]:
            raise RuntimeError("native source and rollout configuration differ")
        self.management_identity = verify_overlay(ROOT, native, self.management_release, self.management_release_sha256)
        return native

    def pool_action(self, action, commands=None):
        argv = [self.site["rtx_python"], self.source / "pipeline/deployment/rtx_actor/resident_pool.py",
                "--pool", self.pool_file, "--action", action]
        if commands is not None:
            path = self.remote / (action+"-"+commands[0]["command_id"]+".json")
            self.publish(path, canonical(commands), "beta", 0o444)
            argv += ["--commands", path]
        return self.json(argv, "beta", timeout=21900 if action == "wait" else 60)

    def barrier(self):
        status = self.pool_action("status")
        if status.get("quiescent") is not True:
            raise RuntimeError("resident native phase is still running")
        rows = [{"physical_gpu": gpu, **{key: status["workers"][gpu][key] for key in
                 ("gpu_uuid", "worker_instance_id", "source_sha256", "phase")}} for gpu in (0, 1)]
        return {"owned_actors_stopped": False,
                "native_phase_barrier": {"schema": "roborender.isaac-phase-barrier.v1", "workers": rows}}

    def prepare(self):
        self.barrier()  # Pin existing owners before allocating cloud GPUs.
        receipt = {**super().prepare(), "native_residency": "across_rounds_and_phases", **self.management_identity}
        if self.request.get("mse_campaign") is True:
            # Explicit owned-allocation command; the CPU controller never
            # chooses a free GPU or starts validation on the access pod.
            receipt["mse_worker_command"] = [self.site["cpu_python"],
                str(ROOT / "pipeline/deployment/h200/mse_worker_proxy.py"),
                "--ssh-config", str(self.gpu_ssh), "--host", self.pod,
                "--run-id", self.run, "--source-root", str(ROOT),
                "--asset-root", str(ROOT), "--netrc-file", self.site["netrc_file"]]
        return receipt

    def collect(self):
        if self.request.get("heldout_collection") is not None:
            from pipeline.deployment.resident_heldout_collection import collect
            return collect(self)
        self.barrier()
        self.publish(self.remote / "config.json", Path(self.request["round_config_path"]).read_bytes(), "beta", 0o444)
        def relative(path):
            return str(path.relative_to(self.jobs))
        connection_root = getattr(self, 'connection_root', self.remote)
        commands = [{"gpu": gpu, "command_id": f"{self.run}-r{self.rid}-collect-g{gpu}",
                     "operation": "collect_round", "payload": {
                         "round_config": relative(self.remote / "config.json"),
                         "round_config_sha256": hashlib.sha256(canonical(self.config)).hexdigest(),
                         "token_file": relative(self.remote / "services.token"),
                         "connection_status_directory": relative(connection_root / "supervisor"),
                         "connection_config_sha256": hashlib.sha256(canonical(self.read(connection_root / "supervisor.json", "beta"))).hexdigest(),
                         "output_name": f"{self.run}-r{self.rid}-collect"}}
                    for gpu in self.gpus]
        if self.request.get("mse_campaign") is True:
            from pipeline.dagger.rtx.code.src.dagger_rtx.mse_round_driver import read_mse_protocol
            path, digest = Path(self.request["mse_protocol_path"]), self.request["mse_protocol_file_sha256"]
            read_mse_protocol(path, digest, self.config)
            self.publish(self.remote / "mse-protocol.json", path.read_bytes(), "beta", 0o444)
            for command in commands:
                command["operation"] = "collect_mse_round"
                command["payload"].update(mse_protocol_file=relative(self.remote / "mse-protocol.json"),
                                           mse_protocol_sha256=digest)
            from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_smoke
            if is_mse_smoke(self.config):
                for command in commands:
                    binding = next(row for row in self.config["motion_assignments"] if row["physical_gpu_index"] == command["gpu"])
                    assignment_path = Path(binding["path"])
                    if file_sha256(assignment_path) != binding["sha256"]:
                        raise ValueError("native training motion assignment changed before publication")
                    destination = self.remote / f"training-assignment-g{command['gpu']}.json"
                    self.publish(destination, assignment_path.read_bytes(), "beta", 0o444)
                    command["payload"].update(motion_assignment_file=relative(destination), motion_assignment_sha256=binding["sha256"])
        self.pool_action("submit", commands)
        result = self.pool_action("wait", commands)
        actors = [row["receipt"]["actor_state"] for row in result["results"]]
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_smoke
        if is_mse_smoke(self.config):
            atomic(self.round / "native-training-receipts.json", result)
            for row in result["results"]:
                receipt = row["receipt"]
                gpu = receipt["actor_state"]["physical_gpu"]
                expected = next(item for item in self.config["motion_assignments"] if item["physical_gpu_index"] == gpu)
                if receipt.get("motion_assignment_sha256") != expected["sha256"] or not receipt.get("episode_assignments"):
                    raise RuntimeError("native training phase lacks observed motion/episode assignments")
        return {"actors": actors, "actor_ids": actor_ids(self.config), **self.barrier()}

    def verify_actors_stopped(self):
        # Base release() uses this predicate to ensure no further env.step is
        # possible in the released phase. The container deliberately survives.
        if getattr(self, "_abort_mode", False):
            # The exact active owner may already have been stopped. Retained
            # journals are uncertainty evidence, not an API-ready worker.
            code = """import json,pathlib,subprocess,sys
pool=json.loads(pathlib.Path(sys.argv[1]).read_bytes())
known=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
for w in pool['workers']:
 s=json.loads((pathlib.Path(w['runtime_root'])/'resident-state.json').read_bytes())
 assert s['worker_instance_id']==w['worker_instance_id']
 if s['phase']!='IDLE' and w['container_name'] in known:
  d=json.loads(subprocess.check_output(['docker','inspect',w['container_name']]))[0]
  assert not d['State']['Running']
print('true')
"""
            return self.json(["python3", "-c", code, self.pool_file], "beta") is True
        self.barrier()
        return True

    def release(self, *, confirm_release=None):
        receipt = super().release(confirm_release=confirm_release)
        receipt.update(self.barrier())  # Do not falsely report a dead process.
        return receipt

    def copy_checkpoint(self, source, digest):
        staged = super().copy_checkpoint(source, digest)
        destination = Path(self.pool["assets_root"]) / (digest+".pt")
        code = ("import hashlib,os,pathlib,sys; s=pathlib.Path(sys.argv[1]);p=pathlib.Path(sys.argv[2]);"
                "p.parent.mkdir(parents=True,exist_ok=True);"
                "assert hashlib.sha256(s.read_bytes()).hexdigest()==sys.argv[3];"
                "os.link(s,p) if not p.exists() else None;"
                "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[3]")
        self.command(["python3", "-c", code, staged, destination, digest], "beta")
        return destination

    def evaluate(self):
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import uses_explicit_motion
        if uses_explicit_motion(self.config):
            return self.evaluate_explicit()
        raise ValueError('legacy ray-traced student execution is disabled; stage the rendered explicit-panel protocol')

    def evaluate_explicit(self):
        """Only the selected policy's independent test; never a stopping input."""
        from pipeline.dagger.beta_validation import load_panel
        from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_result
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_production, is_corrected_smoke
        from pipeline.deployment.rtx_actor.production_success_contract import evaluation_options, protocol_identity, validate_production_result
        panel = load_panel(self.config)
        path = Path(self.config['validation']['scenario_manifest'])
        raw = path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        if raw_sha != self.config['validation']['scenario_manifest_sha256']:
            raise ValueError('final test panel changed before publication')
        production = is_mse_production(self.config)
        options = evaluation_options(panel, raw_sha256=raw_sha) if production else {}
        corrected_smoke = is_corrected_smoke(self.config)
        if corrected_smoke:
            from pipeline.deployment.rtx_actor.smoke_success_contract import evaluation_options as smoke_options
            options = smoke_options(panel)
        if not (production or corrected_smoke):
            raise ValueError('new student success execution requires the rendered production/smoke protocol')
        from pipeline.deployment.rendered_success_dispatch import artifact, payload_fields
        final = self.request['final_checkpoint']
        rendered_config, rendered_raw, rendered_sha = artifact(
            self.config['validation'].get('rendered_observation_config'), panel, final['sha256'])
        suffix = '-image3f-' + rendered_sha[:12]
        output_name = (f"mse-test-{hashlib.sha256(self.run.encode()).hexdigest()[:16]}-r{self.rid}{suffix}"
                       if production else f"mse-test-{hashlib.sha256(self.run.encode()).hexdigest()[:12]}-r{self.rid}{suffix}")
        # Resident command and output identities are limited to 96 characters.
        # A long human-facing campaign name must not break the stage-end test.
        command_id = output_name
        self.barrier()
        final = self.request["final_checkpoint"]
        self.copy_checkpoint(final["path"], final["sha256"])
        remote_panel = self.remote / ("explicit-test-panel" + suffix + ".json")
        self.publish(remote_panel, raw, "beta", 0o444)
        remote_config = self.remote / ('rendered-success-' + rendered_sha + '.json')
        self.publish(remote_config, rendered_raw, 'beta', 0o444)
        commands = [{"gpu": 0, "command_id": command_id,
            "operation": "evaluate_explicit_panel", "payload": {
                "panel_file": str(remote_panel.relative_to(self.jobs)), "panel_sha256": raw_sha,
                "checkpoint_sha256": final["sha256"], "arm": "dagger",
                "output_name": output_name, **options,
                **payload_fields(str(remote_config.relative_to(self.jobs)), rendered_sha)}}]
        self.pool_action("evaluation-memory")
        self.pool_action("submit", commands)
        result = self.pool_action("wait", commands)
        if len(result["results"]) != 1:
            raise RuntimeError("explicit final-test owner cardinality differs")
        receipt = result["results"][0]["receipt"]
        if receipt.get("complete") is not True or receipt.get("purpose") != "test" or receipt.get("optimizer_steps") != 0:
            raise RuntimeError("explicit final test is incomplete")
        if production:
            outcome = validate_production_result(panel, receipt['result'], final['sha256'], allowed_arms=('dagger',), rendered_config=rendered_config)
        elif corrected_smoke:
            from pipeline.deployment.rtx_actor.smoke_success_contract import validate_smoke_result
            outcome = validate_smoke_result(panel, receipt['result'], final['sha256'], allowed_arms=('dagger',), rendered_config=rendered_config)
        else:
            outcome = validate_result(panel, receipt['result'], final['sha256'])
        self.pool_action("training-memory")
        output = self.round / ("explicit-success-test" + suffix + ".json")
        self.publish(output, canonical(outcome), mode=0o444)
        return {"comparison_path": str(output), "comparison_sha256": file_sha256(output), **self.barrier()}

    def abort(self, *, confirm_release=None):
        # An active owned resident phase is stopped, not replayed. Idle owners
        # need not be destroyed. Native uncertainty makes this pool unusable
        # until explicit reconciliation with a fresh worker identity.
        self.validate_release_confirmation(confirm_release)
        self.pool_action("abort", [{"gpu": gpu, "command_id": self.run} for gpu in self.gpus])
        self._abort_mode = True
        return super().abort(confirm_release=confirm_release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--site-sha256", required=True)
    parser.add_argument("--resident-pool", type=Path, required=True)
    parser.add_argument("--resident-pool-sha256", required=True)
    parser.add_argument("--management-release", type=Path)
    parser.add_argument("--management-release-sha256")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--confirm-release", metavar="EXACT_POD_NAME",
                        help="Only after explicit user approval, permit this release/abort to delete the exact pod")
    args = parser.parse_args()
    if file_sha256(args.site) != args.site_sha256:
        raise ValueError("site binding changed")
    request = json.loads(args.request.read_bytes())
    site = ResidentSite(json.loads(args.site.read_bytes()), request, args.resident_pool, args.resident_pool_sha256,
                        management_release=args.management_release, management_release_sha256=args.management_release_sha256)
    phase = request["phase"]
    if phase not in {"prepare", "collect", "release", "evaluate", "control_only", "abort"}:
        raise ValueError("unknown resident campaign phase")
    if args.confirm_release is not None and phase not in {"release", "abort"}:
        raise ValueError("release confirmation is only valid for release/abort cleanup")
    if phase == "prepare":
        # Existing watchdog remains GPU-allocation scoped. Resident owners are
        # separately supervised and never auto-restarted by that watchdog.
        site.watchdog_command = [site.site["cpu_python"], ROOT / "pipeline/deployment/beta_lease_watchdog.py",
                                "--site", args.site, "--site-sha256", args.site_sha256, "--request", args.request,
                                "--resident-pool", args.resident_pool, "--resident-pool-sha256", args.resident_pool_sha256]
    result = getattr(site, phase)(**({"confirm_release": args.confirm_release}
                                    if args.confirm_release is not None else {}))
    atomic(args.receipt, {"ok": True, "operation_id": request["operation_id"], "run_id": request["run_id"],
                          "round_id": request["round_id"], "phase": phase, **result})


if __name__ == "__main__":
    main()
