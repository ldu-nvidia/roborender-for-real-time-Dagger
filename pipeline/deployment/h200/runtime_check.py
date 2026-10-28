"""Bounded three-window H200 qualification controller; isolated state only."""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import sqlite3
import shlex
import subprocess
import time
import urllib.request
import urllib.error

from pipeline.dagger.services.client import ServiceClient, ServiceError
from pipeline.dagger.services.contracts import canonical_json
from pipeline.deployment.h200.migration_smoke_control import CHECKPOINT, CHECKPOINT_SHA, ROOT


class QualificationServiceClient(ServiceClient):
    """Bounded CPU-controller recovery for idempotent qualification requests.

    GETs are read-only; learner POSTs carry the same immutable job ID/body on
    every retry. HTTP/application errors and identity mismatches are terminal.
    """
    def __init__(self, *args, guard, monotonic=time.monotonic, sleep=time.sleep, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard, self.clock, self.sleep = guard, monotonic, sleep

    def _request(self, method, path, body=None, content_type="application/json"):
        from dagger_rtx.http_transport import TransientTransportError
        deadline = self.clock() + 90
        while True:
            try:
                self.guard()
                return super()._request(method, path, body, content_type)
            except (TransientTransportError, urllib.error.URLError, TimeoutError,
                    ConnectionError, OSError, http.client.HTTPException):
                if self.clock() >= deadline:
                    raise RuntimeError("qualification controller connection recovery expired")
                self.sleep(2)


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def accepted_job(client, job_id, request, *, monotonic=time.monotonic, sleep=time.sleep):
    """Read a known accepted job; never resubmit an optimizer on a 404.

    A just-accepted job can transiently be invisible to a new SQLite reader on
    the shared mount. Only this exact, already-accepted job's 404 is boundedly
    retried. Changed requests, other HTTP errors and persistent absence fail.
    """
    deadline = monotonic() + 20
    while True:
        try:
            job = client.get('/v1/jobs/' + job_id)['job']
        except ServiceError as exc:
            if (str(exc) != f'404 not_found: {job_id}' or monotonic() >= deadline):
                raise
            sleep(1)
            continue
        if job.get('job_id') != job_id or job.get('request') != request:
            raise RuntimeError('accepted learner job identity/request changed')
        return job


def remote_window_audit(supervisor, work, run_id, index, parent, count, gpus):
    """Inspect the WAL only on its owning H200 host, never from the CPU pod.

    This is a small read-only receipt query, not rollout payload validation.
    The pinned SSH route is the same route used by the service supervisor.
    """
    values = [str(work), run_id, index, parent, count, list(gpus)]
    source = ('import json,sys; from pathlib import Path; '
              'from pipeline.deployment.h200.runtime_check import window_audit; '
              'a=json.loads(sys.argv[1]); a[0]=Path(a[0]); '
              'print(json.dumps(window_audit(*a)))')
    command = shlex.join(['env', f'PYTHONPATH={ROOT}', '/usr/bin/python3', '-c', source,
                          json.dumps(values)])
    result = subprocess.run(['ssh', '-F', supervisor['ssh_config'], '-o', 'BatchMode=yes',
                             supervisor['ssh_host'], command], check=True, capture_output=True,
                            text=True, timeout=45)
    audit = json.loads(result.stdout)
    if (audit.get('window_index') != index or audit.get('parent_checkpoint') != parent
            or audit.get('committed_transitions') != 81 * count * len(gpus)):
        raise RuntimeError('remote receipt audit identity/count mismatch')
    return audit


def window_audit(work, run_id, index, parent, num_envs_per_gpu, physical_rtx_gpus=(1, 2)):
    window = f"window-{index:06d}"
    expected = {f"{run_id}-g{gpu}-lane-{lane:03d}" for gpu in physical_rtx_gpus for lane in range(num_envs_per_gpu)}
    from pipeline.dagger.services.ledger_owner import open_readonly
    with open_readonly(Path(work) / 'services/state.sqlite3') as db:
        db.row_factory = sqlite3.Row
        rows = list(db.execute(
            "SELECT actor_id,COUNT(*) n,MIN(rollout_step_index) lo,MAX(rollout_step_index) hi,"
            "COUNT(DISTINCT checkpoint_sha256) versions,MIN(checkpoint_sha256) parent,"
            "SUM(status='committed') committed FROM step_prepares WHERE run_id=? AND rollout_window_id=? GROUP BY actor_id",
            (run_id, window)))
        committed = db.execute(
            "SELECT COUNT(*) FROM step_commits c JOIN step_prepares p ON p.idempotency_key=c.prepare_idempotency_key "
            "WHERE p.run_id=? AND p.rollout_window_id=?", (run_id, window)).fetchone()[0]
        chunks = [dict(r) for r in db.execute("SELECT * FROM chunk_reservations WHERE chunk_id LIKE ?",
                                             (f"d1-{run_id}-%-{window}",))]
    if (len(rows) != len(expected) or {r["actor_id"] for r in rows} != expected or committed != 81 * len(expected)
            or any((r["n"], r["lo"], r["hi"], r["versions"], r["parent"], r["committed"])
                   != (81, 0, 80, 1, parent, 81) for r in rows)
            or {r["chunk_id"] for r in chunks} != {f"d1-{a}-{window}" for a in expected}
            or any(r["status"] != "committed" for r in chunks)):
        raise RuntimeError("incomplete window, checkpoint crosstalk or duplicate transition")
    return {"window_index": index, "committed_transitions": committed,
            "train_chunk_ids": sorted(r["chunk_id"] for r in chunks), "parent_checkpoint": parent}


def checkpoint_continuity(parent_path, output_path):
    import torch
    before = torch.load(parent_path, map_location="cpu", weights_only=False)
    after = torch.load(output_path, map_location="cpu", weights_only=False)
    if after["global_step"] != before["global_step"] + 1:
        raise RuntimeError("learner global step did not advance exactly once")
    old = before["optimizer_state_dict"]["state"]
    new = after["optimizer_state_dict"]["state"]
    if set(old) != set(new) or not new:
        raise RuntimeError("optimizer state topology changed")
    for key in new:
        if float(new[key]["step"]) != float(old[key]["step"]) + 1:
            raise RuntimeError("AdamW step counter was reset or updated more than once")
        if not all(bool(torch.isfinite(new[key][field]).all()) for field in ("exp_avg", "exp_avg_sq")):
            raise RuntimeError("nonfinite optimizer state")
    changed = any(not torch.equal(value, after["policy_state_dict"][name])
                  for name, value in before["policy_state_dict"].items())
    if not changed:
        raise RuntimeError("no student parameter/statistic changed")
    return {"global_step_before": before["global_step"], "global_step_after": after["global_step"],
            "optimizer_states": len(new), "adam_steps_advanced_once": True, "weights_changed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["register", "run"])
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--seed", type=int, default=610191)
    parser.add_argument("--num-envs-per-gpu", type=int, choices=[4, 6, 8], default=4,
                        help="Diagnostic-only batch size; B4 and B6 both require a revised RAM/process plan")
    parser.add_argument("--single-actor", action="store_true", help="Versioned GPU1 B8 / one H200 memory-layout qualification")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-api-key-file", type=Path,
                        help="Owner-only credential file on the CPU controller; never stored in run config")
    parser.add_argument("--supervisor-config", type=Path)
    parser.add_argument("--supervisor-status-dir", type=Path)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / "pipeline/dagger/rtx/code/src"))
    from dagger_rtx.qualification_driver import validate_config, actor_ids
    work = args.work.resolve()
    if not work.is_relative_to(ROOT / "runtime/migrations/alpha-to-beta-20261008/h200"):
        raise ValueError("runtime checks may only use isolated migration services")
    token = (work / "services.token").read_text().strip()
    coordinator = ServiceClient("http://127.0.0.1:18610", token, timeout=30)
    registry = ServiceClient("http://127.0.0.1:18612", token, timeout=120)
    learner = ServiceClient("http://127.0.0.1:18613", token, timeout=120)
    config_path = work / (args.run_id + ".runtime-check.json")
    if args.phase == "register":
        pids = []
        for port in range(18620, 18621 if args.single_actor else 18624):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5) as response:
                ready = json.load(response)
            cmdline = Path(f"/proc/{ready['pid']}/cmdline").read_bytes().split(b"\0")
            if (not ready.get("ok") or ready.get("role") != "render_student" or b"--fake" in cmdline
                    or "roborender.dagger.rtx.v3" not in ready.get("supported_contract_versions", [])):
                raise RuntimeError("genuine qualification worker pool required")
            pids.append(ready["pid"])
        config = validate_config({
            "schema": "roborender.beta-runtime-check.v2" if args.single_actor else "roborender.beta-runtime-check.v1", "run_id": args.run_id,
            "windows": 3, "num_envs_per_gpu": args.num_envs_per_gpu, "physical_rtx_gpus": [1] if args.single_actor else [1, 2],
            "initial_checkpoint_sha256": CHECKPOINT_SHA, "seed": args.seed,
            "gateway_pids": pids, "source_sha256": args.source_sha, "teacher_beta": .75,
            "wandb_run_name": "smoke-dagger-beta-runtime-" + args.run_id.removeprefix("rtx-beta-migration-runtime-"),
            "wandb_run_id": "beta-runtime-" + hashlib.sha256(args.run_id.encode()).hexdigest()[:16],
            "wandb_mode": args.wandb_mode,
            "ack_loss_probe": {"gpu": 1, "window_index": 1, "step_index": 5},
        })
        raw = canonical_json(config).encode()
        if config_path.exists():
            raise RuntimeError("runtime-check registration requires a fresh identity")
        config_path.write_bytes(raw)
        config_path.chmod(0o444)
        registry.post("/v1/checkpoints/register-local", {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA,
                                                       "metadata": {"role": "runtime_check_BC_parent", "epoch": 41}})
        coordinator.post("/v1/runs", {"run_id": args.run_id, "actor_ids": list(actor_ids(config)),
            "rollout_steps": 81, "config_sha256": hashlib.sha256(raw).hexdigest(),
            "initial_checkpoint_sha256": CHECKPOINT_SHA, "validation_interval_iterations": 3})
        coordinator.post("/v1/rounds/start", {"run_id": args.run_id, "round_id": 1, "beta": .75,
                                            "checkpoint_sha256": CHECKPOINT_SHA})
        print(json.dumps({"registered": args.run_id, "config_path": str(config_path),
                          "config_sha256": hashlib.sha256(raw).hexdigest(), "gateway_pids": pids,
                          "wandb_name": config["wandb_run_name"], "wandb_run_id": config["wandb_run_id"]}), flush=True)
        return
    config = validate_config(json.loads(config_path.read_bytes()))
    environment_batch = len(actor_ids(config))
    if config["source_sha256"] != args.source_sha or config["run_id"] != args.run_id:
        raise RuntimeError("controller identity differs from registered actors")
    from pipeline.deployment.wandb_preflight import admit_logging, verify_remote_summary
    logging_api, logging_identity = admit_logging(config["wandb_mode"], args.wandb_entity,
                                                  args.wandb_api_key_file)
    logging_evidence = work / (args.run_id + ".logging-admission.json")
    atomic(logging_evidence, {**logging_identity, "run_id": args.run_id,
                             "online_metrics_verified": False})
    if args.supervisor_config is None or args.supervisor_status_dir is None:
        raise RuntimeError("CPU qualification controller requires owned supervised forwards")
    from pipeline.deployment.rtx_actor.tunnel_supervisor import read_config
    from dagger_rtx.connection_health import ConnectionHealthGate
    supervisor, supervisor_sha = read_config(args.supervisor_config)
    routes = {(row["local_port"], row["remote_port"], row["expected_ready"]["role"])
              for row in supervisor["forwards"]}
    if (supervisor["run_id"] != args.run_id or routes != {
            (18610, 18610, "coordinator"), (18612, 18612, "registry"), (18613, 18613, "learner")}):
        raise RuntimeError("controller supervisor scope differs from the qualification")
    guard = ConnectionHealthGate(args.supervisor_status_dir / "status.json", run_id=args.run_id,
                                config_sha256=supervisor_sha, gateway_pids=())
    guard()
    coordinator = QualificationServiceClient("http://127.0.0.1:18610", token, timeout=10, guard=guard)
    learner = QualificationServiceClient("http://127.0.0.1:18613", token, timeout=10, guard=guard)
    import atexit
    import wandb
    wandb_run = wandb.init(project="dagger", entity=logging_identity.get("entity"),
                           group="beta-runtime-qualification",
                           name=config["wandb_run_name"], id=config["wandb_run_id"],
                           mode=config["wandb_mode"], dir=str(work),
                           settings=wandb.Settings(init_timeout=45), config={
                               **config, "learning_rate": .0002, "max_grad_norm": 1.,
                               "full_environment_batch": environment_batch, "rollout_steps": 81,
                               "image3f_steps": 5, "cfg": 1., "teacache_threshold": .3,
                               "activation_checkpointing": False, "success_rate_tested": False})
    atexit.register(wandb_run.finish)
    print(json.dumps({"phase": "logging_initialized", "mode": config["wandb_mode"],
                      "wandb_run_id": config["wandb_run_id"], "url": wandb_run.url,
                      "online_metrics_verified": False}), flush=True)
    wandb_run.define_metric("global_step")
    wandb_run.define_metric("train/*", step_metric="global_step")
    result_path = work / (args.run_id + '.runtime-results.json')
    parent, results = CHECKPOINT_SHA, []
    for index in range(3):
        deadline = time.monotonic() + 1200
        print(json.dumps({"phase": "waiting_for_actor_barriers", "expected_actors": environment_batch,
                          "iteration": index + 1}), flush=True)
        while True:
            status = coordinator.get("/v1/status/" + args.run_id)
            row = next(r for r in status["iterations"] if r["round_id"] == 1 and r["iteration_id"] == index + 1)
            if row["input_checkpoint_sha256"] != parent:
                raise RuntimeError("controller parent changed")
            if row["status"] == "ready_for_learning":
                break
            if row["status"] != "collecting" or time.monotonic() >= deadline:
                raise RuntimeError("unexpected runtime-check state or collection timeout")
            time.sleep(2)
        audit = remote_window_audit(supervisor, work, args.run_id, index, parent,
                                    config['num_envs_per_gpu'], config['physical_rtx_gpus'])
        job_id = f"{args.run_id}-iteration-{index+1:04d}"
        request = {"job_id": job_id, "run_id": args.run_id, "round_id": 1,
            "input_checkpoint_sha256": parent, "train_chunk_ids": audit["train_chunk_ids"],
            "parameters": {"iteration_id": index + 1, "learning_rate": 2e-4, "max_grad_norm": 1.,
                "num_mini_batches": 1, "seed": config["seed"], "microbatch_environments": environment_batch,
                "raven_exact_full_batch": True, "activation_checkpointing": False,
                "deep_validation_samples": 1, "final_validation_iteration": 3, "validation_interval_iterations": 3,
                "wandb_mode": "disabled", "wandb_project": "dagger", "wandb_group": "beta-runtime-qualification",
                "wandb_run_name": config["wandb_run_name"], "wandb_run_id": config["wandb_run_id"],
                "wandb_resume": "allow", "wandb_extra_config": {"smoke_test": True,
                    "concurrent_environments": environment_batch, "rollout_steps_per_environment": 81,
                    "renderer_worker_count": len(config["gateway_pids"]), "student_action_probability": .25,
                    "teacher_action_probability": .75, "raven_exact_raytraced_augmentation": True,
                    "raven_random_initial_episode_lengths": True}}}
        acknowledgement = learner.post("/v1/jobs", request)
        if acknowledgement.get('ok') is not True or acknowledgement.get('job_id') != job_id:
            raise RuntimeError('learner submission acknowledgement differs')
        started = time.monotonic()
        print(json.dumps({"phase": "learner_submitted", "iteration": index + 1, "job_id": job_id}), flush=True)
        while True:
            job = accepted_job(learner, job_id, request)
            if job["status"] == "succeeded":
                break
            if job["status"] == "failed" or time.monotonic() - started > 240:
                raise RuntimeError("learner failed or timed out; no simulator replay")
            time.sleep(1)
        result = job.get("result") or json.loads(job["result_json"])
        output = result["output_checkpoint_sha256"]
        continuity = checkpoint_continuity(work / "services/checkpoints" / f"{parent}.pt",
                                           work / "services/checkpoints" / f"{output}.pt")
        results.append({**audit, "candidate": output, "mse": result["train_mse"],
                        "learner_seconds": time.monotonic() - started, "continuity": continuity})
        wandb_run.log({"global_step": continuity["global_step_after"], "iteration": index + 1,
                       "train/iteration_mse": result["train_mse"],
                       "train/grad_norm": result["gradient_norm_before_clip"],
                       "timing/learner_seconds": results[-1]["learner_seconds"],
                       "collection/committed_transitions": 81 * environment_batch * (index + 1)})
        atomic(result_path, {"completed_updates": len(results), "iterations": results})
        print(json.dumps({"phase": "learner_pass", **results[-1]}), flush=True)
        parent = output
    print(json.dumps({"phase": "three_window_runtime_check_complete", "updates": 3,
                      "transitions": 243 * environment_batch, "success_evaluation_performed": False,
                      "production_promoted": False, "wandb_run_id": config["wandb_run_id"]}), flush=True)
    wandb_run.summary.update({"completed_updates": 3, "success_rate_tested": False,
                             "online_logging_requested": config["wandb_mode"] == "online"})
    wandb_run.finish()
    if logging_api is not None:
        verify_remote_summary(logging_api, logging_identity["entity"], config["wandb_run_id"], {
            "completed_updates": 3, "train/iteration_mse": results[-1]["mse"],
            "global_step": results[-1]["continuity"]["global_step_after"],
        })
        atomic(logging_evidence, {**logging_identity, "run_id": args.run_id,
                                  "project_write_verified": True, "online_metrics_verified": True,
                                  "wandb_run_id": config["wandb_run_id"], "url": wandb_run.url})
        print(json.dumps({"phase": "online_logging_verified", "url": wandb_run.url}), flush=True)


if __name__ == "__main__":
    main()
