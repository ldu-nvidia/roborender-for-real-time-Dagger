"""Scoped registration, receipt audit and one-update learner handoff for beta."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import urllib.request

from pipeline.dagger.services.client import ServiceClient
from pipeline.dagger.services.contracts import canonical_json

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "checkpoints/students/BC/bc_3view_autoregressive_v1_20260914T044707Z/best_validation.pt"
CHECKPOINT_SHA = "053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["register-one-step", "register-window", "register-probe", "audit", "submit", "job"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-envs", type=int, choices=[2, 4, 6, 8, 12, 16, 28, 40], default=2)
    parser.add_argument("--rtx-gpus", type=int, nargs="+", default=[1])
    parser.add_argument("--gateway-count", type=int, choices=[1, 4], default=1)
    parser.add_argument("--fanout", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-activations", choices=["on", "off"], default="on")
    parser.add_argument("--wait-seconds", type=int, choices=[0, 180, 300], default=0,
                        help="Bounded wait after one disposable learner submission")
    parser.add_argument("--code-sha", default="6672bdb2221b4d7c71c20ded926a6c9254dc6a7afc0abd6c8fd113befdaff15e")
    args = parser.parse_args()
    work = args.work.resolve()
    if not work.is_relative_to(ROOT / "runtime/migrations/alpha-to-beta-20261008/h200"):
        raise RuntimeError("not an isolated migration state root")
    if not re.fullmatch(r"rtx-beta-migration-[a-z0-9-]+", args.run_id):
        raise RuntimeError("unexpected run identity")
    if args.rtx_gpus not in ([1], [1, 2]) or not re.fullmatch(r"[0-9a-f]{64}", args.code_sha):
        raise RuntimeError("unexpected benchmark GPU scope/source SHA")
    token = (work / "services.token").read_text().strip()
    coordinator = ServiceClient("http://127.0.0.1:18610", token)
    registry = ServiceClient("http://127.0.0.1:18612", token)
    learner = ServiceClient("http://127.0.0.1:18613", token)
    if args.phase == "job":
        result = learner.get("/v1/jobs/" + args.run_id + "-one-update")
        (work / f"{args.run_id}.learner.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.phase.startswith("register-"):
        gateways = []
        for port in range(18620, 18620 + args.gateway_count):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=10) as response:
                ready = json.load(response)
            cmdline = Path(f"/proc/{ready['pid']}/cmdline").read_bytes().split(b"\0")
            if not ready.get("ok") or b"--fake" in cmdline or "roborender.dagger.rtx.v3" not in ready.get("supported_contract_versions", []):
                raise RuntimeError("live v3 gateway identity gate failed")
            ready.pop("uptime_seconds", None)
            gateways.append(ready)
        config = {
            "schema": "roborender.beta-migration-smoke.v1", "run_id": args.run_id,
            "num_envs": args.num_envs, "physical_rtx_gpus": args.rtx_gpus,
            "rtx_gpu_uuids": {str(gpu): {
                1: "GPU-0c1949ae-379a-2f2e-0f23-621de2eee409",
                2: "GPU-0540f144-9826-ba8f-140d-af4f95a475ce",
            }[gpu] for gpu in args.rtx_gpus},
            "native_vector_steps": 1 if args.phase == "register-one-step" else 10 if args.phase == "register-probe" else 81,
            "teacher_beta": 0.75, "simulator_seed": args.seed, "rpc_fanout_per_rtx": args.fanout,
            "learner_activation_checkpointing": args.checkpoint_activations == "on",
            "isaac_sim": "6.0.0", "lab_commit": "28a37cecdd433c22d9eabd6a5954add9f13a8951",
            "rtx_code_archive_sha256": args.code_sha,
            "rtx_image_id": "sha256:7aa2c5aaf53e55e749c9f63b6430b9554b4ca49928cf35f376a646669da7b509",
            "checkpoint_sha256": CHECKPOINT_SHA, "gateways": gateways,
            "production_promoted": False, "success_rate_tested": False,
        }
        # Uptime is transient evidence, not immutable run configuration.
        config_path = work / f"{args.run_id}.config.json"
        config_bytes = canonical_json(config).encode()
        if config_path.exists() and config_path.read_bytes() != config_bytes:
            raise RuntimeError("refusing to change a frozen smoke identity")
        if not config_path.exists():
            config_path.write_bytes(config_bytes)
        registered = registry.post("/v1/checkpoints/register-local", {
            "path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA,
            "metadata": {"role": "migration_smoke_BC_input", "original_epoch": 41},
        })
        admitted = coordinator.post("/v1/runs", {
            "run_id": args.run_id,
            "actor_ids": [f"{args.run_id}-g{gpu}-lane-{i:03d}" for gpu in args.rtx_gpus for i in range(args.num_envs)],
            "rollout_steps": 81, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "initial_checkpoint_sha256": CHECKPOINT_SHA, "validation_interval_iterations": 1,
        })
        started = coordinator.post("/v1/rounds/start", {
            "run_id": args.run_id, "round_id": 1, "beta": 0.75,
            "checkpoint_sha256": CHECKPOINT_SHA,
        })
        print(json.dumps({"checkpoint": registered, "run": admitted, "round": started,
                          "gateway_pids": [x["pid"] for x in gateways], "config_path": str(config_path)}))
        return
    from pipeline.dagger.services.ledger_owner import open_readonly
    with open_readonly(work / 'services/state.sqlite3') as db:
        db.row_factory = sqlite3.Row
        counts = [dict(row) for row in db.execute(
            "SELECT actor_id,status,COUNT(*) AS rows,MIN(rollout_step_index) AS first_index,"
            "MAX(rollout_step_index) AS last_index FROM step_prepares WHERE run_id=? GROUP BY actor_id,status",
            (args.run_id,),
        )]
        commits = db.execute("SELECT COUNT(*) FROM step_commits c JOIN step_prepares p ON "
                             "p.idempotency_key=c.prepare_idempotency_key WHERE p.run_id=?", (args.run_id,)).fetchone()[0]
        chunks = [dict(row) for row in db.execute("SELECT * FROM chunk_reservations")
                  if json.loads(row["metadata_json"])["run_id"] == args.run_id]
        jobs = [dict(row) for row in db.execute("SELECT job_id,status FROM learner_jobs WHERE run_id=?", (args.run_id,))]
        selections = [dict(row) for row in db.execute(
            "SELECT json_extract(response_json,'$.selected_source') AS source,COUNT(*) AS rows "
            "FROM step_prepares WHERE run_id=? GROUP BY source", (args.run_id,))]
        observations = [dict(row) for row in db.execute(
            "SELECT json_extract(metadata_json,'$.observation_source') AS source,COUNT(*) AS rows "
            "FROM step_prepares WHERE run_id=? GROUP BY source", (args.run_id,))]
        outcomes = dict(db.execute(
            "SELECT SUM(json_extract(c.transition_json,'$.done')) AS dones,"
            "SUM(json_extract(c.transition_json,'$.timeout')) AS timeouts "
            "FROM step_commits c JOIN step_prepares p ON p.idempotency_key=c.prepare_idempotency_key "
            "WHERE p.run_id=?", (args.run_id,)).fetchone())
    audit = {"run_id": args.run_id, "prepares_by_actor": counts, "commit_count": commits,
             "chunks": [{k: x[k] for k in ("chunk_id", "status", "path", "sha256", "size_bytes")} for x in chunks],
             "jobs": jobs, "action_selections": selections, "observation_sources": observations,
             "outcomes": outcomes}
    if args.phase == "submit":
        frozen = json.loads((work / f"{args.run_id}.config.json").read_text())
        expected = frozen["num_envs"] * len(frozen.get("physical_rtx_gpus", [1]))
        if frozen["native_vector_steps"] != 81 or len(counts) != expected or commits != 81*expected or len(chunks) != expected or any(
            x["rows"] != 81 or x["first_index"] != 0 or x["last_index"] != 80 or x["status"] != "committed" for x in counts
        ) or any(x["status"] != "committed" for x in chunks):
            raise RuntimeError("learner requires every registered actor's complete committed 81-row window")
        for chunk in chunks:
            metadata = json.loads(chunk["metadata_json"])
            coordinator.post("/v1/barriers", {
                "run_id": args.run_id, "round_id": 1, "actor_id": metadata["actor_id"],
                "checkpoint_sha256": CHECKPOINT_SHA, "chunk_ids": [chunk["chunk_id"]],
            })
        submitted_at = time.monotonic()
        audit["submission"] = learner.post("/v1/jobs", {
            "job_id": args.run_id + "-one-update", "run_id": args.run_id, "round_id": 1,
            "input_checkpoint_sha256": CHECKPOINT_SHA,
            "train_chunk_ids": sorted(x["chunk_id"] for x in chunks),
            "parameters": {"iteration_id": 1, "learning_rate": 2e-4, "max_grad_norm": 1.0,
                           "num_mini_batches": 1, "seed": frozen["simulator_seed"], "microbatch_environments": expected,
                           "raven_exact_full_batch": True, "deep_validation_samples": 1,
                           "activation_checkpointing": frozen.get("learner_activation_checkpointing", True),
                           "final_validation_iteration": 1, "validation_interval_iterations": 1,
                           "wandb_mode": "disabled"},
        })
        if args.wait_seconds:
            while True:
                result = learner.get("/v1/jobs/" + args.run_id + "-one-update")
                audit["learner_job"] = result["job"]
                audit["submit_to_result_seconds"] = time.monotonic() - submitted_at
                if result["job"]["status"] not in {"queued", "running"}:
                    break
                if audit["submit_to_result_seconds"] >= args.wait_seconds:
                    audit["wait_timed_out"] = True
                    break
                time.sleep(1)
    (work / f"{args.run_id}.{args.phase}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if audit.get("wait_timed_out") or (
        "learner_job" in audit and audit["learner_job"]["status"] != "succeeded"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
