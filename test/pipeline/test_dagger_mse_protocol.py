"""CPU and real loopback HTTP checks for the opt-in MSE round controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from pipeline.dagger.services import mse_protocol as protocol
from pipeline.dagger.services.contracts import file_sha256, canonical_json
from pipeline.dagger.services.heldout_mse import MANIFEST_SCHEMA, RESULT_SCHEMA, canonical_digest
from pipeline.dagger.services.store import StateStore
from pipeline.dagger.services.client import ServiceError, ServiceClient, IdempotentControlClient
from pipeline.dagger.mse_controller import MSEControllerAdapter, atomic_json, digest
from test.pipeline.test_dagger_h200_services import running_service


class MSERoundProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = StateStore(self.root)
        self.checkpoints = []
        with self.store.transaction() as db:
            for index in range(4):
                path = self.root / f"checkpoint-{index}.pt"
                path.write_bytes(f"isolated-fixture-{index}".encode())
                sha = file_sha256(path)
                self.checkpoints.append(sha)
                db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", (sha, path.stat().st_size, "{}", str(path), 0))
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("mse-run", '["actor0"]', 81, "a" * 64, self.checkpoints[0],
                        self.checkpoints[0], 0, None, "created", 50, 0, 0))
        val = self.root / "heldout.npz"
        val.write_bytes(b"metadata-only-test-not-a-GPU-payload")
        self.manifest = {"schema": MANIFEST_SCHEMA, "purpose": "validation", "round_id": 1,
                         "policy_spec_sha256": "b" * 64, "teacher_checkpoint_sha256": "c" * 64,
                         "view_order": ["head_color", "hand_left", "hand_right"],
                         "chunks": [{"chunk_id": "heldout", "path": str(val), "sha256": file_sha256(val),
                                     "size_bytes": val.stat().st_size, "frames": 81,
                                     "source_episode_ids": ["heldout-episode"],
                                     "source_window_ids": ["heldout-window"],
                                     "observation_sources": ["generated_controls", "raytraced"]}]}
        self.manifest_path = self.root / "validation.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.config = {"schema": protocol.SCHEMA, "run_id": "mse-run", "round_id": 1,
                       "initial_checkpoint_sha256": self.checkpoints[0],
                       "validation_manifest_path": str(self.manifest_path),
                       "validation_manifest_sha256": file_sha256(self.manifest_path),
                       "interval_iterations": 1, "max_iterations": 3,
                       "min_validation_epochs": 1, "patience": 2,
                       "relative_improvement": .05, "absolute_improvement": 1e-6,
                       "final_test": {"scenario_manifest_sha256": "d" * 64,
                                      "scenario_ids": ["test-0", "test-1"], "observation_mode": "raytraced"}}

    def tearDown(self):
        self.temporary.cleanup()

    def registration(self):
        return {"protocol": self.config, "protocol_sha256": canonical_digest(self.config)}

    def result(self, iteration, mse):
        return {"run_id": "mse-run", "round_id": 1, "iteration_id": iteration,
                "result": {"schema": RESULT_SCHEMA, "purpose": "validation", "round_id": 1,
                           "validation_manifest_sha256": self.config["validation_manifest_sha256"],
                           "checkpoint_sha256": self.checkpoints[iteration], "mse": mse,
                           "per_action_mse": [mse] * 11, "frames": 81, "elements": 891,
                           "chunk_count": 1, "inference_only": True,
                           "selection_metric": "action_mse", "test_payloads_opened": False}}

    def candidate_ready(self, iteration):
        with self.store.transaction() as db:
            db.execute("UPDATE round_iterations SET candidate_checkpoint_sha256=?,status='awaiting_mse_validation' WHERE run_id='mse-run' AND round_id=1 AND iteration_id=?",
                       (self.checkpoints[iteration], iteration))
            db.execute("UPDATE runs SET status='awaiting_mse_validation' WHERE run_id='mse-run'")

    def start(self, coordinator):
        coordinator.post("/v1/rounds/start", {"run_id": "mse-run", "round_id": 1,
                                             "beta": .75, "checkpoint_sha256": self.checkpoints[0]})

    def test_baseline_selection_patience_idempotency_final_test_and_round_continuity(self):
        with running_service("coordinator", self.root) as coordinator, running_service("evaluation", self.root) as evaluation:
            coordinator = IdempotentControlClient(coordinator.base_url, coordinator.token, sleep=lambda _: None)
            evaluation = IdempotentControlClient(evaluation.base_url, evaluation.token, sleep=lambda _: None)
            original = ServiceClient._request
            lost = set()
            def durable_then_lost(instance, method, path, body=None, content_type="application/json"):
                result = original(instance, method, path, body, content_type)
                key = (path, body)
                if method == "POST" and path.startswith("/v1/mse-rounds/") and key not in lost:
                    lost.add(key)
                    raise ConnectionResetError("test-only lost acknowledgement after real ledger commit")
                return result
            patcher = mock.patch.object(ServiceClient, "_request", durable_then_lost)
            patcher.start()
            self.addCleanup(patcher.stop)
            coordinator.post("/v1/mse-rounds/register", self.registration())
            with self.assertRaisesRegex(ServiceError, "baseline"):
                self.start(coordinator)
            coordinator.post("/v1/mse-rounds/record", self.result(0, 1.0))
            self.start(coordinator)
            before = {"evaluation_id": "final-test", "run_id": "mse-run", "round_id": 1,
                      "purpose": "test", "checkpoint_sha256": self.checkpoints[0],
                      "metric": "raven_goal_reached_rate", **self.config["final_test"]}
            with self.assertRaisesRegex(ServiceError, "only after"):
                evaluation.post("/v2/evaluations", before)
            for iteration, mse in enumerate((.9, .899, .8995), 1):
                self.candidate_ready(iteration)
                response = coordinator.post("/v1/mse-rounds/record", self.result(iteration, mse))
                retry = coordinator.post("/v1/mse-rounds/record", self.result(iteration, mse))
                self.assertTrue(retry["idempotent"])
            self.assertEqual(response["status"], "selected_pending_test")
            selected = coordinator.get("/v1/mse-rounds/mse-run/1")["mse_round"]
            self.assertEqual(selected["stop_reason"], "validation_mse_plateau")
            self.assertEqual(selected["selection"]["best_checkpoint_sha256"], self.checkpoints[2])
            status = coordinator.get("/v1/status/mse-run")
            self.assertEqual(len(status["iterations"]), 3)
            self.assertEqual(status["run"]["status"], "selected_pending_test")
            with self.assertRaisesRegex(ServiceError, "cannot advance"):
                coordinator.post("/v1/iterations/advance", {"run_id": "mse-run", "round_id": 1, "evaluation_id": "final-test"})
            before["checkpoint_sha256"] = self.checkpoints[2]
            evaluation.post("/v2/evaluations", before)
            # Simulate the separate physical executor's completed 0/2 result.
            # No success value may change the selected checkpoint.
            with self.store.transaction() as db:
                for index in range(2):
                    db.execute("INSERT INTO success_evaluation_outcomes VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("final-test", f"test-{index}", f"outcome-{index}", "{}", 0, 0., 10, "done", 0, 0))
                db.execute("UPDATE success_evaluations SET status='complete' WHERE evaluation_id='final-test'")
            finished = coordinator.post("/v1/mse-rounds/finish-test", {"run_id": "mse-run", "round_id": 1, "evaluation_id": "final-test"})
            self.assertEqual(finished["status"], "round_complete")
            self.assertEqual(coordinator.get("/v1/mse-rounds/mse-run/1")["mse_round"]["selection"]["best_checkpoint_sha256"], self.checkpoints[2])
            self.assertEqual(coordinator.get("/v1/status/mse-run")["run"]["status"], "round_complete")
            self.manifest["round_id"] = 2
            self.manifest_path.write_text(json.dumps(self.manifest))
            self.config.update(round_id=2, initial_checkpoint_sha256=self.checkpoints[2], validation_manifest_sha256=file_sha256(self.manifest_path))
            coordinator.post("/v1/mse-rounds/register", self.registration())
            baseline = self.result(0, .899)
            baseline.update(round_id=2)
            baseline["result"].update(round_id=2, checkpoint_sha256=self.checkpoints[2])
            coordinator.post("/v1/mse-rounds/record", baseline)
            coordinator.post("/v1/rounds/start", {"run_id": "mse-run", "round_id": 2, "beta": .5, "checkpoint_sha256": self.checkpoints[2]})
            self.assertEqual(coordinator.get("/v1/status/mse-run")["run"]["active_checkpoint_sha256"], self.checkpoints[2])
            self.assertEqual({path for path, _ in lost}, {"/v1/mse-rounds/register", "/v1/mse-rounds/record", "/v1/mse-rounds/finish-test"})

    def test_actual_retry_client_lost_ack_and_changed_mse_registration_conflict(self):
        with running_service("coordinator", self.root) as raw:
            client = IdempotentControlClient(raw.base_url, raw.token, sleep=lambda _: None)
            original = ServiceClient._request
            seen = []
            def lost_ack(instance, method, path, body=None, content_type="application/json"):
                result = original(instance, method, path, body, content_type)
                seen.append((method, path, body))
                if len(seen) == 1:
                    raise TimeoutError("simulated lost acknowledgement after actual durable HTTP commit")
                return result
            with mock.patch.object(ServiceClient, "_request", lost_ack):
                result = client.post("/v1/mse-rounds/register", self.registration())
            self.assertTrue(result["idempotent"])
            self.assertEqual(seen[0], seen[1])
            changed = self.registration()
            changed["protocol"] = {**changed["protocol"], "patience": 9}
            changed["protocol_sha256"] = canonical_digest(changed["protocol"])
            with self.assertRaises(ServiceError):
                client.post("/v1/mse-rounds/register", changed)
            before = len(seen)
            with mock.patch.object(ServiceClient, "_request", lost_ack):
                with self.assertRaisesRegex(ValueError, "not permitted"):
                    client.post("/v1/mse-rounds/delete", {})
            self.assertEqual(len(seen), before)

    def test_maximum_budget_and_off_cadence_receipt_are_fail_closed(self):
        self.config.update(interval_iterations=2, max_iterations=3, patience=10)
        with running_service("coordinator", self.root) as coordinator:
            coordinator.post("/v1/mse-rounds/register", self.registration())
            coordinator.post("/v1/mse-rounds/record", self.result(0, 1.0))
            self.start(coordinator)
            self.candidate_ready(1)
            with self.assertRaisesRegex(ServiceError, "configured validation barrier"):
                coordinator.post("/v1/mse-rounds/record", self.result(1, .9))
            with self.store.transaction() as db:
                db.execute("UPDATE round_iterations SET iteration_id=3,candidate_checkpoint_sha256=? WHERE run_id='mse-run'", (self.checkpoints[3],))
            result = coordinator.post("/v1/mse-rounds/record", self.result(3, .8))
            self.assertEqual(result["stop_reason"], "maximum_iterations")

    def test_compact_provenance_is_immutable_and_missing_or_overlap_rejected_before_learning(self):
        protocol.register(self.store, self.registration())
        meta = {"run_id": "mse-run", "round_id": 1, "actor_id": "actor0", "rollout_window_id": "train-window",
                "teacher_checkpoint_sha256": "c" * 64, "sealed_window_sha256": "f" * 64}
        with self.store.transaction() as db:
            db.execute("INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)",
                       ("train-chunk", "e" * 64, 10, canonical_json(meta), "committed", "unused", 0, 0))
            with self.assertRaisesRegex(ValueError, "lacks durable"):
                protocol.check_training_admission(db, "mse-run", 1, ["train-chunk"])
            db.execute("INSERT INTO chunk_provenance VALUES (?,?,?,?,?,?)", ("train-chunk", "e" * 64,
                       '["heldout-episode"]', "train-window", "c" * 64, "f" * 64))
            with self.assertRaisesRegex(ValueError, "overlap"):
                protocol.check_training_admission(db, "mse-run", 1, ["train-chunk"])
            db.execute("UPDATE chunk_provenance SET episode_ids_json='[\"train-episode\"]'")
            admitted = protocol.check_training_admission(db, "mse-run", 1, ["train-chunk"])
            self.assertEqual(admitted["source_episode_ids"], ["train-episode"])
            db.execute("INSERT INTO learner_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("job", "mse-run", 1, self.checkpoints[0], '["train-chunk"]', "succeeded", "{}", None, "log", None, 0, 0,))
            self.assertEqual(protocol.training_provenance(db, "mse-run", 1), admitted)

    def test_compact_seal_hash_and_changed_retry(self):
        sealed = {"episode_ids": ["one"] * 40 + ["two"] * 41, "rollout_window_id": "window",
                  "provenance": {"teacher_checkpoint_sha256": "c" * 64}}
        reservation = SimpleNamespace(chunk_id="compact", sha256="a" * 64, rollout_window_id="window",
                                      teacher_checkpoint_sha256="c" * 64, sealed_window_sha256=canonical_digest(sealed))
        protocol.persist_chunk_provenance(self.store, reservation, sealed)
        protocol.persist_chunk_provenance(self.store, reservation, sealed)
        reservation.sha256 = "b" * 64
        with self.assertRaisesRegex(ValueError, "changed on retry"):
            protocol.persist_chunk_provenance(self.store, reservation, sealed)

    def test_controller_adapter_baseline_request_and_recovery_without_gpu(self):
        path = self.root / "protocol.json"
        path.write_text(json.dumps(self.config))
        with running_service("coordinator", self.root) as coordinator:
            coordinator = IdempotentControlClient(coordinator.base_url, coordinator.token, sleep=lambda _: None)
            controller = SimpleNamespace(coordinator=coordinator, run_id="mse-run", round_id=1,
                wandb_mode="disabled", config={"student": {"initial_checkpoint_sha256": self.checkpoints[0]},
                "execution": {"window_count": 50}, "validation": self.config["final_test"], "logging": {}})
            adapter = MSEControllerAdapter(controller, path, ["not-a-real-worker"], output_root=self.root / "requests")
            def fake_worker(argv, check):
                request = json.loads(Path(argv[argv.index("--request") + 1]).read_bytes())
                atomic_json(Path(argv[argv.index("--result") + 1]), {"request_sha256": digest(request), "result": self.result(0, 1.0)["result"]})
            with mock.patch("pipeline.dagger.mse_controller.subprocess.run", side_effect=fake_worker) as worker:
                adapter.ensure_registered()
                adapter.ensure_registered()
                self.assertEqual(worker.call_count, 1)
            self.assertEqual(adapter.context()["status"], "active")


if __name__ == "__main__":
    unittest.main()
