from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline.dagger.services.client import ServiceClient, ServiceError
from pipeline.dagger.services.contracts import ROLLOUT_STEPS, file_sha256
from pipeline.dagger.services.server import DaggerHTTPServer, Handler


TOKEN = "dagger-service-test-token-0123456789abcdef"
ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


@contextmanager
def running_service(role: str, root: Path, worker: Path | None = None):
    server = DaggerHTTPServer(
        ("127.0.0.1", 0), Handler, role=role, root=root, token=TOKEN,
        max_upload_bytes=64 * 1024 * 1024, worker=worker,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ServiceClient(f"http://127.0.0.1:{server.server_port}", TOKEN, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def write_chunk(path: Path, *, malformed: bool = False, bad_selector: bool = False) -> None:
    steps = ROLLOUT_STEPS - 1 if malformed else ROLLOUT_STEPS
    arrays = {
        "gt_actions": np.zeros((steps, 26), np.float32),
        "student_actions": np.zeros((steps, 26), np.float32),
        "behavior_actions": np.zeros((steps, 26), np.float32),
        "behavior_is_teacher": np.ones((steps,), np.bool_),
        "beta": np.full((steps,), 0.75, np.float32),
        "dones": np.zeros((steps,), np.bool_),
        "obs_actor_obs": np.zeros((steps, 113), np.float32),
        "obs_teacher_obs": np.zeros((steps, 226), np.float32),
        "homie_actions": np.zeros((steps, 15), np.float32),
        "timeouts": np.zeros((steps,), np.bool_),
        "episode_ids": np.array(["episode0"] * steps, dtype="U128"),
        "episode_step_indices": np.arange(steps, dtype=np.int64),
        "rollout_step_indices": np.arange(steps, dtype=np.int64),
        "env_id_per_step": np.zeros((steps,), np.int64),
        "seq_per_step": np.zeros((steps,), np.int64),
        "observation_source": np.array(["generated_controls"] * steps, dtype="U20"),
    }
    for view in ("head_color", "hand_left", "hand_right"):
        arrays[f"rgb_gen__{view}"] = np.zeros((steps, 108, 192, 3), np.float32)
        arrays[f"depth_ctrl__{view}"] = np.zeros((steps, 240, 416), np.uint8)
        arrays[f"rgb_robot__{view}"] = np.zeros((steps, 240, 416, 3), np.uint8)
    if bad_selector:
        arrays["behavior_actions"][0, 0] = 1.0
    np.savez_compressed(path, **arrays)


def reservation(path: Path, chunk_id: str = "d1-actor0-window0") -> dict:
    return {
        "chunk_id": chunk_id,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "run_id": "run-smoke",
        "round_id": 1,
        "actor_id": "actor0",
        "environment_id": 0,
        "rollout_window_id": "window0",
        "first_step_id": 0,
        "last_step_id": 80,
        "checkpoint_sha256": ZERO_SHA,
        "teacher_checkpoint_sha256": ONE_SHA,
        "sealed_window_sha256": "2" * 64,
    }


class H200ServiceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_health_and_authentication(self):
        with running_service("trajectory", self.tmp_path) as client:
            self.assertEqual(client.get("/ready")["role"], "trajectory")
            bad = ServiceClient(client.base_url, "wrong-token-that-is-still-long-enough")
            with self.assertRaisesRegex(ServiceError, "401 unauthorized"):
                bad.get("/v1/chunks/missing")

    def test_run_validation_cadence_is_immutable(self):
        state = self.tmp_path / "cadence-state"
        checkpoint = self.tmp_path / "cadence-bc.pt"
        checkpoint.write_bytes(b"cadence checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with (
            mock.patch.dict(
                os.environ,
                {"DAGGER_CHECKPOINT_SOURCE_ROOT": str(self.tmp_path)},
            ),
            running_service("registry", state) as registry,
            running_service("coordinator", state) as coordinator,
        ):
            registry.post("/v1/checkpoints/register-local", {
                "path": str(checkpoint), "sha256": digest,
                "metadata": {"generation": "BC", "selected_by": "success_rate"},
            })
            request = {
                "run_id": "cadence-run", "actor_ids": ["actor0"],
                "rollout_steps": 81, "config_sha256": "a" * 64,
                "initial_checkpoint_sha256": digest,
                "validation_interval_iterations": 40,
            }
            created = coordinator.post("/v1/runs", request)
            self.assertEqual(created["validation_interval_iterations"], 40)
            self.assertTrue(coordinator.post("/v1/runs", request)["idempotent"])
            conflicting = dict(request, validation_interval_iterations=1)
            with self.assertRaisesRegex(ServiceError, "409 conflict"):
                coordinator.post("/v1/runs", conflicting)

    def test_intermediate_candidate_advances_without_early_validation(self):
        state = self.tmp_path / "interval-state"
        checkpoint = self.tmp_path / "interval-bc.pt"
        checkpoint.write_bytes(b"interval checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        chunk = self.tmp_path / "interval-rollout.npz"
        write_chunk(chunk)
        metadata = reservation(chunk)
        metadata["checkpoint_sha256"] = digest
        worker = Path(__file__).parent / "fixtures/dagger_test_worker.py"
        with (
            mock.patch.dict(
                os.environ,
                {"DAGGER_CHECKPOINT_SOURCE_ROOT": str(self.tmp_path)},
            ),
            running_service("registry", state) as registry,
            running_service("trajectory", state) as trajectory,
            running_service("coordinator", state) as coordinator,
            running_service("learner", state, worker=worker) as learner,
        ):
            registry.post("/v1/checkpoints/register-local", {
                "path": str(checkpoint), "sha256": digest,
                "metadata": {"generation": "BC", "selected_by": "success_rate"},
            })
            coordinator.post("/v1/runs", {
                "run_id": "run-smoke", "actor_ids": ["actor0"],
                "rollout_steps": 81, "config_sha256": "b" * 64,
                "initial_checkpoint_sha256": digest,
                "validation_interval_iterations": 40,
            })
            coordinator.post("/v1/rounds/start", {
                "run_id": "run-smoke", "round_id": 1, "beta": 0.75,
                "checkpoint_sha256": digest,
            })
            reserved = trajectory.post("/v1/chunks/reserve", metadata)
            trajectory.put_file(reserved["upload_path"], chunk)
            coordinator.post("/v1/barriers", {
                "run_id": "run-smoke", "round_id": 1, "actor_id": "actor0",
                "checkpoint_sha256": digest,
                "chunk_ids": [metadata["chunk_id"]],
            })
            learner.post("/v1/jobs", {
                "job_id": "interval-learn-1", "run_id": "run-smoke",
                "round_id": 1, "input_checkpoint_sha256": digest,
                "train_chunk_ids": [metadata["chunk_id"]],
                "parameters": {
                    "learning_rate": 0.0002, "num_mini_batches": 1,
                    "iteration_id": 1,
                },
            })
            deadline = time.time() + 10
            while time.time() < deadline:
                job = learner.get("/v1/jobs/interval-learn-1")["job"]
                if job["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "succeeded", job)
            status = coordinator.get("/v1/status/run-smoke")
            self.assertEqual(status["run"]["status"], "collecting")
            self.assertEqual(
                status["run"]["active_checkpoint_sha256"],
                job["result"]["output_checkpoint_sha256"],
            )
            self.assertEqual(status["iterations"][-1]["iteration_id"], 2)
            self.assertEqual(status["iterations"][-1]["status"], "collecting")
            self.assertEqual(status["iterations"][0]["status"], "advanced")
            replayed_run = coordinator.post("/v1/runs", {
                "run_id": "run-smoke", "actor_ids": ["actor0"],
                "rollout_steps": 81, "config_sha256": "b" * 64,
                "initial_checkpoint_sha256": digest,
                "validation_interval_iterations": 40,
            })
            self.assertTrue(replayed_run["idempotent"])
            replayed_barrier = coordinator.post("/v1/barriers", {
                "run_id": "run-smoke", "round_id": 1, "actor_id": "actor0",
                "checkpoint_sha256": digest,
                "chunk_ids": [metadata["chunk_id"]],
            })
            self.assertTrue(replayed_barrier["idempotent"])
            self.assertEqual(replayed_barrier["iteration_id"], 1)

    def test_trajectory_two_phase_commit_and_idempotency(self):
        chunk = self.tmp_path / "source.npz"
        write_chunk(chunk)
        metadata = reservation(chunk)
        with running_service("trajectory", self.tmp_path / "state") as client:
            created = client.post("/v1/chunks/reserve", metadata)
            self.assertEqual(created["status"], "reserved")
            committed = client.put_file(created["upload_path"], chunk)
            self.assertEqual(committed["status"], "committed")
            self.assertEqual(committed["validation"]["steps"], 81)
            self.assertEqual(committed["validation"]["views"], ["head_color", "hand_left", "hand_right"])
            repeated = client.post("/v1/chunks/reserve", metadata)
            self.assertTrue(repeated["idempotent"])
            self.assertEqual(repeated["status"], "committed")

            conflicting = dict(metadata)
            conflicting["sha256"] = "f" * 64
            with self.assertRaisesRegex(ServiceError, "409 conflict"):
                client.post("/v1/chunks/reserve", conflicting)

    def test_malformed_chunk_never_commits(self):
        chunk = self.tmp_path / "bad.npz"
        write_chunk(chunk, malformed=True)
        metadata = reservation(chunk, "d1-bad")
        with running_service("trajectory", self.tmp_path / "state") as client:
            client.post("/v1/chunks/reserve", metadata)
            with self.assertRaisesRegex(ServiceError, "400 contract_error"):
                client.put_file("/v1/chunks/d1-bad", chunk)
            state = client.get("/v1/chunks/d1-bad")["chunk"]
            self.assertEqual(state["status"], "reserved")
            self.assertFalse((self.tmp_path / "state/chunks/d1-bad.npz").exists())

    def test_behavior_action_must_match_recorded_selector(self):
        chunk = self.tmp_path / "bad-selector.npz"
        write_chunk(chunk, bad_selector=True)
        metadata = reservation(chunk, "d1-bad-selector")
        with running_service("trajectory", self.tmp_path / "state") as client:
            client.post("/v1/chunks/reserve", metadata)
            with self.assertRaisesRegex(ServiceError, "behavior_actions"):
                client.put_file("/v1/chunks/d1-bad-selector", chunk)

    def test_success_evaluation_is_sealed_idempotent_and_never_uses_mse(self):
        state = self.tmp_path / "state"
        checkpoint = self.tmp_path / "candidate.pt"
        checkpoint.write_bytes(b"candidate checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest_sha = "9" * 64
        with (
            mock.patch.dict(os.environ, {"DAGGER_CHECKPOINT_SOURCE_ROOT": str(self.tmp_path)}),
            running_service("registry", state) as registry,
            running_service("evaluation", state) as evaluation,
        ):
            registry.post("/v1/checkpoints/register-local", {
                "path": str(checkpoint), "sha256": digest,
                "metadata": {"generation": "dagger_4", "selected_by": "success_rate"},
            })
            created = evaluation.post("/v2/evaluations", {
                "evaluation_id": "final-test", "run_id": "run-smoke",
                "round_id": 4, "purpose": "test",
                "checkpoint_sha256": digest,
                "scenario_manifest_sha256": manifest_sha,
                "scenario_ids": ["scenario0", "scenario1", "scenario2"],
                "observation_mode": "raytraced",
                "metric": "raven_goal_reached_rate",
            })
            self.assertEqual(created["status"], "collecting")

            def outcome(index: int, success: bool):
                return {
                    "idempotency_key": f"outcome-{index}",
                    "scenario_id": f"scenario{index}",
                    "episode_id": f"episode{index}",
                    "goal_reached": success,
                    "reward": float(index),
                    "steps": 100 + index,
                    "termination": "success" if success else "failure",
                    "safety_violation": False,
                    "checkpoint_sha256": digest,
                    "scenario_manifest_sha256": manifest_sha,
                }

            first = outcome(0, True)
            evaluation.post("/v2/evaluations/final-test/outcomes", first)
            replay = evaluation.post("/v2/evaluations/final-test/outcomes", first)
            self.assertTrue(replay["idempotent"])
            evaluation.post("/v2/evaluations/final-test/outcomes", outcome(1, False))
            completed = evaluation.post(
                "/v2/evaluations/final-test/outcomes", outcome(2, True)
            )
            self.assertEqual(completed["status"], "complete")
            report = evaluation.get("/v2/evaluations/final-test")["evaluation"]
            self.assertEqual(report["successes"], 2)
            self.assertEqual(report["completed"], 3)
            self.assertAlmostEqual(report["goal_reached_rate"], 2 / 3)
            self.assertEqual(len(report["wilson_95_interval"]), 2)
            self.assertNotIn("mse", report)

    def test_raven_success_validation_latches_goal_independently_of_termination(self):
        state = self.tmp_path / "state-v2"
        checkpoint = self.tmp_path / "candidate-v2.pt"
        checkpoint.write_bytes(b"candidate checkpoint v2")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest_sha = "8" * 64
        with (
            mock.patch.dict(os.environ, {"DAGGER_CHECKPOINT_SOURCE_ROOT": str(self.tmp_path)}),
            running_service("registry", state) as registry,
            running_service("evaluation", state) as evaluation,
        ):
            registry.post("/v1/checkpoints/register-local", {
                "path": str(checkpoint), "sha256": digest,
                "metadata": {"generation": "dagger_1", "selected_by": "pending_success_validation"},
            })
            created = evaluation.post("/v2/evaluations", {
                "evaluation_id": "validation-epoch-1", "run_id": "run-smoke",
                "round_id": 0, "purpose": "validation",
                "checkpoint_sha256": digest,
                "scenario_manifest_sha256": manifest_sha,
                "scenario_ids": ["scenario0", "scenario1"],
                "observation_mode": "raytraced",
                "metric": "raven_goal_reached_rate",
            })
            self.assertEqual(created["scenario_count"], 2)

            def outcome(index: int, reached: bool, termination: str):
                return {
                    "idempotency_key": f"validation-outcome-{index}",
                    "scenario_id": f"scenario{index}",
                    "episode_id": f"validation-episode-{index}",
                    "goal_reached": reached,
                    "reward": -10.0 * index,
                    "steps": 200 + index,
                    "termination": termination,
                    "safety_violation": termination == "safety_violation",
                    "checkpoint_sha256": digest,
                    "scenario_manifest_sha256": manifest_sha,
                }

            # Raven OR-latches success during the episode. A later timeout must
            # not erase an already reached goal.
            first = outcome(0, True, "timeout")
            evaluation.post("/v2/evaluations/validation-epoch-1/outcomes", first)
            replay = evaluation.post("/v2/evaluations/validation-epoch-1/outcomes", first)
            self.assertTrue(replay["idempotent"])
            evaluation.post(
                "/v2/evaluations/validation-epoch-1/outcomes",
                outcome(1, False, "failure"),
            )
            report = evaluation.get(
                "/v2/evaluations/validation-epoch-1"
            )["evaluation"]
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["successes"], 1)
            self.assertEqual(report["completed"], 2)
            self.assertEqual(report["goal_reached_rate"], 0.5)
            self.assertEqual(report["metric"], "raven_goal_reached_rate")
            self.assertNotIn("mse", report)

    def test_end_to_end_control_boundary(self):
        state = self.tmp_path / "state"
        checkpoint_source = self.tmp_path / "checkpoint-source"
        checkpoint_source.mkdir()
        initial_checkpoint = checkpoint_source / "bc.pt"
        initial_checkpoint.write_bytes(b"frozen three-view BC checkpoint")
        initial_sha = hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()

        chunk = self.tmp_path / "rollout.npz"
        write_chunk(chunk)
        chunk_meta = reservation(chunk)
        chunk_meta["checkpoint_sha256"] = initial_sha
        validation_chunk = self.tmp_path / "rollout-validation.npz"
        write_chunk(validation_chunk)
        validation_meta = reservation(validation_chunk, "d1-validation")
        validation_meta["checkpoint_sha256"] = initial_sha

        worker = Path(__file__).parent / "fixtures/dagger_test_worker.py"
        with (
            mock.patch.dict(os.environ, {"DAGGER_CHECKPOINT_SOURCE_ROOT": str(self.tmp_path)}),
            running_service("registry", state) as registry,
            running_service("trajectory", state) as trajectory,
            running_service("coordinator", state) as coordinator,
            running_service("learner", state, worker=worker) as learner,
            running_service("evaluation", state) as evaluation,
        ):
            registry.post("/v1/checkpoints/register-local", {
                "path": str(initial_checkpoint), "sha256": initial_sha,
                "metadata": {"generation": "BC", "selected_by": "raven_goal_reached_rate"},
            })
            downloaded = self.tmp_path / "actor-cache/bc.pt"
            registry.download_file(
                f"/v1/checkpoints/{initial_sha}", downloaded, initial_sha
            )
            self.assertEqual(downloaded.read_bytes(), initial_checkpoint.read_bytes())
            coordinator.post("/v1/runs", {
                "run_id": "run-smoke", "actor_ids": ["actor0"],
                "rollout_steps": 81, "config_sha256": "3" * 64,
                "initial_checkpoint_sha256": initial_sha,
            })
            coordinator.post("/v1/rounds/start", {
                "run_id": "run-smoke", "round_id": 1, "beta": 0.75,
                "checkpoint_sha256": initial_sha,
            })
            reserved = trajectory.post("/v1/chunks/reserve", chunk_meta)
            trajectory.put_file(reserved["upload_path"], chunk)
            validation_reserved = trajectory.post("/v1/chunks/reserve", validation_meta)
            trajectory.put_file(validation_reserved["upload_path"], validation_chunk)
            barrier = coordinator.post("/v1/barriers", {
                "run_id": "run-smoke", "round_id": 1, "actor_id": "actor0",
                "checkpoint_sha256": initial_sha,
                "chunk_ids": [chunk_meta["chunk_id"], validation_meta["chunk_id"]],
            })
            self.assertTrue(barrier["ready_for_learning"])

            accepted = learner.post("/v1/jobs", {
                "job_id": "learn-d1", "run_id": "run-smoke", "round_id": 1,
                "input_checkpoint_sha256": initial_sha,
                "train_chunk_ids": [chunk_meta["chunk_id"], validation_meta["chunk_id"]],
                "parameters": {
                    "learning_rate": 0.0002, "num_mini_batches": 1,
                    "iteration_id": 1,
                },
            })
            self.assertEqual(accepted["status"], "queued")
            deadline = time.time() + 10
            while time.time() < deadline:
                job = learner.get("/v1/jobs/learn-d1")["job"]
                if job["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "succeeded", job)
            output = job["result"]

            registered_candidate = self.tmp_path / "candidate-download.pt"
            registry.download_file(
                f"/v1/checkpoints/{output['output_checkpoint_sha256']}",
                registered_candidate,
                output["output_checkpoint_sha256"],
            )
            with self.assertRaisesRegex(ServiceError, "412 not_ready"):
                registry.post("/v1/promotions", {
                    "run_id": "run-smoke", "round_id": 1,
                    "checkpoint_sha256": output["output_checkpoint_sha256"],
                })

            manifest_sha = "8" * 64
            evaluation.post("/v2/evaluations", {
                "evaluation_id": "run-smoke-r1-i1", "run_id": "run-smoke",
                "round_id": 1, "purpose": "validation",
                "checkpoint_sha256": output["output_checkpoint_sha256"],
                "scenario_manifest_sha256": manifest_sha,
                "scenario_ids": ["scenario0", "scenario1"],
                "observation_mode": "raytraced",
                "metric": "raven_goal_reached_rate",
            })
            for index, reached in enumerate((True, False)):
                evaluation.post("/v2/evaluations/run-smoke-r1-i1/outcomes", {
                    "idempotency_key": f"run-smoke-r1-i1-outcome-{index}",
                    "scenario_id": f"scenario{index}",
                    "episode_id": f"episode{index}",
                    "goal_reached": reached,
                    "reward": float(index),
                    "steps": 81,
                    "termination": "success" if reached else "failure",
                    "safety_violation": False,
                    "checkpoint_sha256": output["output_checkpoint_sha256"],
                    "scenario_manifest_sha256": manifest_sha,
                })
            advanced = coordinator.post("/v1/iterations/advance", {
                "run_id": "run-smoke", "round_id": 1,
                "evaluation_id": "run-smoke-r1-i1",
            })
            self.assertEqual(advanced["completed_iteration_id"], 1)
            self.assertEqual(advanced["next_iteration_id"], 2)
            self.assertFalse(advanced["saturated"])
            replay = coordinator.post("/v1/iterations/advance", {
                "run_id": "run-smoke", "round_id": 1,
                "evaluation_id": "run-smoke-r1-i1",
            })
            self.assertTrue(replay["idempotent"])
            status = coordinator.get("/v1/status/run-smoke")["run"]
            self.assertEqual(status["status"], "collecting")
            self.assertEqual(
                status["active_checkpoint_sha256"], output["output_checkpoint_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
