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
    }
    for view in ("head_color", "hand_left", "hand_right"):
        arrays[f"rgb_gen__{view}"] = np.zeros((steps, 108, 192, 3), np.uint8)
        arrays[f"depth_ctrl__{view}"] = np.zeros((steps, 240, 416), np.uint8)
        arrays[f"rgb_robot__{view}"] = np.zeros((steps, 240, 416, 3), np.uint8)
    if bad_selector:
        arrays["behavior_actions"][0, 0] = 1.0
    np.savez_compressed(path, **arrays)


def reservation(path: Path, chunk_id: str = "d1-actor0-env0-ep0") -> dict:
    return {
        "chunk_id": chunk_id,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "run_id": "run-smoke",
        "round_id": 1,
        "actor_id": "actor0",
        "environment_id": 0,
        "episode_id": 0,
        "first_step_id": 0,
        "last_step_id": 80,
        "checkpoint_sha256": ZERO_SHA,
        "teacher_checkpoint_sha256": ONE_SHA,
        "anchor_sha256": "2" * 64,
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

        worker = Path(__file__).parent / "fixtures/dagger_test_worker.py"
        with (
            mock.patch.dict(os.environ, {"DAGGER_CHECKPOINT_SOURCE_ROOT": str(self.tmp_path)}),
            running_service("registry", state) as registry,
            running_service("trajectory", state) as trajectory,
            running_service("coordinator", state) as coordinator,
            running_service("learner", state, worker=worker) as learner,
        ):
            registry.post("/v1/checkpoints/register-local", {
                "path": str(initial_checkpoint), "sha256": initial_sha,
                "metadata": {"generation": "BC", "selected_by": "validation_mse"},
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
            barrier = coordinator.post("/v1/barriers", {
                "run_id": "run-smoke", "round_id": 1, "actor_id": "actor0",
                "checkpoint_sha256": initial_sha,
                "chunk_ids": [chunk_meta["chunk_id"]],
            })
            self.assertTrue(barrier["ready_for_learning"])

            accepted = learner.post("/v1/jobs", {
                "job_id": "learn-d1", "run_id": "run-smoke", "round_id": 1,
                "input_checkpoint_sha256": initial_sha,
                "chunk_ids": [chunk_meta["chunk_id"]],
                "parameters": {"learning_rate": 0.0002, "num_mini_batches": 1},
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

            unrelated = checkpoint_source / "unrelated.pt"
            unrelated.write_bytes(b"unrelated checkpoint")
            unrelated_sha = hashlib.sha256(unrelated.read_bytes()).hexdigest()
            registry.post("/v1/checkpoints/register-local", {
                "path": str(unrelated), "sha256": unrelated_sha,
                "metadata": {"generation": "unrelated"},
            })
            with self.assertRaisesRegex(ServiceError, "412 not_ready"):
                registry.post("/v1/promotions", {
                    "run_id": "run-smoke", "round_id": 1,
                    "checkpoint_sha256": unrelated_sha,
                })

            registry.post("/v1/checkpoints/register-local", {
                "path": output["output_checkpoint_path"],
                "sha256": output["output_checkpoint_sha256"],
                "metadata": {"generation": "dagger_1", "parent": initial_sha},
            })
            promoted = registry.post("/v1/promotions", {
                "run_id": "run-smoke", "round_id": 1,
                "checkpoint_sha256": output["output_checkpoint_sha256"],
            })
            self.assertEqual(promoted["checkpoint_sha256"], output["output_checkpoint_sha256"])
            active = registry.get("/v1/active/run-smoke")["promotion"]
            self.assertEqual(active["round_id"], 1)
            status = coordinator.get("/v1/status/run-smoke")["run"]
            self.assertEqual(status["status"], "round_complete")


if __name__ == "__main__":
    unittest.main()
