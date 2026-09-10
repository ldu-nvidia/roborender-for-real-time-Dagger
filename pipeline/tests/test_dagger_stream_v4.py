from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pipeline.dagger.services.batch_contract import (
    STREAM_CONTENT_TYPE,
    STREAM_CONTRACT,
    encode_stream_batch,
    lean_arrays_digest,
)
from pipeline.dagger.services.client import ServiceClient, ServiceError
from pipeline.dagger.services.contracts import VIEWS
from pipeline.dagger.services.render_student import FakeRenderStudentBackend
from pipeline.dagger.services.render_student_server import (
    Handler,
    LocalTrajectorySink,
    RenderStudentHTTPServer,
)
from pipeline.dagger.services.step_contract import (
    MODEL_VIEW_MAP,
    RAVEN_COMMIT,
    bytes_sha256,
    canonical_bytes,
)


TOKEN = "render-student-v4-test-token-0123456789abcdef"


def key(
    index: int, *, episode_id: str = "episode0", episode_step_index: int | None = None
) -> dict:
    return {
        "run_id": "v4-smoke",
        "round_id": 1,
        "actor_id": "actor0",
        "rollout_window_id": "window0",
        "rollout_step_index": index,
        "episode_id": episode_id,
        "episode_step_index": index if episode_step_index is None else episode_step_index,
    }


def lean_prepare(
    index: int,
    gateway_instance_id: str,
    *,
    episode_id: str = "episode0",
    episode_step_index: int | None = None,
):
    step_key = key(
        index, episode_id=episode_id, episode_step_index=episode_step_index
    )
    metadata = {
        "contract_version": STREAM_CONTRACT,
        "step_key": step_key,
        "beta": 0.75,
        "view_order": list(VIEWS),
        "model_view_map": MODEL_VIEW_MAP,
        "observation_source": "raytraced",
        "prompt": "A photorealistic humanoid robot task.",
        "env_id": 0,
        "seq": 0,
        "depth_units": "meter",
        "depth_control_encoding": "raven_h5_v4",
        "robot_control_encoding": "textured_robot_rgb_on_black",
        "episode_assignment": {
            "use_raytraced": True,
            "observation_source": "raytraced",
            "anchor_sha256": "a" * 64,
            "assignment_source": "pinned_raven_id_sample_anchor",
            "raven_source": "/home/ldu/BC_render/rtx_BC_transfer",
            "raven_commit": RAVEN_COMMIT,
        },
    }
    homie = np.linspace(-0.2, 0.2, 15, dtype=np.float32)
    arrays = {
        "actor_obs": np.linspace(-1, 1, 113, dtype=np.float32),
        "teacher_action": np.concatenate((
            np.linspace(-1, 1, 11, dtype=np.float32), homie,
        )).astype(np.float32),
    }
    for view_index, view in enumerate(VIEWS):
        arrays[f"raytraced_rgb__{view}"] = np.full(
            (108, 192, 3), (40 + view_index + index) / 255.0, np.float32,
        )
    wire_digest = lean_arrays_digest(metadata, arrays)
    capture_digest = bytes_sha256(canonical_bytes({
        "contract_version": STREAM_CONTRACT,
        "step_key": step_key,
        "wire_digest": wire_digest,
    }))
    lane = {
        "contract_version": STREAM_CONTRACT,
        "idempotency_key": f"prepare-{index}",
        "step_key": step_key,
        "prepare_receipt": hashlib.sha256(f"prepare:{index}".encode()).hexdigest(),
        "capture_digest": capture_digest,
        "wire_digest": wire_digest,
        "beta": 0.75,
        "metadata": metadata,
        "arrays": [],
    }
    return lane, arrays


def commit_for(prepared: dict, *, done: bool = False) -> dict:
    transition = {
        "reward": 1.0, "done": done, "timeout": False, "metadata": {},
    }
    commit_body = {
        "step_key": prepared["step_key"],
        "prepare_receipt": prepared["prepare_receipt"],
        "selection_id": prepared["selection_id"],
        "selected_source": prepared["selected_source"],
        "beta": prepared["beta"],
        "selected_action": prepared["selected_action"],
    }
    commit_digest = bytes_sha256(canonical_bytes(commit_body))
    return {
        "contract_version": STREAM_CONTRACT,
        "idempotency_key": f"commit-{prepared['idempotency_key']}",
        "step_key": prepared["step_key"],
        "prepare_receipt": prepared["prepare_receipt"],
        "selection_id": prepared["selection_id"],
        "selection_receipt": prepared["selection_receipt"],
        "commit_receipt": bytes_sha256(canonical_bytes({
            "commit_digest": commit_digest,
            "result": transition,
        })),
        "transition": transition,
        "sealed_window": None,
    }


def sealed_manifest(prepared: list[dict], commits: list[dict]) -> dict:
    commit_digests = []
    for response in prepared:
        commit_digests.append(bytes_sha256(canonical_bytes({
            "step_key": response["step_key"],
            "prepare_receipt": response["prepare_receipt"],
            "selection_id": response["selection_id"],
            "selected_source": response["selected_source"],
            "beta": response["beta"],
            "selected_action": response["selected_action"],
        })))
    return {
        "schema": "roborender.raven.dagger-window.v1",
        "contract_version": STREAM_CONTRACT,
        "run_id": "v4-smoke",
        "round_id": 1,
        "actor_id": "actor0",
        "rollout_window_id": "window0",
        "row_count": 81,
        "artifact_role": "rtx_identity_outcome_audit",
        "learner_complete": False,
        "contains_policy_rgb": False,
        "authoritative_learner_chunk_owner": "h200_render_student_ledger",
        "episode_ids_are_per_step": True,
        "actual_dones_preserved": True,
        "no_previous_round_replay": True,
        "view_order": list(VIEWS),
        "per_step_identity_fields": [
            "episode_ids", "episode_step_indices", "rollout_step_indices", "round_ids",
            "env_id_per_step", "seq_per_step", "observation_source", "dones", "timeouts",
            "rewards", "behavior_source", "prepare_receipts", "capture_digests",
            "selection_ids", "commit_digests", "commit_receipts",
        ],
        "episode_ids": ["episode0"] * 81,
        "episode_step_indices": list(range(81)),
        "env_ids": [0] * 81,
        "seqs": [0] * 81,
        "round_ids": [1] * 81,
        "observation_sources": ["raytraced"] * 81,
        "dones": [False] * 81,
        "timeouts": [False] * 81,
        "rewards": [1.0] * 81,
        "behavior_sources": [item["selected_source"] for item in prepared],
        "prepare_receipts": [item["prepare_receipt"] for item in prepared],
        "capture_digests": [item["capture_digest"] for item in prepared],
        "selection_ids": [item["selection_id"] for item in prepared],
        "commit_digests": commit_digests,
        "commit_receipts": [item["commit_receipt"] for item in commits],
        "path": "sealed.npz",
        "bytes": 1,
        "sha256": "1" * 64,
        "provenance": {
            "raven_root": "/home/ldu/BC_render/rtx_BC_transfer",
            "raven_commit": RAVEN_COMMIT,
            "raven_videogen_trainer_sha256": "2" * 64,
            "raven_teacher_homie_trainer_sha256": "3" * 64,
            "isaac_capture_sha256": "4" * 64,
            "image_dagger_config_sha256": "5" * 64,
            "teacher_checkpoint_sha256": "6" * 64,
            "homie_walk_sha256": "7" * 64,
            "homie_stand_sha256": "8" * 64,
            "episode_bindings": [{
                "episode_id": "episode0",
                "first_rollout_step_index": 0,
                "use_raytraced": True,
                "observation_source": "raytraced",
                "anchor_sha256": "a" * 64,
            }],
        },
    }


@contextmanager
def service(root: Path):
    backend = FakeRenderStudentBackend()
    backend.test_checkpoint_sha256 = "0" * 64
    server = RenderStudentHTTPServer(
        ("127.0.0.1", 0), Handler, root=root / "state", token=TOKEN,
        backend=backend, trajectory_sink=LocalTrajectorySink(root / "published"),
        ephemeral_window_cache=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            ServiceClient(f"http://127.0.0.1:{server.server_port}", TOKEN),
            server,
            backend,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class StreamingV4Tests(unittest.TestCase):
    def test_prepare_exchange_replay_and_zero_hot_path_disk_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            with service(Path(temporary)) as (client, server, backend):
                ready = client.get("/ready")
                instance = ready["gateway_instance_id"]
                self.assertIn(STREAM_CONTRACT, ready["supported_contract_versions"])
                self.assertEqual(ready["features"]["hot_path_fsyncs"], 0)

                lane0, arrays0 = lean_prepare(0, instance)
                body0 = encode_stream_batch({
                    "contract_version": STREAM_CONTRACT,
                    "kind": "prepare_batch",
                    "batch_id": "step0",
                    "gateway_instance_id": instance,
                    "lanes": [lane0],
                }, [arrays0])
                prepared0 = client.post_bytes(
                    "/v4/steps/prepare-batch", body0, STREAM_CONTENT_TYPE,
                )
                self.assertEqual(prepared0, client.post_bytes(
                    "/v4/steps/prepare-batch", body0, STREAM_CONTENT_TYPE,
                ))
                self.assertEqual(prepared0["gateway_instance_id"], instance)
                self.assertEqual(backend.calls, 1)

                prior_commit = commit_for(prepared0["lanes"][0])
                lane1, arrays1 = lean_prepare(1, instance)
                body1 = encode_stream_batch({
                    "contract_version": STREAM_CONTRACT,
                    "kind": "exchange_batch",
                    "batch_id": "step1",
                    "gateway_instance_id": instance,
                    "previous_commits": [prior_commit],
                    "lanes": [lane1],
                }, [arrays1])
                exchanged = client.post_bytes(
                    "/v4/steps/exchange-batch", body1, STREAM_CONTENT_TYPE,
                )
                self.assertEqual(exchanged, client.post_bytes(
                    "/v4/steps/exchange-batch", body1, STREAM_CONTENT_TYPE,
                ))
                self.assertEqual(exchanged["previous_commit_acks"][0]["status"], "committed")
                self.assertEqual(exchanged["lanes"][0]["step_key"], key(1))
                self.assertEqual(backend.calls, 2)
                self.assertEqual(len(server.window_prepares), 2)
                self.assertEqual(len(server.window_commits), 1)
                self.assertEqual(list((server.root / "prepared").glob("*.npz")), [])
                with server.store.connection() as database:
                    self.assertEqual(database.execute(
                        "SELECT count(*) FROM step_prepares"
                    ).fetchone()[0], 0)
                    self.assertEqual(database.execute(
                        "SELECT count(*) FROM step_commits"
                    ).fetchone()[0], 0)

                changed_lane = dict(lane1)
                changed_lane["capture_digest"] = "b" * 64
                changed = encode_stream_batch({
                    "contract_version": STREAM_CONTRACT,
                    "kind": "exchange_batch",
                    "batch_id": "step1-changed",
                    "gateway_instance_id": instance,
                    "previous_commits": [prior_commit],
                    "lanes": [changed_lane],
                }, [arrays1])
                with self.assertRaisesRegex(ServiceError, "400 contract_error"):
                    client.post_bytes(
                        "/v4/steps/exchange-batch", changed, STREAM_CONTENT_TYPE,
                    )

    def test_commit_survives_current_prepare_failure_and_retry(self):
        class FailOnceBackend(FakeRenderStudentBackend):
            fail_next = False

            def infer(self, metadata, arrays, checkpoint_sha256):
                if self.fail_next:
                    self.fail_next = False
                    raise RuntimeError("injected current-prepare failure")
                return super().infer(metadata, arrays, checkpoint_sha256)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FailOnceBackend()
            backend.test_checkpoint_sha256 = "0" * 64
            server = RenderStudentHTTPServer(
                ("127.0.0.1", 0), Handler, root=root / "state", token=TOKEN,
                backend=backend, trajectory_sink=LocalTrajectorySink(root / "published"),
                ephemeral_window_cache=True,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = ServiceClient(
                f"http://127.0.0.1:{server.server_port}", TOKEN,
            )
            try:
                instance = server.gateway_instance_id
                lane0, arrays0 = lean_prepare(0, instance)
                prepared0 = client.post_bytes(
                    "/v4/steps/prepare-batch",
                    encode_stream_batch({
                        "contract_version": STREAM_CONTRACT,
                        "kind": "prepare_batch",
                        "batch_id": "step0",
                        "gateway_instance_id": instance,
                        "lanes": [lane0],
                    }, [arrays0]),
                    STREAM_CONTENT_TYPE,
                )
                commit0 = commit_for(prepared0["lanes"][0], done=True)
                lane1, arrays1 = lean_prepare(
                    1, instance, episode_id="episode1", episode_step_index=0,
                )
                body = encode_stream_batch({
                    "contract_version": STREAM_CONTRACT,
                    "kind": "exchange_batch",
                    "batch_id": "step1-after-reset",
                    "gateway_instance_id": instance,
                    "previous_commits": [commit0],
                    "lanes": [lane1],
                }, [arrays1])
                backend.fail_next = True
                with self.assertRaisesRegex(ServiceError, "500 internal_error"):
                    client.post_bytes(
                        "/v4/steps/exchange-batch", body, STREAM_CONTENT_TYPE,
                    )
                self.assertEqual(len(server.window_commits), 1)
                self.assertEqual(len(server.window_prepares), 1)

                recovered = client.post_bytes(
                    "/v4/steps/exchange-batch", body, STREAM_CONTENT_TYPE,
                )
                self.assertEqual(recovered["lanes"][0]["step_key"], key(
                    1, episode_id="episode1", episode_step_index=0,
                ))
                self.assertEqual(len(server.window_commits), 1)
                self.assertEqual(len(server.window_prepares), 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_gateway_generation_and_final_step_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with service(Path(temporary)) as (client, server, _backend):
                lane0, arrays0 = lean_prepare(0, server.gateway_instance_id)
                bad = encode_stream_batch({
                    "contract_version": STREAM_CONTRACT,
                    "kind": "prepare_batch",
                    "batch_id": "wrong-generation",
                    "gateway_instance_id": "f" * 64,
                    "lanes": [lane0],
                }, [arrays0])
                with self.assertRaisesRegex(ServiceError, "409 conflict"):
                    client.post_bytes(
                        "/v4/steps/prepare-batch", bad, STREAM_CONTENT_TYPE,
                    )
                self.assertEqual(server.window_prepares, {})

                with self.assertRaisesRegex(ServiceError, "400 contract_error"):
                    client.post("/v4/steps/finalize-batch", {
                        "contract_version": STREAM_CONTRACT,
                        "kind": "finalize_batch",
                        "batch_id": "too-early",
                        "gateway_instance_id": server.gateway_instance_id,
                        "lanes": [{"step_key": key(0)}],
                    })

    def test_full_window_finalizes_once_and_publishes_only_final_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            with service(Path(temporary)) as (client, server, _backend):
                instance = server.gateway_instance_id
                prepared: list[dict] = []
                commits: list[dict] = []
                lane, arrays = lean_prepare(0, instance)
                response = client.post_bytes(
                    "/v4/steps/prepare-batch",
                    encode_stream_batch({
                        "contract_version": STREAM_CONTRACT,
                        "kind": "prepare_batch",
                        "batch_id": "step0",
                        "gateway_instance_id": instance,
                        "lanes": [lane],
                    }, [arrays]),
                    STREAM_CONTENT_TYPE,
                )
                prepared.append(response["lanes"][0])
                for index in range(1, 81):
                    commits.append(commit_for(prepared[-1]))
                    lane, arrays = lean_prepare(index, instance)
                    response = client.post_bytes(
                        "/v4/steps/exchange-batch",
                        encode_stream_batch({
                            "contract_version": STREAM_CONTRACT,
                            "kind": "exchange_batch",
                            "batch_id": f"step{index}",
                            "gateway_instance_id": instance,
                            "previous_commits": [commits[-1]],
                            "lanes": [lane],
                        }, [arrays]),
                        STREAM_CONTENT_TYPE,
                    )
                    prepared.append(response["lanes"][0])

                commits.append(commit_for(prepared[-1]))
                commits[-1]["sealed_window"] = sealed_manifest(prepared, commits)
                final_payload = {
                    "contract_version": STREAM_CONTRACT,
                    "kind": "finalize_batch",
                    "batch_id": "final",
                    "gateway_instance_id": instance,
                    "lanes": [commits[-1]],
                }
                finalized = client.post("/v4/steps/finalize-batch", final_payload)
                self.assertEqual(finalized, client.post(
                    "/v4/steps/finalize-batch", final_payload,
                ))
                self.assertEqual(finalized["lanes"][0]["status"], "committed")
                self.assertEqual(server.trajectory_sink.calls, 1)
                self.assertEqual(len(list((server.root / "sealed").glob("*.npz"))), 1)
                self.assertEqual(list((server.root / "prepared").glob("*.npz")), [])
                self.assertIn(
                    ("v4-smoke", 1, "actor0", "window0"),
                    server.completed_window_keys,
                )


if __name__ == "__main__":
    unittest.main()
