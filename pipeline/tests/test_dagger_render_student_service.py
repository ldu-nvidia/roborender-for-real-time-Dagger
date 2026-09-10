from __future__ import annotations

import hashlib
import json
import os
import pickle
import socket
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline.dagger.services.client import ServiceClient, ServiceError
from pipeline.dagger.services.batch_contract import (
    BATCH_CONTENT_TYPE, BATCH_CONTRACT, decode_binary_batch, encode_binary_batch,
    lean_arrays_digest,
)
from pipeline.dagger.services.contracts import (
    ChunkReservation, VIEWS, file_sha256, validate_dagger_chunk,
)
from pipeline.dagger.services.render_student import (
    FakeRenderStudentBackend, LiveRenderStudentBackend, RenderStudentBackendPool,
    _Image3FLoopbackClient,
)
from pipeline.dagger.services.render_student_server import (
    Handler, LocalTrajectorySink, RenderStudentHTTPServer,
    SharedFilesystemTrajectorySink, load_prepared_artifact, main,
)
from pipeline.dagger.services.raven_learner_worker import (
    backward_global_mse_in_environment_microbatches,
    load_batch,
)
from pipeline.dagger.services.step_contract import (
    MODEL_VIEW_MAP, RAVEN_COMMIT, STEP_CONTRACT, StepKey, anchor_digest,
    arrays_digest, bytes_sha256, canonical_bytes, decode_array, encode_array,
    observation_input_digest,
)


TOKEN = "render-student-test-token-0123456789abcdef"
ZERO_SHA = "0" * 64


def capture_arrays(raytraced: bool = False) -> dict[str, np.ndarray]:
    homie = np.linspace(-0.2, 0.2, 15, dtype=np.float32)
    teacher = np.concatenate((np.linspace(-1, 1, 11, dtype=np.float32), homie))
    arrays = {
        "actor_obs": np.linspace(-1, 1, 113, dtype=np.float32),
        "teacher_obs": np.linspace(-1, 1, 226, dtype=np.float32),
        "critic_obs": np.linspace(1, -1, 226, dtype=np.float32),
        "homie_obs": np.zeros(516, dtype=np.float32),
        "gt_obj_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "teacher_action": teacher.astype(np.float32),
        "homie_action": homie,
    }
    for index, view in enumerate(VIEWS):
        arrays[f"depth_metric__{view}"] = np.full((480, 832), index + 1, np.float32)
        arrays[f"depth_rawq__{view}"] = np.full((480, 832), index + 1, np.uint16)
        arrays[f"depth_raw__{view}"] = np.full((240, 416), index + 1, np.float16)
        arrays[f"depth_ctrl__{view}"] = np.full((240, 416), 20 + index, np.uint8)
        arrays[f"rgb_robot__{view}"] = np.full((240, 416, 3), 30 + index, np.uint8)
        arrays[f"anchor_rgb__{view}"] = np.full((240, 416, 3), 50 + index, np.uint8)
        if raytraced:
            arrays[f"raytraced_rgb__{view}"] = np.full(
                (108, 192, 3), (40 + index) / 255.0, np.float32
            )
    return arrays


def step_key(index: int = 0, *, round_id: int = 1, episode: str = "episode0",
             episode_index: int | None = None) -> dict:
    return {
        "run_id": "run-smoke", "round_id": round_id, "actor_id": "actor0",
        "rollout_window_id": f"window{round_id}", "rollout_step_index": index,
        "episode_id": episode,
        "episode_step_index": index if episode_index is None else episode_index,
    }


def metadata(key: dict, source: str = "generated_controls", anchor: str = "a" * 64) -> dict:
    return {
        "contract_version": STEP_CONTRACT,
        "step_key": key,
        "beta": {1: .75, 2: .5, 3: .25, 4: 0.0}[key["round_id"]],
        "view_order": list(VIEWS),
        "model_view_map": MODEL_VIEW_MAP,
        "observation_source": source,
        "prompt": "A photorealistic humanoid robot task.",
        "env_id": 0,
        "seq": 0,
        "depth_units": "meter",
        "depth_control_encoding": "raven_h5_v4",
        "robot_control_encoding": "textured_robot_rgb_on_black",
        "episode_assignment": {
            "use_raytraced": source == "raytraced",
            "observation_source": source,
            "anchor_sha256": anchor,
            "assignment_source": "pinned_raven_id_sample_anchor",
            "raven_source": "/home/ldu/BC_render/rtx_BC_transfer",
            "raven_commit": RAVEN_COMMIT,
        },
    }


def prepare_payload(key: dict, *, identity: str | None = None,
                    source: str = "generated_controls") -> dict:
    arrays = capture_arrays(source == "raytraced")
    meta = metadata(key, source, anchor_digest(arrays))
    return {
        "contract_version": STEP_CONTRACT,
        "idempotency_key": identity or f"prepare-{key['round_id']}-{key['rollout_step_index']}",
        "step_key": key,
        "prepare_receipt": hashlib.sha256(f"prepare:{key}".encode()).hexdigest(),
        "capture_digest": arrays_digest(meta, arrays),
        "beta": meta["beta"],
        "metadata": meta,
        "arrays": {name: encode_array(value) for name, value in arrays.items()},
    }


def commit_payload(prepared: dict, *, done: bool = False, timeout: bool = False,
                   sealed_window=None) -> dict:
    transition = {"reward": 1.25, "done": done, "timeout": timeout, "metadata": {}}
    commit_body = {
        "step_key": prepared["step_key"],
        "prepare_receipt": prepared["prepare_receipt"],
        "selection_id": prepared["selection_id"],
        "selected_source": prepared["selected_source"],
        "beta": prepared["beta"],
        "selected_action": prepared["selected_action"],
    }
    commit_digest = bytes_sha256(canonical_bytes(commit_body))
    result = {
        "contract_version": STEP_CONTRACT,
        "idempotency_key": f"commit-{prepared['idempotency_key']}",
        "step_key": prepared["step_key"],
        "prepare_receipt": prepared["prepare_receipt"],
        "selection_id": prepared["selection_id"],
        "selection_receipt": prepared["selection_receipt"],
        "commit_receipt": bytes_sha256(canonical_bytes({
            "commit_digest": commit_digest, "result": transition
        })),
        "transition": transition,
        "sealed_window": sealed_window,
    }
    return result


@contextmanager
def service(root: Path, backend=None, *, telemetry_path: Path | None = None,
            phase_telemetry_path: Path | None = None):
    backend = backend or FakeRenderStudentBackend()
    backend.test_checkpoint_sha256 = ZERO_SHA
    sink = LocalTrajectorySink(root / "published")
    server = RenderStudentHTTPServer(
        ("127.0.0.1", 0), Handler, root=root / "state", token=TOKEN,
        backend=backend, trajectory_sink=sink, telemetry_path=telemetry_path,
        phase_telemetry_path=phase_telemetry_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ServiceClient(f"http://127.0.0.1:{server.server_port}", TOKEN), backend, sink
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class RenderStudentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_shared_trajectory_publish_is_zero_copy_hard_link(self):
        root = self.root / "shared"
        source = root / "sealed" / "lean.npz"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"lean-learner-chunk")
        digest = file_sha256(source)
        reservation = ChunkReservation(
            chunk_id="d1-actor-window", sha256=digest,
            size_bytes=source.stat().st_size, run_id="run-smoke", round_id=1,
            actor_id="actor0", environment_id=0, rollout_window_id="window0",
            first_step_id=0, last_step_id=80, checkpoint_sha256=ZERO_SHA,
            teacher_checkpoint_sha256="1" * 64, sealed_window_sha256="2" * 64,
        )
        sink = SharedFilesystemTrajectorySink(root)
        first = sink.publish(source, reservation)
        second = sink.publish(source, reservation)
        destination = root / "chunks" / "d1-actor-window.npz"
        self.assertEqual(first["status"], "committed")
        self.assertTrue(second["idempotent"])
        self.assertTrue(os.path.samefile(source, destination))

    def test_local_test_sink_is_rejected_for_live_backend(self):
        token_file = self.root / "token"
        token_file.write_text(TOKEN, encoding="utf-8")
        argv = [
            "render_student_server",
            "--token-file", str(token_file),
            "--local-test-sink-root", str(self.root / "published"),
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "allowed only with --fake"):
                main()

    def test_private_telemetry_is_append_only_and_does_not_change_wire_response(self):
        telemetry = self.root / "isolated" / "render-student.jsonl"
        payload = prepare_payload(step_key())
        with service(self.root, telemetry_path=telemetry) as (client, _, _):
            prepared = client.post("/v1/steps/prepare", payload)
            self.assertEqual(prepared, client.post("/v1/steps/prepare", payload))
            committed = client.post("/v1/steps/commit", commit_payload(prepared))
            self.assertEqual(committed["status"], "committed")

        records = [json.loads(line) for line in telemetry.read_text().splitlines()]
        self.assertEqual(
            [(item["route"], item["status"]) for item in records],
            [("prepare", "prepared"), ("prepare", "idempotent_replay"),
             ("commit", "committed")],
        )
        first = records[0]
        self.assertEqual(first["schema"], "roborender.dagger.h200-telemetry.v1")
        self.assertEqual(first["idempotency_key"], payload["idempotency_key"])
        self.assertEqual(first["worker_slot"], 0)
        self.assertEqual(first["gpu_device"], "fake")
        self.assertGreater(first["request_bytes"], 0)
        self.assertGreater(first["response_bytes"], 0)
        self.assertGreaterEqual(first["queue_wait_ms"], 0.0)

    def test_v3_lean_batch_registers_anchor_and_persists_one_uncompressed_barrier(self):
        anchors = capture_arrays()
        anchor_arrays = {
            f"anchor_rgb__{view}": anchors[f"anchor_rgb__{view}"] for view in VIEWS
        }
        anchor_sha = anchor_digest(anchors)
        anchor_header = {
            "contract_version": BATCH_CONTRACT,
            "kind": "anchor_batch",
            "batch_id": "anchors-smoke",
            "anchors": [{"anchor_sha256": anchor_sha, "arrays": []}],
        }
        lane_headers = []
        lane_arrays = []
        for index in range(2):
            key = step_key()
            key["actor_id"] = f"actor{index}"
            key["rollout_window_id"] = f"window-{index}"
            arrays = capture_arrays()
            lean = {
                "actor_obs": arrays["actor_obs"],
                "teacher_action": arrays["teacher_action"],
                **{
                    name: arrays[name]
                    for view in VIEWS
                    for name in (f"depth_ctrl__{view}", f"rgb_robot__{view}")
                },
            }
            meta = metadata(key, anchor=anchor_sha)
            meta["contract_version"] = BATCH_CONTRACT
            meta["step_key"] = key
            meta["env_id"] = index
            lane_headers.append({
                "contract_version": BATCH_CONTRACT,
                "idempotency_key": f"batch-prepare-{index}",
                "step_key": key,
                "prepare_receipt": hashlib.sha256(f"batch-prepare:{index}".encode()).hexdigest(),
                "capture_digest": hashlib.sha256(f"full-capture:{index}".encode()).hexdigest(),
                "wire_digest": lean_arrays_digest(meta, lean),
                "beta": 0.75,
                "metadata": meta,
                "arrays": [],
            })
            lane_arrays.append(lean)
        prepare_header = {
            "contract_version": BATCH_CONTRACT,
            "kind": "prepare_batch",
            "batch_id": "prepare-smoke",
            "lanes": lane_headers,
        }

        with service(self.root) as (client, backend, _):
            ready = client.get("/ready")
            self.assertEqual(
                ready["supported_contract_versions"], [STEP_CONTRACT, BATCH_CONTRACT]
            )
            self.assertTrue(ready["features"]["minimal_compressed_prepare_batch"])
            registered = client.post_bytes(
                "/v3/anchors/register-batch",
                encode_binary_batch(anchor_header, [anchor_arrays]),
                BATCH_CONTENT_TYPE,
            )
            self.assertEqual(registered["anchors"][0]["status"], "registered")
            cached = client.post_bytes(
                "/v3/anchors/register-batch",
                encode_binary_batch(anchor_header, [anchor_arrays]),
                BATCH_CONTENT_TYPE,
            )
            self.assertEqual(cached["anchors"][0]["status"], "cached")
            body = encode_binary_batch(prepare_header, lane_arrays)
            prepared = client.post_bytes(
                "/v3/steps/prepare-batch", body, BATCH_CONTENT_TYPE
            )
            replay = client.post_bytes(
                "/v3/steps/prepare-batch", body, BATCH_CONTENT_TYPE
            )
            self.assertEqual(prepared, replay)
            self.assertEqual(backend.calls, 2)
            self.assertEqual(len(prepared["lanes"]), 2)
            artifacts = list((self.root / "state" / "prepared").glob("batch-*.npz"))
            self.assertEqual(len(artifacts), 1)
            with zipfile.ZipFile(artifacts[0]) as archive:
                self.assertTrue(archive.infolist())
                self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()))
            with sqlite3.connect(self.root / "state" / "state.sqlite3") as database:
                references = [row[0] for row in database.execute(
                    "SELECT artifact_path FROM step_prepares ORDER BY actor_id"
                )]
            self.assertEqual(len(references), 2)
            self.assertEqual(load_prepared_artifact(references[0])["actor_obs"].shape, (113,))

            changed_headers = json.loads(json.dumps(lane_headers))
            changed_headers[0]["metadata"]["prompt"] = "Changed semantic payload."
            changed_headers[0]["wire_digest"] = lean_arrays_digest(
                changed_headers[0]["metadata"], lane_arrays[0]
            )
            with self.assertRaisesRegex(ServiceError, "409 conflict"):
                client.post_bytes(
                    "/v3/steps/prepare-batch",
                    encode_binary_batch({
                        "contract_version": BATCH_CONTRACT,
                        "kind": "prepare_batch",
                        "batch_id": "prepare-changed",
                        "lanes": changed_headers,
                    }, lane_arrays),
                    BATCH_CONTENT_TYPE,
                )

            commits = []
            for lane in prepared["lanes"]:
                commit = commit_payload(lane)
                commit["contract_version"] = BATCH_CONTRACT
                commits.append(commit)
            committed = client.post("/v3/steps/commit-batch", {
                "contract_version": BATCH_CONTRACT,
                "kind": "commit_batch",
                "batch_id": "commit-smoke",
                "lanes": commits,
            })
            self.assertEqual([lane["status"] for lane in committed["lanes"]], ["committed"] * 2)

    def test_v3_prepare_defers_sqlite_writer_transaction_until_after_inference(self):
        """Eight gateways must not serialize on a write lock before rendering."""

        write_transaction_entries = 0
        test = self

        class WriterBoundaryBackend(FakeRenderStudentBackend):
            def infer(self, metadata, arrays, checkpoint_sha256):
                test.assertEqual(
                    write_transaction_entries, 0,
                    "prepare acquired SQLite's writer slot before inference",
                )
                return super().infer(metadata, arrays, checkpoint_sha256)

        backend = WriterBoundaryBackend()
        backend.test_checkpoint_sha256 = ZERO_SHA
        server = RenderStudentHTTPServer(
            ("127.0.0.1", 0), Handler, root=self.root / "writer-boundary",
            token=TOKEN, backend=backend,
            trajectory_sink=LocalTrajectorySink(self.root / "writer-boundary-published"),
        )
        try:
            anchors = capture_arrays()
            anchor_sha = anchor_digest(anchors)
            server.anchor_cache[anchor_sha] = {
                f"anchor_rgb__{view}": anchors[f"anchor_rgb__{view}"] for view in VIEWS
            }
            key = step_key()
            arrays = capture_arrays()
            lean = {
                "actor_obs": arrays["actor_obs"],
                "teacher_action": arrays["teacher_action"],
                **{
                    name: arrays[name]
                    for view in VIEWS
                    for name in (f"depth_ctrl__{view}", f"rgb_robot__{view}")
                },
            }
            meta = metadata(key, anchor=anchor_sha)
            meta["contract_version"] = BATCH_CONTRACT
            lane = {
                "contract_version": BATCH_CONTRACT,
                "idempotency_key": "writer-boundary-prepare",
                "step_key": key,
                "prepare_receipt": hashlib.sha256(b"writer-boundary").hexdigest(),
                "capture_digest": hashlib.sha256(b"full-capture").hexdigest(),
                "wire_digest": lean_arrays_digest(meta, lean),
                "beta": 0.75,
                "metadata": meta,
                "arrays": [],
            }
            header = {
                "contract_version": BATCH_CONTRACT,
                "kind": "prepare_batch",
                "batch_id": "writer-boundary-batch",
                "lanes": [lane],
            }
            original_transaction = server.store.transaction

            @contextmanager
            def counted_transaction():
                nonlocal write_transaction_entries
                write_transaction_entries += 1
                with original_transaction() as database:
                    yield database

            server.store.transaction = counted_transaction
            handler = object.__new__(Handler)
            handler.server = server
            wire = encode_binary_batch(header, [lean])
            decoded_header, decoded_arrays = decode_binary_batch(wire)
            response = handler._prepare_batch(decoded_header, decoded_arrays, wire)
            self.assertEqual(response["lanes"][0]["idempotency_key"], lane["idempotency_key"])
            self.assertEqual(backend.calls, 1)
            self.assertEqual(
                write_transaction_entries, 1,
                "prepare should take exactly one short writer transaction to persist its result",
            )
        finally:
            server.server_close()

    def test_phase_telemetry_has_start_end_boundaries_without_wire_changes(self):
        phases = self.root / "isolated" / "phase-timing.jsonl"
        payload = prepare_payload(step_key())
        with service(self.root, phase_telemetry_path=phases) as (client, _, _):
            prepared = client.post("/v1/steps/prepare", payload)
            client.post("/v1/steps/commit", commit_payload(prepared))

        records = [json.loads(line) for line in phases.read_text().splitlines()]
        pairs = [(item["phase"], item["event"]) for item in records]
        for phase in (
            "h200_image3f_render_and_student_inference",
            "h200_beta_action_selection",
            "h200_prepare_artifact_persistence",
        ):
            self.assertIn((phase, "start"), pairs)
            self.assertIn((phase, "end"), pairs)
        for record in records:
            self.assertEqual(record["schema"], "roborender.dagger.phase-timing.v1")
            self.assertEqual(record["run_id"], "run-smoke")
            self.assertEqual(record["round_id"], 1)
            self.assertEqual(record["vector_step_index"], 0)
            if record["event"] == "end":
                self.assertGreaterEqual(record["elapsed_ms"], 0.0)

    def test_image3f_loopback_reuses_one_persistent_connection(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        accepts = []

        def serve_two_pings():
            connection, _ = listener.accept()
            accepts.append(1)
            with connection:
                for _ in range(2):
                    size = struct.unpack("!Q", _Image3FLoopbackClient._recv_exact(connection, 8))[0]
                    request = pickle.loads(_Image3FLoopbackClient._recv_exact(connection, size))
                    self.assertEqual(request, {"cmd": "ping"})
                    response = pickle.dumps({"ok": True}, protocol=pickle.HIGHEST_PROTOCOL)
                    connection.sendall(struct.pack("!Q", len(response)) + response)

        thread = threading.Thread(target=serve_two_pings, daemon=True)
        thread.start()
        client = _Image3FLoopbackClient("127.0.0.1", listener.getsockname()[1])
        try:
            self.assertTrue(client.ping())
            self.assertTrue(client.ping())
        finally:
            client._close()
            listener.close()
            thread.join(timeout=5)
        self.assertEqual(len(accepts), 1)

    def test_image3f_loopback_serializes_concurrent_framed_exchanges(self):
        class ProbeStream:
            def __init__(self):
                self.guard = threading.Lock()
                self.local = threading.local()
                self.active = 0
                self.maximum_active = 0
                self.closed = False

            def sendall(self, frame):
                with self.guard:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                try:
                    # Make simultaneous callers overlap deterministically when
                    # the client's framed stream is not protected by a lock.
                    time.sleep(0.05)
                    size = struct.unpack("!Q", frame[:8])[0]
                    request = pickle.loads(frame[8:8 + size])
                    response = pickle.dumps(
                        {"request_id": request["request_id"]},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    self.local.response = bytearray(struct.pack("!Q", len(response)) + response)
                finally:
                    with self.guard:
                        self.active -= 1

            def recv(self, count):
                response = self.local.response
                block = bytes(response[:count])
                del response[:count]
                return block

            def close(self):
                self.closed = True

        client = _Image3FLoopbackClient("127.0.0.1", 1)
        stream = ProbeStream()
        client._stream = stream
        barrier = threading.Barrier(4)
        outputs = {}

        def invoke(index):
            barrier.wait(timeout=5)
            payload = pickle.dumps(
                {"request_id": index}, protocol=pickle.HIGHEST_PROTOCOL
            )
            outputs[index] = client._exchange(payload, max_response_bytes=1024)

        threads = [threading.Thread(target=invoke, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        client._close()

        self.assertEqual(stream.maximum_active, 1)
        self.assertEqual(
            outputs, {index: {"request_id": index} for index in range(4)}
        )

    def test_single_worker_pool_never_overlaps_backend_calls(self):
        active = 0
        maximum_active = 0
        guard = threading.Lock()

        class ProbeBackend:
            def infer(self, metadata, arrays, checkpoint_sha256):
                nonlocal active, maximum_active
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    time.sleep(0.05)
                    return checkpoint_sha256
                finally:
                    with guard:
                        active -= 1

        pool = RenderStudentBackendPool([ProbeBackend()])
        barrier = threading.Barrier(4)
        outputs = []

        def invoke():
            barrier.wait(timeout=5)
            outputs.append(pool.infer({}, {}, ZERO_SHA))

        threads = [threading.Thread(target=invoke) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(pool.size, 1)
        self.assertEqual(maximum_active, 1)
        self.assertEqual(outputs, [ZERO_SHA] * 4)

    def test_worker_pool_dispatches_unrelated_actors_concurrently(self):
        barrier = threading.Barrier(2)

        class RendezvousBackend(FakeRenderStudentBackend):
            def infer(self, metadata, arrays, checkpoint_sha256):
                barrier.wait(timeout=5)
                return super().infer(metadata, arrays, checkpoint_sha256)

        pool = RenderStudentBackendPool([RendezvousBackend(), RendezvousBackend()])
        self.assertEqual(pool.size, 2)
        results = []
        failures = []
        with service(self.root, pool) as (client, _, _):
            def invoke(actor: str):
                key = step_key()
                key["actor_id"] = actor
                key["rollout_window_id"] = f"window-{actor}"
                try:
                    results.append(client.post(
                        "/v1/steps/prepare",
                        prepare_payload(key, identity=f"prepare-{actor}-0"),
                    ))
                except Exception as exc:  # pragma: no cover - assertion reports it
                    failures.append(exc)

            threads = [threading.Thread(target=invoke, args=(f"actor{i}",)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertFalse(failures)
        self.assertEqual(len(results), 2)

    def test_eight_worker_pool_exposes_all_inference_slots(self):
        barrier = threading.Barrier(8)
        active = 0
        maximum = 0
        counter_lock = threading.Lock()
        result = object()

        class ProbeBackend:
            def ready(self):
                return True, "ready"

            def infer(self, metadata, arrays, checkpoint_sha256):
                nonlocal active, maximum
                with counter_lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    barrier.wait(timeout=5)
                    return result
                finally:
                    with counter_lock:
                        active -= 1

        pool = RenderStudentBackendPool([ProbeBackend() for _ in range(8)])
        outputs = []
        threads = [
            threading.Thread(target=lambda: outputs.append(pool.infer({}, {}, ZERO_SHA)))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(pool.ready(), (True, "8 render/student workers ready"))
        self.assertEqual(maximum, 8)
        self.assertEqual(outputs, [result] * 8)

    def test_concurrent_identical_prepare_has_one_inference(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingBackend(FakeRenderStudentBackend):
            def infer(self, metadata, arrays, checkpoint_sha256):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test timed out")
                return super().infer(metadata, arrays, checkpoint_sha256)

        backend = BlockingBackend()
        payload = prepare_payload(step_key())
        results = []
        failures = []
        with service(self.root, backend) as (client, _, _):
            def invoke():
                try:
                    results.append(client.post("/v1/steps/prepare", payload))
                except Exception as exc:  # pragma: no cover - assertion reports it
                    failures.append(exc)

            first = threading.Thread(target=invoke)
            second = threading.Thread(target=invoke)
            first.start()
            self.assertTrue(entered.wait(timeout=5))
            second.start()
            release.set()
            first.join(timeout=10)
            second.join(timeout=10)
        self.assertFalse(failures)
        self.assertEqual(results[0], results[1])
        self.assertEqual(backend.calls, 1)

    def test_prepare_is_idempotent_and_constructs_26d_action(self):
        with service(self.root) as (client, backend, _):
            payload = prepare_payload(step_key())
            first = client.post("/v1/steps/prepare", payload)
            second = client.post("/v1/steps/prepare", payload)
            self.assertEqual(first, second)
            self.assertEqual(backend.calls, 1)
            arrays = {name: decode_array(value) for name, value in payload["arrays"].items()}
            self.assertEqual(
                first["observation_digest"],
                observation_input_digest(payload["metadata"], arrays),
            )
            self.assertEqual(decode_array(first["student_action_11"]).shape, (11,))
            selected = decode_array(first["selected_action"])
            self.assertEqual(selected.shape, (26,))
            self.assertEqual(first["selected_source"], "teacher" if first["selection_draw"] < .75 else "student")

            changed = prepare_payload(step_key(), identity=payload["idempotency_key"])
            changed["metadata"]["prompt"] = "Changed but otherwise valid."
            arrays = {name: decode_array(value) for name, value in changed["arrays"].items()}
            changed["capture_digest"] = arrays_digest(changed["metadata"], arrays)
            with self.assertRaisesRegex(ServiceError, "409 conflict"):
                client.post("/v1/steps/prepare", changed)

    def test_commit_is_idempotent_and_ordering_tracks_resets(self):
        with service(self.root) as (client, backend, _):
            p0 = client.post("/v1/steps/prepare", prepare_payload(step_key()))
            with self.assertRaisesRegex(ServiceError, "409 conflict"):
                client.post("/v1/steps/prepare", prepare_payload(step_key(1)))
            bad_commit = commit_payload(p0, done=True)
            bad_commit["commit_receipt"] = "f" * 64
            with self.assertRaisesRegex(ServiceError, "commit_receipt"):
                client.post("/v1/steps/commit", bad_commit)
            commit = commit_payload(p0, done=True)
            self.assertNotEqual(commit["idempotency_key"], p0["idempotency_key"])
            c0 = client.post("/v1/steps/commit", commit)
            self.assertEqual(c0, client.post("/v1/steps/commit", commit))
            second_key = dict(commit)
            second_key["idempotency_key"] = "commit-same-selection-new-key"
            with self.assertRaisesRegex(ServiceError, "already committed"):
                client.post("/v1/steps/commit", second_key)

            wrong = prepare_payload(step_key(1))
            with self.assertRaisesRegex(ServiceError, "new episode"):
                client.post("/v1/steps/prepare", wrong)
            reset = prepare_payload(step_key(1, episode="episode1", episode_index=0))
            p1 = client.post("/v1/steps/prepare", reset)
            self.assertEqual(p1["step_key"]["episode_id"], "episode1")
            self.assertEqual(backend.calls, 2)

    def test_round_four_never_selects_teacher(self):
        with service(self.root) as (client, _, _):
            result = client.post("/v1/steps/prepare", prepare_payload(step_key(round_id=4)))
            self.assertEqual(result["beta"], 0.0)
            self.assertEqual(result["selected_source"], "student")
            student = decode_array(result["student_action_11"])
            selected = decode_array(result["selected_action"])
            np.testing.assert_array_equal(selected[:11], student)
            np.testing.assert_array_equal(selected[11:], capture_arrays()["homie_action"])

    def test_raytraced_views_are_policy_resolution(self):
        with service(self.root) as (client, _, _):
            result = client.post(
                "/v1/steps/prepare", prepare_payload(step_key(), source="raytraced")
            )
            self.assertEqual(decode_array(result["student_action_11"]).dtype, np.float32)
            artifact = next((self.root / "state" / "prepared").glob("*.npz"))
            with np.load(artifact, allow_pickle=False) as data:
                for view in VIEWS:
                    self.assertEqual(data[f"rgb_gen__{view}"].dtype, np.float32)
                    self.assertGreaterEqual(float(data[f"rgb_gen__{view}"].min()), 0.0)
                    self.assertLessEqual(float(data[f"rgb_gen__{view}"].max()), 1.0)

    def test_raytraced_views_reject_lossy_uint8_wire_data(self):
        with service(self.root) as (client, backend, _):
            payload = prepare_payload(step_key(), source="raytraced")
            payload["arrays"]["raytraced_rgb__head_color"] = encode_array(
                np.zeros((108, 192, 3), np.uint8)
            )
            arrays = {name: decode_array(value) for name, value in payload["arrays"].items()}
            payload["capture_digest"] = arrays_digest(payload["metadata"], arrays)
            with self.assertRaisesRegex(ServiceError, "float32"):
                client.post("/v1/steps/prepare", payload)
            self.assertEqual(backend.calls, 0)

    def test_anchor_bytes_are_bound_to_episode_digest(self):
        with service(self.root) as (client, backend, _):
            payload = prepare_payload(step_key())
            payload["metadata"]["episode_assignment"]["anchor_sha256"] = "f" * 64
            arrays = {name: decode_array(value) for name, value in payload["arrays"].items()}
            payload["capture_digest"] = arrays_digest(payload["metadata"], arrays)
            with self.assertRaisesRegex(ServiceError, "anchor RGB bytes"):
                client.post("/v1/steps/prepare", payload)
            self.assertEqual(backend.calls, 0)

    def test_full_window_sealer_preserves_cross_episode_identity(self):
        backend = FakeRenderStudentBackend()
        backend.test_checkpoint_sha256 = ZERO_SHA
        sink = LocalTrajectorySink(self.root / "published")
        server = RenderStudentHTTPServer(
            ("127.0.0.1", 0), Handler, root=self.root / "state", token=TOKEN,
            backend=backend, trajectory_sink=sink,
        )
        try:
            artifact = self.root / "artifact.npz"
            base = capture_arrays()
            with artifact.open("wb") as stream:
                np.savez_compressed(
                    stream, **base,
                    **{f"rgb_gen__{view}": np.zeros((108, 192, 3), np.float32) for view in VIEWS},
                    student_action_11=np.zeros(11, np.float32),
                    student_action=np.concatenate((np.zeros(11, np.float32), base["homie_action"])),
                    behavior_action=base["teacher_action"],
                    behavior_is_teacher=np.array(True, np.bool_), beta=np.array(.75, np.float32),
                )
            now = 1.0
            selection_ids = []
            commit_digests = []
            commit_receipts = []
            with server.store.transaction() as db:
                for index in range(81):
                    episode = "episode0" if index < 40 else "episode1"
                    episode_index = index if index < 40 else index - 40
                    key = step_key(index, episode=episode, episode_index=episode_index)
                    meta = metadata(key)
                    identity = f"direct-{index:03d}"
                    selection_id = f"{index:064x}"
                    selection = {
                        "step_key": key,
                        "prepare_receipt": "3" * 64,
                        "selection_id": selection_id,
                        "selected_source": "teacher",
                        "beta": 0.75,
                        "selected_action": encode_array(base["teacher_action"]),
                    }
                    commit_digest = bytes_sha256(canonical_bytes(selection))
                    transition = {
                        "reward": 0.0, "done": index == 39, "timeout": False, "metadata": {}
                    }
                    commit_receipt = bytes_sha256(canonical_bytes({
                        "commit_digest": commit_digest, "result": transition,
                    }))
                    selection_ids.append(selection_id)
                    commit_digests.append(commit_digest)
                    commit_receipts.append(commit_receipt)
                    db.execute(
                        "INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (identity, "1" * 64, key["run_id"], key["round_id"], key["actor_id"],
                         key["rollout_window_id"], index, episode, episode_index, "2" * 64,
                         "3" * 64, json.dumps(meta, sort_keys=True, separators=(",", ":")),
                         json.dumps(selection, sort_keys=True, separators=(",", ":")),
                         str(artifact), ZERO_SHA, "committed", now, now),
                    )
                    db.execute(
                        "INSERT INTO step_commits VALUES (?,?,?,?,?,?,?,?)",
                        (f"commit-{identity}", "4" * 64, identity,
                         json.dumps(transition, sort_keys=True, separators=(",", ":")),
                         None, "{}", now, now),
                    )
            sealed = {
                "schema": "roborender.raven.dagger-window.v1",
                "contract_version": STEP_CONTRACT,
                "run_id": "run-smoke", "round_id": 1, "actor_id": "actor0",
                "rollout_window_id": "window1", "row_count": 81,
                "artifact_role": "rtx_identity_outcome_audit",
                "learner_complete": False, "contains_policy_rgb": False,
                "authoritative_learner_chunk_owner": "h200_render_student_ledger",
                "episode_ids_are_per_step": True, "actual_dones_preserved": True,
                "no_previous_round_replay": True, "view_order": list(VIEWS),
                "per_step_identity_fields": [
                    "episode_ids", "episode_step_indices", "rollout_step_indices", "round_ids",
                    "env_id_per_step", "seq_per_step", "observation_source", "dones", "timeouts",
                    "rewards", "behavior_source", "prepare_receipts", "capture_digests",
                    "selection_ids", "commit_digests", "commit_receipts",
                ],
                "episode_ids": ["episode0" if index < 40 else "episode1" for index in range(81)],
                "episode_step_indices": [index if index < 40 else index - 40 for index in range(81)],
                "env_ids": [0] * 81,
                "seqs": [0] * 81,
                "round_ids": [1] * 81,
                "observation_sources": ["generated_controls"] * 81,
                "dones": [index == 39 for index in range(81)],
                "timeouts": [False] * 81,
                "rewards": [0.0] * 81,
                "behavior_sources": ["teacher"] * 81,
                "prepare_receipts": ["3" * 64] * 81,
                "capture_digests": ["2" * 64] * 81,
                "selection_ids": selection_ids,
                "commit_digests": commit_digests,
                "commit_receipts": commit_receipts,
                "path": "sealed.npz",
                "bytes": 123, "sha256": "5" * 64,
                "provenance": {
                    "raven_root": "/home/ldu/BC_render/rtx_BC_transfer",
                    "raven_commit": RAVEN_COMMIT,
                    "raven_videogen_trainer_sha256": "6" * 64,
                    "raven_teacher_homie_trainer_sha256": "7" * 64,
                    "isaac_capture_sha256": "b" * 64,
                    "image_dagger_config_sha256": "c" * 64,
                    "teacher_checkpoint_sha256": "8" * 64,
                    "homie_walk_sha256": "9" * 64,
                    "homie_stand_sha256": "a" * 64,
                    "episode_bindings": [
                        {"episode_id": "episode0", "first_rollout_step_index": 0,
                         "use_raytraced": False, "observation_source": "generated_controls",
                         "anchor_sha256": "a" * 64},
                        {"episode_id": "episode1", "first_rollout_step_index": 40,
                         "use_raytraced": False, "observation_source": "generated_controls",
                         "anchor_sha256": "a" * 64},
                    ],
                },
            }
            handler = object.__new__(Handler)
            handler.server = server
            handler._seal_window(
                StepKey.parse(step_key(80, episode="episode1", episode_index=40)), sealed
            )
            self.assertEqual(sink.calls, 1)
            chunk = next((self.root / "published").glob("*.npz"))
            report = validate_dagger_chunk(chunk)
            self.assertEqual(report["steps"], 81)
            with np.load(chunk, allow_pickle=False) as data:
                self.assertEqual(
                    str(data["learner_chunk_format"].item()),
                    "roborender.dagger.learner.v1",
                )
                self.assertEqual(
                    set(data.files),
                    {
                        "learner_chunk_format", "gt_actions", "obs_actor_obs",
                        *(f"rgb_gen__{view}" for view in VIEWS),
                    },
                )
            # Cross-episode identity and outcomes remain authoritative in the
            # compact ledger/seal; the gradient artifact intentionally does
            # not duplicate them.
            with server.store.connection() as db:
                rows = list(db.execute(
                    "SELECT p.episode_id,c.transition_json FROM step_prepares p "
                    "JOIN step_commits c ON c.prepare_idempotency_key=p.idempotency_key "
                    "ORDER BY p.rollout_step_index"
                ))
            self.assertEqual(rows[39]["episode_id"], "episode0")
            self.assertEqual(rows[40]["episode_id"], "episode1")
            self.assertTrue(json.loads(rows[39]["transition_json"])["done"])
        finally:
            server.server_close()

    def test_learner_preserves_float01_policy_rgb_scale(self):
        import torch

        path = self.root / "learner-window.npz"
        arrays = {
            "obs_actor_obs": np.zeros((81, 113), np.float32),
            "gt_actions": np.zeros((81, 26), np.float32),
        }
        for view in VIEWS:
            arrays[f"rgb_gen__{view}"] = np.full((81, 108, 192, 3), 0.5, np.float32)
        np.savez_compressed(path, **arrays)
        spec = {
            "input_contract": {
                "views": list(VIEWS),
                "rgb_mean": [0.5, 0.5, 0.5],
                "rgb_std": [0.25, 0.25, 0.25],
            },
            "output_contract": {"policy_action_dim": 11},
        }
        vision, actor, target = load_batch([path], torch.device("cpu"), spec)
        self.assertEqual(tuple(vision.shape), (1, 81, 324, 192, 3))
        self.assertTrue(torch.equal(vision, torch.zeros_like(vision)))
        self.assertEqual(tuple(actor.shape), (1, 81, 113))
        self.assertEqual(tuple(target.shape), (1, 81, 11))

    def test_learner_microbatches_preserve_one_global_mse_gradient(self):
        import torch

        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(4, 2)

            def forward(self, obs):
                image_feature = obs["vision_obs"].mean(dim=(2, 3, 4)).unsqueeze(-1)
                features = torch.cat((obs["actor_obs"], image_feature), dim=-1)
                return {"actions": self.head(features)}

        generator = torch.Generator().manual_seed(17)
        vision = torch.randn((5, 3, 2, 2, 3), generator=generator)
        actor = torch.randn((5, 3, 3), generator=generator)
        target = torch.randn((5, 3, 2), generator=generator)
        reference = TinyPolicy()
        microbatched = TinyPolicy()
        microbatched.load_state_dict(reference.state_dict())

        reference_loss = torch.nn.functional.mse_loss(
            reference({"vision_obs": vision, "actor_obs": actor})["actions"], target
        )
        reference_loss.backward()
        measured = backward_global_mse_in_environment_microbatches(
            microbatched, vision, actor, target, microbatch_environments=2
        )

        self.assertAlmostEqual(measured, float(reference_loss.detach()), places=6)
        for expected, actual in zip(reference.parameters(), microbatched.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)

    def test_generated_policy_resize_is_exact_raven_tiled_torch_path(self):
        import torch
        import torch.nn.functional as functional

        rng = np.random.default_rng(7)
        source = {
            view: rng.integers(0, 256, (240, 416, 3), dtype=np.uint8)
            for view in VIEWS
        }
        actual = LiveRenderStudentBackend._policy_size(source)
        tiled = np.concatenate([source[view] for view in VIEWS], axis=0)
        expected = torch.from_numpy(tiled.copy()).float().div_(255.0)
        expected = functional.interpolate(
            expected.permute(2, 0, 1).unsqueeze(0),
            size=(324, 192), mode="bilinear", align_corners=False,
        )[0].permute(1, 2, 0).numpy()
        for index, view in enumerate(VIEWS):
            np.testing.assert_array_equal(actual[view], expected[index * 108:(index + 1) * 108])


if __name__ == "__main__":
    unittest.main()
