"""Production gates for retry-safe Image3F actor stream lifecycle."""

from __future__ import annotations

import itertools
import os
import pickle
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger.actor_pair_terminal import publish_actor_pair_terminal
from integrations.gr00t.image3f_frame_client import Image3FFrameClient
from round_dagger.image3f_lifecycle import (
    Image3FLifecycleError,
    _discover_plans,
    cleanup_prior_attempts,
    cleanup_sealed_cohort,
    create_attempt_plan,
    record_rank_completion,
    recover_completed_rank_ownership,
    verify_cleanup_report,
)
from tests.test_round_dagger_actor_pair import DIGESTS, _arrays, _cohort


class ResetServer:
    def __init__(self, *, server_id: str, renderer_sha256: str):
        self.server_id = server_id
        self.renderer_sha256 = renderer_sha256
        self.streams: set[str] = set()
        self.reject_streams: set[str] = set()
        self.requests: list[str] = []
        self._stop = threading.Event()
        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(128)
        self._socket.settimeout(0.1)
        self.port = int(self._socket.getsockname()[1])
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = sock.recv(size - len(result))
            if not chunk:
                raise ConnectionError("client closed")
            result.extend(chunk)
        return bytes(result)

    @classmethod
    def _recv(cls, sock: socket.socket) -> dict:
        size = struct.unpack("!Q", cls._recv_exact(sock, 8))[0]
        return pickle.loads(cls._recv_exact(sock, size))  # noqa: S301 - test protocol

    @staticmethod
    def _send(sock: socket.socket, document: dict) -> None:
        payload = pickle.dumps(document, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendall(struct.pack("!Q", len(payload)) + payload)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except (TimeoutError, OSError):
                continue
            with connection:
                try:
                    request = self._recv(connection)
                except (ConnectionError, OSError):
                    continue
                stream_id = str(request.get("stream_id", ""))
                if request.get("cmd") == "ping":
                    self._send(
                        connection,
                        {
                            "ok": True,
                            "protocol_version": 3,
                            "server_id": self.server_id,
                            "renderer_identity_sha256": self.renderer_sha256,
                        },
                    )
                    continue
                self.requests.append(stream_id)
                if stream_id in self.reject_streams:
                    self._send(
                        connection,
                        {
                            "ok": False,
                            "error": "injected reset failure",
                            "stream_id": stream_id,
                            "server_id": self.server_id,
                            "renderer_identity_sha256": self.renderer_sha256,
                        },
                    )
                    continue
                removed = stream_id in self.streams
                self.streams.discard(stream_id)
                self._send(
                    connection,
                    {
                        "ok": True,
                        "removed": removed,
                    },
                )

    def stop(self) -> None:
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=2)


class Image3FStreamLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.cohort = self.root / "cohort"
        _cohort(self.cohort)
        self.train = self.root / "actor_run"
        (self.train / "actor_pair_attempts").mkdir(parents=True)
        self.servers = [
            ResetServer(server_id=f"renderer:{index}", renderer_sha256=DIGESTS["renderer_sha256"])
            for index in range(2)
        ]
        for server in self.servers:
            server.start()
        self.endpoints: list[Path] = []
        for index, server in enumerate(self.servers):
            endpoint = self.root / f"renderer_{index:03d}.txt"
            endpoint.write_text(f"127.0.0.1 {server.port}\n", encoding="utf-8")
            self.endpoints.append(endpoint)
        self.manifest = self.root / "endpoint_manifest.txt"
        self.manifest.write_text(
            "".join(f"{endpoint}\n" for endpoint in self.endpoints), encoding="utf-8"
        )

    def tearDown(self) -> None:
        for server in self.servers:
            server.stop()
        self.temporary.cleanup()

    def _plan(self, attempt: int, ranks: list[int]) -> tuple[Path, dict]:
        attempt_dir = self.train / "actor_pair_attempts" / f"attempt_{attempt:06d}"
        attempt_dir.mkdir()
        plan_path = attempt_dir / "image3f_stream_plan.json"
        plan = create_attempt_plan(
            plan_path,
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            attempt_dir=attempt_dir,
            cohort_dir=self.cohort,
            endpoint_manifest=self.manifest,
            expected_renderers=2,
            renderer_sha256=DIGESTS["renderer_sha256"],
            local_environments=32,
            actor_world_size=2,
            launched_ranks=ranks,
        )
        return plan_path, plan

    def _publish_rank(self, rank: int) -> None:
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=rank,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(rank),
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            **DIGESTS,
        )

    def _complete(self, plan_path: Path, rank: int) -> None:
        record_rank_completion(
            plan_path.parent / f"rank_{rank:04d}_stream_completion.json",
            plan_path=plan_path,
            cohort_dir=self.cohort,
            rank=rank,
        )

    def test_retry_namespace_is_unique_and_endpoint_mapping_matches_client(self) -> None:
        _, first = self._plan(1, [0, 1])
        _, retry = self._plan(2, [1])
        self.assertNotEqual(first["attempt_namespace"], retry["attempt_namespace"])
        first_ids = {row["stream_id"] for row in first["streams"]}
        retry_ids = {row["stream_id"] for row in retry["streams"]}
        self.assertTrue(first_ids.isdisjoint(retry_ids))
        self.assertEqual(len(first_ids), 64)
        self.assertEqual(len(retry_ids), 32)
        for plan in (first, retry):
            for row in plan["streams"]:
                self.assertEqual(row["endpoint_index"], row["global_slot"] % 2)
                self.assertEqual(
                    Path(row["endpoint_file"]), self.endpoints[row["endpoint_index"]]
                )
                self.assertEqual(
                    row["stream_id"],
                    f"{plan['attempt_namespace']}:rank{row['rank']}:env{row['global_slot']}",
                )

        # Exercise the actual production client allocator, rather than merely
        # restating its formula in this lifecycle test.
        for rank in (0, 1):
            Image3FFrameClient._counter = itertools.count()
            with mock.patch.dict(
                os.environ,
                {
                    "IMAGE3F_ENDPOINT_MANIFEST": str(self.manifest),
                    "IMAGE3F_ENDPOINT_DIR": "",
                    "IMAGE3F_LOCAL_ENVS": "32",
                    "IMAGE3F_RUN_ID": first["attempt_namespace"],
                    "ROUND_DAGGER_ACTOR_RANK": str(rank),
                },
                clear=False,
            ):
                clients = [
                    Image3FFrameClient.from_endpoint_file(self.endpoints[0])
                    for _ in range(32)
                ]
            expected = [row for row in first["streams"] if row["rank"] == rank]
            self.assertEqual(
                [(client.endpoint_file, client._stream_id) for client in clients],
                [(row["endpoint_file"], row["stream_id"]) for row in expected],
            )

    def test_empty_retry_reconciliation_prevents_ambiguous_ownership(self) -> None:
        first_path, first = self._plan(1, [0, 1])
        for row in first["streams"]:
            self.servers[row["endpoint_index"]].streams.add(row["stream_id"])
        recovered = recover_completed_rank_ownership(
            self.train,
            cohort_dir=self.cohort,
            endpoint_manifest=self.manifest,
            expected_renderers=2,
            renderer_sha256=DIGESTS["renderer_sha256"],
            completed_ranks=[],
        )
        self.assertEqual(recovered["completed_ranks"], [])
        self.assertTrue(
            (first_path.parent / "rank_0000_stream_abandonment.json").is_file()
        )
        self.assertTrue(
            (first_path.parent / "rank_0001_stream_abandonment.json").is_file()
        )
        cleanup = cleanup_prior_attempts(self.train, timeout_s=1, retries=0)
        self.assertEqual(cleanup["planned_stream_count"], 64)
        self.assertEqual(cleanup["acknowledged_stream_count"], 64)
        self.assertTrue(all(not server.streams for server in self.servers))

        self._plan(2, [0])
        self._publish_rank(0)
        recovered = recover_completed_rank_ownership(
            self.train,
            cohort_dir=self.cohort,
            endpoint_manifest=self.manifest,
            expected_renderers=2,
            renderer_sha256=DIGESTS["renderer_sha256"],
            completed_ranks=[0],
        )
        self.assertEqual(recovered["owned_ranks"], [0])
        self.assertEqual(recovered["recovered_receipts"], 1)

    def test_partial_retry_cleans_successful_64_and_abandoned_streams_idempotently(self) -> None:
        first_path, first = self._plan(1, [0, 1])
        retry_path, retry = self._plan(2, [1])
        self._publish_rank(0)
        self._complete(first_path, 0)
        self._publish_rank(1)
        self._complete(retry_path, 1)
        self.assertFalse((self.cohort / "seal.json").exists())

        all_rows = first["streams"] + retry["streams"]
        for row in all_rows:
            self.servers[row["endpoint_index"]].streams.add(row["stream_id"])
        # Prove an already-absent stream is still a valid idempotent ACK.
        absent = all_rows[0]
        self.servers[absent["endpoint_index"]].streams.remove(absent["stream_id"])

        first_report = cleanup_sealed_cohort(
            self.train, cohort_dir=self.cohort, timeout_s=1, retries=0
        )
        self.assertEqual(first_report["successful_stream_count"], 64)
        self.assertEqual(first_report["planned_stream_count"], 96)
        self.assertEqual(first_report["acknowledged_stream_count"], 96)
        self.assertEqual(
            verify_cleanup_report(self.train, cohort_dir=self.cohort)[
                "report_sha256"
            ],
            first_report["report_sha256"],
        )
        self.assertTrue(all(not server.streams for server in self.servers))
        self.assertFalse((self.cohort / "seal.json").exists())
        artifacts.seal_cohort(self.cohort)
        requests_after_first = [list(server.requests) for server in self.servers]

        second_report = cleanup_sealed_cohort(
            self.train, cohort_dir=self.cohort, timeout_s=1, retries=0
        )
        self.assertEqual(second_report["report_sha256"], first_report["report_sha256"])
        self.assertEqual(
            [server.requests for server in self.servers], requests_after_first
        )

    def test_cleanup_failure_records_prior_acks_and_resumes(self) -> None:
        plan_path, plan = self._plan(1, [0, 1])
        for rank in (0, 1):
            self._publish_rank(rank)
            self._complete(plan_path, rank)
        self.assertFalse((self.cohort / "seal.json").exists())
        for row in plan["streams"]:
            self.servers[row["endpoint_index"]].streams.add(row["stream_id"])

        rejected = plan["streams"][1]
        self.servers[rejected["endpoint_index"]].reject_streams.add(rejected["stream_id"])
        with self.assertRaisesRegex(Image3FLifecycleError, "injected reset failure"):
            cleanup_sealed_cohort(
                self.train, cohort_dir=self.cohort, timeout_s=1, retries=0
            )
        ack_dir = self.train / "image3f_stream_cleanup"
        self.assertEqual(len(list(ack_dir.glob("reset_*.json"))), 1)
        self.assertFalse((ack_dir / "cleanup_complete.json").exists())
        self.assertFalse((self.cohort / "seal.json").exists())

        self.servers[rejected["endpoint_index"]].reject_streams.clear()
        report = cleanup_sealed_cohort(
            self.train, cohort_dir=self.cohort, timeout_s=1, retries=0
        )
        self.assertEqual(report["successful_stream_count"], 64)
        self.assertEqual(report["planned_stream_count"], 64)
        self.assertEqual(len(list(ack_dir.glob("reset_*.json"))), 64)
        self.assertFalse((self.cohort / "seal.json").exists())
        artifacts.seal_cohort(self.cohort)

    def test_legacy_terminal_only_recovery_is_ignored_but_not_loosened(self) -> None:
        """A historical W&B terminal cannot poison later stream cleanup."""

        for rank in (0, 1):
            self._publish_rank(rank)
        artifacts.seal_cohort(self.cohort)
        recovery = (
            self.train
            / "actor_pair_attempts"
            / "wandb_recovery_20260831T065831Z_1590047_yhYPPsth"
        )
        recovery.mkdir()
        publish_actor_pair_terminal(
            attempt_dir=recovery,
            run_dir=self.train,
            cohort_dir=self.cohort,
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            rank0_exit_code=0,
            rank1_exit_code=0,
            lifecycle_exit_code=0,
            quality_exit_code=0,
            wandb_wrapper_exit_code=0,
            delivery_pending=True,
        )

        self.assertEqual(_discover_plans(self.train, allow_empty=True), [])

        (recovery / "unexpected.txt").write_text("must remain fail-closed\n")
        with self.assertRaisesRegex(Image3FLifecycleError, "unexpected contents"):
            _discover_plans(self.train, allow_empty=True)


if __name__ == "__main__":
    unittest.main()
