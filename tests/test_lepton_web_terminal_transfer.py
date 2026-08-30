from __future__ import annotations

import base64
import collections
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.round_dagger import lepton_web_terminal_transfer as transfer


def identity(payload: bytes) -> transfer.FileIdentity:
    return transfer.FileIdentity(len(payload), hashlib.sha256(payload).hexdigest())


class FakeStore:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.cleaned: list[str] = []
        self.created: list[str] = []
        self.published: list[tuple[str, str]] = []

    def inspect(self, path: str) -> transfer.FileIdentity | None:
        value = self.files.get(path)
        return None if value is None else identity(value)

    def create_incoming(self, path: str, mode: int) -> None:
        if path in self.files:
            raise transfer.TransferError("collision")
        self.created.append(path)
        self.files[path] = b""

    def append(self, path: str, payload: bytes) -> None:
        self.files[path] += payload

    def read_chunk(self, path: str, offset: int, length: int) -> bytes:
        return self.files[path][offset : offset + length]

    def publish(
        self,
        incoming: str,
        destination: str,
        expected: transfer.FileIdentity,
        mode: int,
    ) -> None:
        if destination in self.files:
            raise transfer.TransferError("destination exists")
        if identity(self.files[incoming]) != expected:
            raise transfer.TransferError("identity mismatch")
        self.files[destination] = self.files.pop(incoming)
        self.published.append((incoming, destination))

    def cleanup_incoming(self, path: str) -> None:
        self.cleaned.append(path)
        self.files.pop(path, None)


class FragmentedTransport:
    def __init__(self, data: tuple[bytes, ...] = (), return_code: int = 0) -> None:
        self.data = data
        self.return_code = return_code
        self.sent: bytes | None = None
        self.fragments: list[bytes] = []
        self.closed = False

    def send(self, payload: bytes) -> None:
        self.sent = payload
        text = payload.decode("utf-8")
        match = __import__("re").search(r"LXFT1:([0-9a-f]+):PID", text)
        assert match is not None
        nonce = match.group(1).encode("ascii")

        def frame(kind: bytes, value: bytes) -> bytes:
            return b"\x1eLXFT1:" + nonce + b":" + kind + b":" + value + b"\x1f\r\n"

        response = b"shell prompt and echoed command\r\n"
        response += frame(b"PID", b"123")
        response += frame(b"WORKER", b"124")
        for item in self.data:
            response += frame(b"DATA", base64.b64encode(item))
        response += frame(b"DONE", str(self.return_code).encode("ascii"))
        self.fragments = [response[:7], response[7:31], response[31:53], response[53:]]

    def receive(self, timeout_seconds: float) -> bytes:
        return self.fragments.pop(0) if self.fragments else b""

    def close(self) -> None:
        self.closed = True


class CaptureShell:
    def __init__(self, records: tuple[bytes, ...] = ()) -> None:
        self.records = records
        self.scripts: list[str] = []

    def run(self, script: str, *, timeout_seconds: float | None = None) -> transfer.CommandResult:
        self.scripts.append(script)
        result = transfer.CommandResult(1, 2, self.records)
        self.records = ()
        return result


class StreamingTransport:
    def __init__(
        self,
        expected: bytes,
        *,
        initial: bytes = b"",
        store: FakeStore | None = None,
        destination: str | None = None,
        corrupt_prefix: bool = False,
    ) -> None:
        self.expected = expected
        self.received = bytearray(initial)
        self.store = store
        self.destination = destination
        self.corrupt_prefix = corrupt_prefix
        self.sent: list[bytes] = []
        self.responses: list[bytes] = []
        self.state = "setup"
        self.aborted = False

    @staticmethod
    def frame(kind: bytes, value: bytes) -> bytes:
        return (
            b"\x1eLXFS1:0123456789abcdef0123456789abcdef:"
            + kind
            + b":"
            + value
            + b"\x1f\r\n"
        )

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)
        if self.state == "setup":
            self.assert_setup(payload)
            prefix_sha = hashlib.sha256(self.received).hexdigest().encode("ascii")
            if self.corrupt_prefix:
                prefix_sha = b"0" * 64
            ready = (
                str(len(self.received)).encode("ascii")
                + b":"
                + prefix_sha
                + b":"
                + str(len(self.expected) - len(self.received)).encode("ascii")
            )
            response = b"prompt/echo noise" + self.frame(b"PID", b"321")
            response += self.frame(b"READY", ready)
            self.responses.extend([response[:13], response[13:47], response[47:]])
            self.state = "go"
            return
        if self.state == "go":
            if payload == b"\x00A":
                self.aborted = True
                self.received.clear()
                self.responses.append(self.frame(b"DONE", b"46:-:-"))
                self.state = "done"
                return
            if payload != b"\x00G":
                raise AssertionError("stream did not send its commit marker")
            self.state = "data"
            if len(self.received) == len(self.expected):
                self.finish()
            return
        if self.state != "data" or not payload.startswith(b"\x00"):
            raise AssertionError("unexpected streaming transport payload")
        self.received.extend(payload[1:])
        if len(self.received) > len(self.expected):
            raise AssertionError("stream sent more than the declared fixed size")
        if len(self.received) == len(self.expected):
            self.finish()

    def assert_setup(self, payload: bytes) -> None:
        if not payload.startswith(b"\x00"):
            raise AssertionError("setup was not sent on Kubernetes stdin channel 0")
        if b"/workspace/projects/roborender" in payload:
            raise AssertionError("setup command exposed decoded paths to PTY echo")

    def finish(self) -> None:
        observed = bytes(self.received)
        if observed != self.expected:
            raise AssertionError("mock receiver got incorrect stream bytes")
        if self.store is not None and self.destination is not None:
            self.store.files[self.destination] = observed
        value = (
            b"0:"
            + str(len(observed)).encode("ascii")
            + b":"
            + hashlib.sha256(observed).hexdigest().encode("ascii")
        )
        self.responses.append(self.frame(b"DONE", value))
        self.state = "done"

    def receive(self, timeout_seconds: float) -> bytes:
        return self.responses.pop(0) if self.responses else b""

    def close(self) -> None:
        pass


class DownloadingTransport:
    def __init__(
        self,
        payload: bytes,
        *,
        ready_identity: transfer.FileIdentity | None = None,
    ) -> None:
        self.payload = payload
        self.ready_identity = ready_identity or identity(payload)
        self.sent: list[bytes] = []
        self.responses: list[bytes] = []
        self.nonce = b""
        self.aborted = False
        self.state = "setup"

    def frame(self, kind: bytes, value: bytes) -> bytes:
        return (
            b"\x1eLXFD1:"
            + self.nonce
            + b":"
            + kind
            + b":"
            + value
            + b"\x1f"
        )

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)
        if self.state == "setup":
            if not payload.startswith(b"\x00"):
                raise AssertionError("download setup did not use stdin channel zero")
            if b"/workspace/projects/roborender" in payload:
                raise AssertionError("download setup exposed its decoded remote path")
            encoded_match = __import__("re").search(
                r"printf '%s' ([A-Za-z0-9+/=]+) \| base64 -d",
                payload.decode("utf-8"),
            )
            assert encoded_match is not None
            decoded = base64.b64decode(encoded_match.group(1)).decode("utf-8")
            match = __import__("re").search(r"LXFD1:([0-9a-f]+):PID", decoded)
            assert match is not None
            self.nonce = match.group(1).encode("ascii")
            ready = (
                str(self.ready_identity.size).encode("ascii")
                + b":"
                + self.ready_identity.sha256.encode("ascii")
            )
            controls = b"prompt/echo noise" + self.frame(b"PID", b"654")
            controls += self.frame(b"READY", ready)
            self.responses.extend([controls[:9], controls[9:41], controls[41:]])
            self.state = "ready"
            return
        if self.state != "ready":
            raise AssertionError("unexpected download transport send")
        if payload == b"\x00A":
            self.aborted = True
            self.responses.append(self.frame(b"DONE", b"46:-:-"))
            self.state = "done"
            return
        if payload != b"\x00G":
            raise AssertionError("download stream omitted commit marker")
        done = (
            b"0:"
            + str(self.ready_identity.size).encode("ascii")
            + b":"
            + self.ready_identity.sha256.encode("ascii")
        )
        combined = self.payload + self.frame(b"DONE", done)
        cuts = (3, 1027, max(1028, len(self.payload) - 7))
        previous = 0
        for cut in cuts:
            cut = min(cut, len(combined))
            if cut > previous:
                self.responses.append(combined[previous:cut])
                previous = cut
        if previous < len(combined):
            self.responses.append(combined[previous:])
        self.state = "done"

    def receive(self, timeout_seconds: float) -> bytes:
        return self.responses.pop(0) if self.responses else b""

    def close(self) -> None:
        pass


class LeptonWebTerminalTransferTest(unittest.TestCase):
    def test_remote_path_is_strictly_confined_and_canonical(self) -> None:
        good = "/workspace/projects/roborender/run/bundle.bin"
        self.assertEqual(transfer.validate_remote_path(good), good)
        for bad in (
            "/workspace/projects/roborender",
            "/workspace/projects/roborender2/value",
            "/workspace/projects/roborender/../escape",
            "/workspace/projects/roborender/run//value",
            "workspace/projects/roborender/value",
            "/tmp/value",
            "/workspace/projects/roborender/value\nnext",
        ):
            with self.subTest(path=bad), self.assertRaises(transfer.TransferError):
                transfer.validate_remote_path(bad)

    def test_remote_shell_parses_fragmented_pid_and_data_frames(self) -> None:
        transport = FragmentedTransport((b"one", b"two"))
        result = transfer.RemoteShell(transport).run("printf secret-payload")
        self.assertEqual(result.pid, 123)
        self.assertEqual(result.worker_pid, 124)
        self.assertEqual(result.data, (b"one", b"two"))
        self.assertIsNotNone(transport.sent)
        self.assertTrue(transport.sent.startswith(b"\x00"))
        self.assertNotIn(b"secret-payload", transport.sent)

    def test_remote_shell_fails_on_nonzero_remote_completion(self) -> None:
        transport = FragmentedTransport(return_code=19)
        with self.assertRaisesRegex(transfer.TransferError, "status 19"):
            transfer.RemoteShell(transport).run("false")

    def test_upload_is_chunked_verified_and_atomically_published(self) -> None:
        store = FakeStore()
        destination = "/workspace/projects/roborender/handoff/bundle.bin"
        payload = bytes(range(251)) * 17
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bundle.bin"
            source.write_bytes(payload)
            report = transfer.TransferEngine(store, chunk_bytes=1024).upload(
                source, destination
            )
        self.assertEqual(store.files[destination], payload)
        self.assertEqual(report["status"], "published")
        self.assertEqual(report["sha256"], identity(payload).sha256)
        self.assertEqual(len(store.published), 1)
        incoming, published = store.published[0]
        self.assertEqual(published, destination)
        self.assertTrue(Path(incoming).name.startswith(".incoming."))
        self.assertNotIn(incoming, store.files)

    def test_streaming_upload_uses_one_setup_then_fixed_count_raw_chunks(self) -> None:
        payload = bytes(range(251)) * 37
        store = FakeStore()
        destination = "/workspace/projects/roborender/handoff/stream.bin"
        transport = StreamingTransport(
            payload, store=store, destination=destination
        )
        streamer = transfer.StreamingUploader(transport, chunk_bytes=1024)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            transfer.secrets,
            "token_hex",
            return_value="0123456789abcdef0123456789abcdef",
        ):
            source = Path(temporary) / "stream.bin"
            source.write_bytes(payload)
            report = transfer.TransferEngine(store).upload_streaming(
                source, destination, streamer
            )
        self.assertEqual(report["status"], "published")
        self.assertEqual(store.files[destination], payload)
        self.assertEqual(transport.sent[1], b"\x00G")
        raw_messages = transport.sent[2:]
        self.assertGreater(len(raw_messages), 1)
        self.assertTrue(all(message.startswith(b"\x00") for message in raw_messages))
        self.assertEqual(sum(len(message) - 1 for message in raw_messages), len(payload))
        self.assertTrue(all(len(message) <= 1025 for message in raw_messages))

    def test_websocket_send_waits_for_ack_without_losing_inbound_data(self) -> None:
        transport = object.__new__(transfer.NodeWebTerminalTransport)
        transport._closed = False
        transport._next_send_id = 1
        transport._stdin = io.BytesIO()
        transport._pending_events = collections.deque()
        inbound = base64.b64encode(b"\x01remote-output").decode("ascii")
        transport._event = mock.Mock(
            side_effect=[
                {"event": "data", "data": inbound},
                {"event": "sent", "id": 1},
            ]
        )
        transfer.NodeWebTerminalTransport.send(transport, b"\x00payload")
        request = __import__("json").loads(transport._stdin.getvalue())
        self.assertEqual(request["id"], 1)
        self.assertEqual(base64.b64decode(request["data"]), b"\x00payload")
        self.assertEqual(
            transfer.NodeWebTerminalTransport.receive(transport, 1.0),
            b"remote-output",
        )

    def test_streaming_upload_resumes_only_after_prefix_hash_match(self) -> None:
        payload = b"resumable-payload-" * 900
        offset = 4097
        transport = StreamingTransport(payload, initial=payload[:offset])
        streamer = transfer.StreamingUploader(transport, chunk_bytes=1024)
        destination = "/workspace/projects/roborender/handoff/resume.bin"
        expected = identity(payload)
        incoming = transfer.resumable_incoming_path(destination, expected)
        with tempfile.TemporaryFile() as handle, mock.patch.object(
            transfer.secrets,
            "token_hex",
            return_value="0123456789abcdef0123456789abcdef",
        ):
            handle.write(payload)
            handle.flush()
            streamer.upload(handle, destination, incoming, expected)
        self.assertEqual(bytes(transport.received), payload)
        self.assertEqual(sum(len(item) - 1 for item in transport.sent[2:]), len(payload) - offset)

    def test_streaming_prefix_mismatch_aborts_and_cleans_exact_stage(self) -> None:
        payload = b"payload" * 1000
        transport = StreamingTransport(
            payload, initial=b"wrong-prefix", corrupt_prefix=True
        )
        streamer = transfer.StreamingUploader(transport, chunk_bytes=1024)
        destination = "/workspace/projects/roborender/handoff/mismatch.bin"
        expected = identity(payload)
        incoming = transfer.resumable_incoming_path(destination, expected)
        with tempfile.TemporaryFile() as handle, mock.patch.object(
            transfer.secrets,
            "token_hex",
            return_value="0123456789abcdef0123456789abcdef",
        ):
            handle.write(payload)
            handle.flush()
            with self.assertRaisesRegex(transfer.TransferError, "prefix does not match"):
                streamer.upload(handle, destination, incoming, expected)
        self.assertTrue(transport.aborted)
        self.assertEqual(bytes(transport.received), b"")
        self.assertEqual(transport.sent[-1], b"\x00A")

    def test_stream_stage_identity_is_deterministic_and_destination_bound(self) -> None:
        expected = identity(b"same bytes")
        first = "/workspace/projects/roborender/a/value.bin"
        second = "/workspace/projects/roborender/a/other.bin"
        self.assertEqual(
            transfer.resumable_incoming_path(first, expected),
            transfer.resumable_incoming_path(first, expected),
        )
        self.assertNotEqual(
            transfer.resumable_incoming_path(first, expected),
            transfer.resumable_incoming_path(second, expected),
        )

    def test_stream_receiver_script_is_root_safe_raw_verified_and_atomic(self) -> None:
        destination = "/workspace/projects/roborender/handoff/value.bin"
        expected = identity(b"payload")
        incoming = transfer.resumable_incoming_path(destination, expected)
        script = transfer._stream_remote_script(
            destination,
            incoming,
            expected,
            0o644,
            "0123456789abcdef0123456789abcdef",
        )
        for required in (
            'realpath -e -- "$__root"',
            'stty raw -echo',
            'head -c "$__remaining"',
            'sha256sum -- "$__incoming"',
            'ln -- "$__incoming" "$__dst"',
            'rm -f -- "$__incoming"',
            "READY:%s:%s:%s",
            "DONE:%s:%s:%s",
        ):
            self.assertIn(required, script)
        self.assertNotIn("rm -rf", script)
        self.assertNotIn("mv ", script)

    def test_streaming_download_is_fixed_count_binary_safe_and_atomic(self) -> None:
        embedded_control = (
            b"\x1eLXFD1:0123456789abcdef0123456789abcdef:DONE:0:1:"
            + b"0" * 64
            + b"\x1f"
        )
        payload = bytes(range(256)) * 41 + embedded_control + os.urandom(2048)
        store = FakeStore()
        source = "/workspace/projects/roborender/results/candidate.pt"
        store.files[source] = payload
        transport = DownloadingTransport(payload)
        downloader = transfer.StreamingDownloader(transport)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            transfer.secrets,
            "token_hex",
            side_effect=[
                "11111111111111111111111111111111",
                "0123456789abcdef0123456789abcdef",
            ],
        ):
            destination = Path(temporary) / "candidate.pt"
            report = transfer.TransferEngine(store).download_streaming(
                source, destination, downloader
            )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(report["status"], "published")
            self.assertEqual(report["sha256"], identity(payload).sha256)
            self.assertEqual(list(Path(temporary).glob(".incoming.*")), [])
        self.assertEqual(transport.sent[1], b"\x00G")

    def test_streaming_download_ready_mutation_aborts_before_payload(self) -> None:
        original = b"original"
        changed = b"changed-after-inspection"
        transport = DownloadingTransport(changed, ready_identity=identity(changed))
        downloader = transfer.StreamingDownloader(transport)
        with tempfile.TemporaryFile() as handle, mock.patch.object(
            transfer.secrets,
            "token_hex",
            return_value="0123456789abcdef0123456789abcdef",
        ):
            with self.assertRaisesRegex(transfer.TransferError, "changed after inspection"):
                downloader.download(
                    "/workspace/projects/roborender/results/candidate.pt",
                    handle,
                    identity(original),
                )
        self.assertTrue(transport.aborted)
        self.assertEqual(transport.sent[-1], b"\x00A")

    def test_streaming_download_reinspection_detects_mutation_and_cleans_incoming(self) -> None:
        class MutatingStore(FakeStore):
            def __init__(self) -> None:
                super().__init__()
                self.inspections = 0

            def inspect(self, path: str) -> transfer.FileIdentity | None:
                self.inspections += 1
                value = super().inspect(path)
                if self.inspections == 2:
                    self.files[path] += b"mutated"
                    return super().inspect(path)
                return value

        payload = b"checkpoint" * 200
        store = MutatingStore()
        source = "/workspace/projects/roborender/results/candidate.pt"
        store.files[source] = payload
        downloader = transfer.StreamingDownloader(DownloadingTransport(payload))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "candidate.pt"
            with self.assertRaisesRegex(transfer.TransferError, "changed"):
                transfer.TransferEngine(store).download_streaming(
                    source, destination, downloader
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_download_sender_script_is_confined_fixed_count_and_reverified(self) -> None:
        script = transfer._download_remote_script(
            "/workspace/projects/roborender/results/candidate.pt",
            "0123456789abcdef0123456789abcdef",
        )
        for required in (
            'realpath -e -- "$__root"',
            "stty raw -echo",
            'sha256sum -- "$__src"',
            'head -c "$__size" -- "$__src"',
            'test "$__after_size" = "$__size"',
            'test "$__after_sha" = "$__sha"',
            "READY:%s:%s",
            "DONE:%s:%s:%s",
        ):
            self.assertIn(required, script)
        self.assertNotIn("rm ", script)
        self.assertNotIn("mv ", script)

    def test_cli_keeps_explicit_legacy_chunked_fallback(self) -> None:
        arguments = transfer._parser().parse_args(
            [
                "upload",
                "--local",
                "/tmp/value",
                "--remote",
                "/workspace/projects/roborender/value",
                "--legacy-chunked-upload",
            ]
        )
        self.assertTrue(arguments.legacy_chunked_upload)
        download = transfer._parser().parse_args(
            [
                "download",
                "--local",
                "/tmp/value",
                "--remote",
                "/workspace/projects/roborender/value",
                "--legacy-chunked-download",
            ]
        )
        self.assertTrue(download.legacy_chunked_download)

    def test_upload_existing_requires_explicit_verified_adoption(self) -> None:
        store = FakeStore()
        destination = "/workspace/projects/roborender/handoff/bundle.bin"
        store.files[destination] = b"same"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bundle.bin"
            source.write_bytes(b"same")
            engine = transfer.TransferEngine(store)
            with self.assertRaisesRegex(transfer.TransferError, "already exists"):
                engine.upload(source, destination)
            report = engine.upload(source, destination, adopt_existing=True)
            self.assertEqual(report["status"], "adopted")
            source.write_bytes(b"different")
            with self.assertRaisesRegex(transfer.TransferError, "not adopted"):
                engine.upload(source, destination, adopt_existing=True)

    def test_upload_failure_cleans_only_exact_incoming_file(self) -> None:
        class BrokenStore(FakeStore):
            def append(self, path: str, payload: bytes) -> None:
                raise transfer.TransferError("injected failure")

        store = BrokenStore()
        destination = "/workspace/projects/roborender/handoff/bundle.bin"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bundle.bin"
            source.write_bytes(b"payload")
            with self.assertRaisesRegex(transfer.TransferError, "injected"):
                transfer.TransferEngine(store).upload(source, destination)
        self.assertEqual(len(store.cleaned), 1)
        self.assertEqual(store.cleaned, store.created)
        self.assertTrue(Path(store.cleaned[0]).name.startswith(".incoming."))

    def test_upload_rejects_a_symlink_source(self) -> None:
        store = FakeStore()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.write_bytes(b"payload")
            link = root / "link"
            link.symlink_to(real)
            with self.assertRaisesRegex(transfer.TransferError, "unsafe"):
                transfer.TransferEngine(store).upload(
                    link, "/workspace/projects/roborender/handoff/value"
                )

    def test_download_is_verified_and_never_replaces_destination(self) -> None:
        store = FakeStore()
        source = "/workspace/projects/roborender/results/result.bin"
        payload = b"download" * 700
        store.files[source] = payload
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result.bin"
            report = transfer.TransferEngine(store, chunk_bytes=1024).download(
                source, destination
            )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(report["status"], "published")
            with self.assertRaisesRegex(transfer.TransferError, "already exists"):
                transfer.TransferEngine(store).download(source, destination)
            adopted = transfer.TransferEngine(store).download(
                source, destination, adopt_existing=True
            )
            self.assertEqual(adopted["status"], "adopted")

    def test_download_detects_remote_mutation_and_removes_local_incoming(self) -> None:
        class MutatingStore(FakeStore):
            def __init__(self) -> None:
                super().__init__()
                self.inspections = 0

            def inspect(self, path: str) -> transfer.FileIdentity | None:
                self.inspections += 1
                if self.inspections == 2:
                    self.files[path] += b"changed"
                return super().inspect(path)

        store = MutatingStore()
        source = "/workspace/projects/roborender/results/result.bin"
        store.files[source] = b"original"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result.bin"
            with self.assertRaisesRegex(transfer.TransferError, "changed"):
                transfer.TransferEngine(store).download(source, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_shell_publication_uses_exact_hardlink_and_cleanup_has_no_wildcard(self) -> None:
        shell = CaptureShell()
        store = transfer.ShellRemoteFileStore(shell)  # type: ignore[arg-type]
        incoming = "/workspace/projects/roborender/handoff/.incoming.deadbeef"
        destination = "/workspace/projects/roborender/handoff/value.bin"
        store.publish(incoming, destination, identity(b"payload"), 0o644)
        self.assertIn("ln --", shell.scripts[0])
        self.assertNotIn("mv ", shell.scripts[0])
        store.cleanup_incoming(incoming)
        self.assertIn("rm -f --", shell.scripts[1])
        cleanup = shell.scripts[1].split("rm -f --", 1)[1]
        self.assertNotIn("*", cleanup)
        self.assertIn('"$__path"', cleanup)

    def test_connection_repr_and_cli_do_not_expose_authentication_values(self) -> None:
        connection = transfer.ConnectionInfo(
            "wss://example.invalid/api/deployments/pod/replicas/r/shell",
            "Bearer extremely-secret",
            "https://example.invalid",
        )
        self.assertNotIn("extremely-secret", repr(connection))
        option_destinations = {
            action.dest
            for action in transfer._parser()._actions
        }
        self.assertTrue(option_destinations.isdisjoint({"token", "cookie", "authorization"}))

    def test_sdk_discovery_never_places_secret_in_subprocess_arguments(self) -> None:
        document = {
            "url": "wss://example.invalid/api/deployments/p/replicas/r/shell",
            "authorization": "Bearer secret-value",
            "origin": "https://example.invalid",
        }
        completed = subprocess_result = mock.Mock(
            returncode=0, stdout=__import__("json").dumps(document).encode("utf-8")
        )
        with mock.patch.object(transfer.subprocess, "run", return_value=completed) as run:
            connection = transfer.discover_connection("pod")
        arguments = run.call_args.args[0]
        self.assertNotIn("secret-value", " ".join(arguments))
        self.assertEqual(connection.authorization, "Bearer secret-value")
        self.assertIs(completed, subprocess_result)

    def test_explicit_sdk_interpreter_is_fail_closed_and_used_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            interpreter = Path(temporary) / "lepton-python"
            interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            interpreter.chmod(0o755)
            document = {
                "url": "wss://example.invalid/api/deployments/p/replicas/r/shell",
                "authorization": "Bearer secret-value",
                "origin": "https://example.invalid",
            }
            completed = mock.Mock(
                returncode=0,
                stdout=__import__("json").dumps(document).encode("utf-8"),
            )
            with mock.patch.object(transfer.subprocess, "run", return_value=completed) as run:
                transfer.discover_connection("pod", sdk_python=interpreter)
            self.assertEqual(run.call_args.args[0][0], str(interpreter))
            with self.assertRaisesRegex(transfer.TransferError, "missing"):
                transfer.discover_connection(
                    "pod", sdk_python=Path(temporary) / "does-not-exist"
                )


if __name__ == "__main__":
    unittest.main()
