from __future__ import annotations

import contextlib
import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.scaling import wandb_training_readback as readback


class SummarySubDict:
    """Match the public item-access behavior of W&B SummarySubDict."""

    def __init__(self, payload):
        self.payload = payload

    def __getitem__(self, key):
        return self.payload[key]


class RemoteFile:
    def __init__(self, *, name, content, advertised_content=None):
        self.name = name
        self.content = content
        advertised = content if advertised_content is None else advertised_content
        self.size = len(advertised)
        self.md5 = base64.b64encode(
            hashlib.md5(advertised, usedforsecurity=False).digest()
        ).decode("ascii")

    def download(self, *, root, replace):
        self.replace = replace
        path = Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)
        return path.open("rb")


class RemoteRun:
    def __init__(self, *, history=(), summary=None, files=()):
        self.history_rows = list(history)
        self.summary = summary if summary is not None else {}
        self.remote_files = list(files)
        self.history_calls = []
        self.history_options = []
        self.sampled_history_calls = []

    def scan_history(self, *, page_size, keys=None, use_cache=True):
        self.history_calls.append(None if keys is None else tuple(keys))
        self.history_options.append(
            {
                "page_size": page_size,
                "keys": None if keys is None else tuple(keys),
                "use_cache": use_cache,
            }
        )
        if keys is None:
            return iter([dict(row) for row in self.history_rows])
        return iter(
            [
                {key: row[key] for key in keys if key in row}
                for row in self.history_rows
                if any(key in row for key in keys)
            ]
        )

    def history(self, *, samples, keys, pandas):
        self.sampled_history_calls.append(
            {"samples": samples, "keys": tuple(keys), "pandas": pandas}
        )
        self.media_history_call = {
            "samples": samples,
            "keys": tuple(keys),
            "pandas": pandas,
        }
        return [
            {key: row[key] for key in keys if key in row}
            for row in self.history_rows[:samples]
            if any(key in row for key in keys)
        ]

    def files(self):
        return list(self.remote_files)


class FakeApi:
    def __init__(self, trainer, sidecar):
        self.paths = []
        self.trainer = trainer
        self.sidecar = sidecar

    def run(self, path):
        self.paths.append(path)
        return self.trainer if path.endswith("/trainer-run") else self.sidecar


class FakeWandb:
    def __init__(self, trainer, sidecar):
        self.api = FakeApi(trainer, sidecar)

    def Api(self, timeout):
        self.timeout = timeout
        return self.api


class WandbTrainingReadbackTest(unittest.TestCase):
    IMAGE_CONTENT = b"image-content"
    VIDEO_CONTENT = b"video-content"

    @staticmethod
    def sha256(content):
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def md5(content):
        return base64.b64encode(
            hashlib.md5(content, usedforsecurity=False).digest()
        ).decode("ascii")

    def remote_media_run(self, *, omit=(), corrupt_metadata=(), corrupt_download=()):
        image_path = "media/images/media/image/conditions_0_a.png"
        video_path = "media/videos/media/video/rollout_1_b.mp4"
        summary = {
            "media/image/conditions": SummarySubDict(
                {
                    "_type": "image-file",
                    "path": image_path,
                    "size": len(self.IMAGE_CONTENT),
                    "sha256": self.sha256(self.IMAGE_CONTENT),
                }
            ),
            "media/video/rollout": SummarySubDict(
                {
                    "_type": "video-file",
                    "path": video_path,
                    "size": len(self.VIDEO_CONTENT),
                    "sha256": self.sha256(self.VIDEO_CONTENT),
                }
            ),
        }
        files = [
            RemoteFile(
                name=image_path,
                content=(b"bad-image-data" if "image" in corrupt_download else self.IMAGE_CONTENT),
                advertised_content=self.IMAGE_CONTENT,
            ),
            RemoteFile(
                name=video_path,
                content=(b"bad-video-data" if "video" in corrupt_download else self.VIDEO_CONTENT),
                advertised_content=self.VIDEO_CONTENT,
            ),
        ]
        files = [
            item
            for item in files
            if ("image" if "/images/" in item.name else "video") not in omit
        ]
        for item in files:
            kind = "image" if "/images/" in item.name else "video"
            if kind in corrupt_metadata:
                item.md5 = "not-the-local-md5"
        history = [
            {
                "_step": 1,
                "trainer_step": 0,
                "media/image/conditions": summary["media/image/conditions"],
            },
            {
                "_step": 2,
                "trainer_step": 1,
                "media/video/rollout": summary["media/video/rollout"],
            },
        ]
        return RemoteRun(history=history, summary=summary, files=files)

    def local_media(self):
        return (
            readback.LocalMediaEvidence(
                key="media/image/conditions",
                kind="image",
                step=0,
                relative_path="telemetry/images/step_000000_conditions.png",
                size=len(self.IMAGE_CONTENT),
                md5=self.md5(self.IMAGE_CONTENT),
                sha256=self.sha256(self.IMAGE_CONTENT),
            ),
            readback.LocalMediaEvidence(
                key="media/video/rollout",
                kind="video",
                step=1,
                relative_path="telemetry/videos/step_000001_rollout.mp4",
                size=len(self.VIDEO_CONTENT),
                md5=self.md5(self.VIDEO_CONTENT),
                sha256=self.sha256(self.VIDEO_CONTENT),
            ),
        )

    def test_authorization_scope_is_required_and_team_only(self):
        required = [
            "--run-dir",
            "/tmp/run",
            "--project",
            "project",
            "--entity",
            "nvidia",
            "--expected-email",
            "ldu@nvidia.com",
            "--output",
            "/tmp/report.json",
            "--rollout-steps",
            "300",
            "--resume-training",
            "0",
            "--resume-checkpoint-step",
            "0",
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                readback._parser().parse_args(required)
            with self.assertRaises(SystemExit):
                readback._parser().parse_args(
                    required + ["--authorization-scope", "personal"]
                )

        args = readback._parser().parse_args(
            required + ["--authorization-scope", "team"]
        )
        self.assertEqual(args.authorization_scope, "team")

    def test_main_queries_and_emits_canonical_team_identity(self):
        identity = SimpleNamespace(
            email="ldu@nvidia.com",
            entity="nvidia",
            authorization_scope="team",
            team_member=True,
            entity_is_team=True,
            personal=False,
        )
        trainer_remote = RemoteRun(
            history=[{"_step": 0, "production/dagger_bc_loss": 0.25}]
        )
        sidecar_remote = self.remote_media_run()
        sidecar_remote.history_rows.extend(
            [
            {"trainer_step": 0, "scalar/loss": 0.5},
            # Sparse media rows must not affect grouped scalar verification.
            {"trainer_step": 0, "media/image/conditions": {"_type": "image-file"}},
            ]
        )
        fake_wandb = FakeWandb(trainer_remote, sidecar_remote)
        query_calls = []

        def query_identity(
            wandb_module,
            *,
            expected_email,
            requested_entity,
            authorization_scope,
        ):
            query_calls.append(
                {
                    "wandb_module": wandb_module,
                    "expected_email": expected_email,
                    "requested_entity": requested_entity,
                    "authorization_scope": authorization_scope,
                }
            )
            return identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "validation" / "wandb_training_readback.json"
            argv = [
                "wandb_training_readback.py",
                "--run-dir",
                str(root),
                "--project",
                "training-project",
                "--entity",
                "nvidia",
                "--expected-email",
                "ldu@nvidia.com",
                "--authorization-scope",
                "team",
                "--output",
                str(output),
                "--rollout-steps",
                "300",
                "--resume-training",
                "0",
                "--resume-checkpoint-step",
                "0",
                "--timeout-s",
                "1",
            ]
            with mock.patch.object(
                readback, "_load_wandb", return_value=fake_wandb
            ), mock.patch.object(
                readback, "query_identity", side_effect=query_identity
            ), mock.patch.object(
                readback, "_trainer_run_id", return_value="trainer-run"
            ), mock.patch.object(
                readback, "_run_id", return_value="sidecar-run"
            ), mock.patch.object(
                readback,
                "_expected_sidecar_evidence",
                return_value=(
                    (
                        readback.ScalarPoint(
                            key="trainer_step",
                            step=0,
                            value=0,
                            relative_path="telemetry/scalars/step_000000000.json",
                        ),
                        readback.ScalarPoint(
                            key="scalar/loss",
                            step=0,
                            value=0.5,
                            relative_path="telemetry/scalars/step_000000000.json",
                        ),
                    ),
                    self.local_media(),
                    {"scalar": 1, "image": 1, "video": 1},
                ),
            ), mock.patch.object(
                readback,
                "_expected_trainer_evidence",
                return_value=(
                    readback.ScalarPoint(
                        key="production/dagger_bc_loss",
                        step=0,
                        value=0.25,
                        relative_path=(
                            "validation/dagger_bc_loss_step_000000001.json"
                        ),
                    ),
                ),
            ), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                code = readback.main()

            self.assertEqual(code, 0)
            self.assertEqual(
                query_calls,
                [
                    {
                        "wandb_module": fake_wandb,
                        "expected_email": "ldu@nvidia.com",
                        "requested_entity": "nvidia",
                        "authorization_scope": "team",
                    }
                ],
            )
            self.assertEqual(
                fake_wandb.api.paths,
                [
                    "nvidia/training-project/trainer-run",
                    "nvidia/training-project/sidecar-run",
                ],
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            report["schema"], "fast_roborender.wandb_training_readback.v3"
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["account_email"], "ldu@nvidia.com")
        self.assertEqual(report["entity"], "nvidia")
        self.assertEqual(report["authorization_scope"], "team")
        self.assertTrue(report["team_member"])
        self.assertTrue(report["entity_is_team"])
        self.assertFalse(report["personal"])
        self.assertEqual(
            sidecar_remote.history_calls,
            [None, None],
        )
        self.assertTrue(
            all(
                call == {"page_size": 500, "keys": None, "use_cache": False}
                for call in sidecar_remote.history_options
            )
        )
        self.assertEqual(sidecar_remote.sampled_history_calls, [])
        self.assertEqual(
            report["observed_sidecar_media_keys"],
            ["media/image/conditions", "media/video/rollout"],
        )
        self.assertEqual(len(report["verified_remote_media_events"]), 2)

    def test_remote_media_requires_structured_summary_and_exact_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            observed, files = readback._verify_remote_media(
                self.remote_media_run(),
                self.local_media(),
                download_root=Path(temporary) / "downloads",
            )

        self.assertEqual(
            observed, {"media/image/conditions", "media/video/rollout"}
        )
        self.assertEqual(len(files), 2)
        self.assertEqual(
            {item["remote_sha256"] for item in files},
            {self.sha256(self.IMAGE_CONTENT), self.sha256(self.VIDEO_CONTENT)},
        )

    def test_remote_media_normalizes_integral_float_history_size(self):
        remote = self.remote_media_run()
        remote.history_rows[0]["media/image/conditions"].payload["size"] = float(
            len(self.IMAGE_CONTENT)
        )

        with tempfile.TemporaryDirectory() as temporary:
            _, records = readback._verify_remote_media(
                remote,
                self.local_media(),
                download_root=Path(temporary) / "downloads",
            )

        image = next(record for record in records if record["kind"] == "image")
        self.assertEqual(image["size"], len(self.IMAGE_CONTENT))
        self.assertIs(type(image["size"]), int)

    def test_remote_media_rejects_invalid_numeric_history_sizes(self):
        invalid_sizes = (
            True,
            1.5,
            float("nan"),
            float("inf"),
            0,
            -1,
            readback.MAX_REMOTE_MEDIA_BYTES["image"] + 1,
        )
        for invalid_size in invalid_sizes:
            with self.subTest(size=invalid_size):
                remote = self.remote_media_run()
                remote.history_rows[0]["media/image/conditions"].payload[
                    "size"
                ] = invalid_size
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaisesRegex(
                        readback.ReadbackError,
                        "distinct exact W&B history event",
                    ):
                        readback._verify_remote_media(
                            remote,
                            self.local_media(),
                            download_root=Path(temporary) / "downloads",
                        )

    def test_remote_media_fails_closed_for_missing_or_mismatched_file(self):
        for label, remote in (
            ("missing image", self.remote_media_run(omit={"image"})),
            ("missing video", self.remote_media_run(omit={"video"})),
            (
                "image metadata mismatch",
                self.remote_media_run(corrupt_metadata={"image"}),
            ),
            (
                "video metadata mismatch",
                self.remote_media_run(corrupt_metadata={"video"}),
            ),
            (
                "image download mismatch",
                self.remote_media_run(corrupt_download={"image"}),
            ),
            (
                "video download mismatch",
                self.remote_media_run(corrupt_download={"video"}),
            ),
        ):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(readback.ReadbackError):
                        readback._verify_remote_media(
                            remote,
                            self.local_media(),
                            download_root=Path(temporary) / "downloads",
                        )

    def test_remote_media_fails_closed_for_unstructured_summary_value(self):
        remote = self.remote_media_run()
        remote.summary["media/image/conditions"] = {
            "_type": "image-file",
            "path": "media/images/media/image/conditions_0_a.png",
            # Missing byte size and SHA256 is not sufficient evidence.
        }

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(readback.ReadbackError):
                readback._verify_remote_media(
                    remote,
                    self.local_media(),
                    download_root=Path(temporary) / "downloads",
                )

    def test_remote_media_rejects_one_remote_file_for_duplicate_local_content(self):
        duplicate = readback.LocalMediaEvidence(
            key="media/image/conditions",
            kind="image",
            step=10,
            relative_path="telemetry/images/step_000000010_conditions.png",
            size=len(self.IMAGE_CONTENT),
            md5=self.md5(self.IMAGE_CONTENT),
            sha256=self.sha256(self.IMAGE_CONTENT),
        )
        expected = (self.local_media()[0], duplicate, self.local_media()[1])

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                readback.ReadbackError, "distinct exact W&B history event"
            ):
                readback._verify_remote_media(
                    self.remote_media_run(),
                    expected,
                    download_root=Path(temporary) / "downloads",
                )

    def test_duplicate_content_passes_with_distinct_history_events_and_one_blob(self):
        duplicate = readback.LocalMediaEvidence(
            key="media/image/conditions",
            kind="image",
            step=10,
            relative_path="telemetry/images/step_000000010_conditions.png",
            size=len(self.IMAGE_CONTENT),
            md5=self.md5(self.IMAGE_CONTENT),
            sha256=self.sha256(self.IMAGE_CONTENT),
        )
        expected = (self.local_media()[0], duplicate, self.local_media()[1])
        remote = self.remote_media_run()
        remote.history_rows.append(
            {
                "_step": 3,
                "trainer_step": 10,
                "media/image/conditions": remote.summary[
                    "media/image/conditions"
                ],
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            observed, records = readback._verify_remote_media(
                remote,
                expected,
                download_root=Path(temporary) / "downloads",
            )

        self.assertEqual(
            observed, {"media/image/conditions", "media/video/rollout"}
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(len({item["remote_history_step"] for item in records}), 3)
        self.assertEqual(len({item["remote_path"] for item in records}), 2)
        image_records = [item for item in records if item["kind"] == "image"]
        self.assertEqual(sum(item["summary_current"] for item in image_records), 1)

    def test_remote_media_uses_complete_uncached_scan_beyond_500_rows(self):
        remote = self.remote_media_run()
        exact_rows = [dict(row) for row in remote.history_rows]
        exact_rows[0]["_step"] = 600
        exact_rows[1]["_step"] = 601
        remote.history_rows = [
            {
                "_step": index,
                "trainer_step": 1000 + index,
                (
                    "media/image/conditions"
                    if index % 2 == 0
                    else "media/video/rollout"
                ): remote.summary[
                    "media/image/conditions"
                    if index % 2 == 0
                    else "media/video/rollout"
                ],
            }
            for index in range(600)
        ] + exact_rows

        with tempfile.TemporaryDirectory() as temporary:
            observed, records = readback._verify_remote_media(
                remote,
                self.local_media(),
                download_root=Path(temporary) / "downloads",
            )

        self.assertEqual(
            observed, {"media/image/conditions", "media/video/rollout"}
        )
        self.assertEqual(
            {record["remote_history_step"] for record in records}, {600, 601}
        )
        self.assertEqual(
            remote.history_options,
            [{"page_size": 500, "keys": None, "use_cache": False}],
        )
        self.assertEqual(remote.sampled_history_calls, [])

    def test_remote_media_rejects_stale_summary_and_reversed_event_order(self):
        key = "media/image/conditions"
        first_path = "media/images/media/image/conditions_0_a.png"
        second_path = "media/images/media/image/conditions_10_b.png"
        second_content = b"newer-image-content"
        first_metadata = SummarySubDict(
            {
                "_type": "image-file",
                "path": first_path,
                "size": len(self.IMAGE_CONTENT),
                "sha256": self.sha256(self.IMAGE_CONTENT),
            }
        )
        second_metadata = SummarySubDict(
            {
                "_type": "image-file",
                "path": second_path,
                "size": len(second_content),
                "sha256": self.sha256(second_content),
            }
        )
        expected = (
            self.local_media()[0],
            readback.LocalMediaEvidence(
                key=key,
                kind="image",
                step=10,
                relative_path="telemetry/images/step_000000010_conditions.png",
                size=len(second_content),
                md5=self.md5(second_content),
                sha256=self.sha256(second_content),
            ),
        )
        files = (
            RemoteFile(name=first_path, content=self.IMAGE_CONTENT),
            RemoteFile(name=second_path, content=second_content),
        )
        rows = [
            {"_step": 1, "trainer_step": 0, key: first_metadata},
            {"_step": 2, "trainer_step": 10, key: second_metadata},
        ]
        stale = RemoteRun(
            history=rows,
            summary={key: first_metadata},
            files=files,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(readback.ReadbackError, "newest exact event"):
                readback._verify_remote_media(
                    stale,
                    expected,
                    download_root=Path(temporary) / "stale",
                )

        reversed_order = RemoteRun(
            history=[
                {"_step": 2, "trainer_step": 0, key: first_metadata},
                {"_step": 1, "trainer_step": 10, key: second_metadata},
            ],
            summary={key: first_metadata},
            files=files,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(readback.ReadbackError, "local order"):
                readback._verify_remote_media(
                    reversed_order,
                    expected,
                    download_root=Path(temporary) / "reversed",
                )

    def test_failed_media_attempt_removes_partial_downloads_immediately(self):
        attempt_directories = []

        def fail_with_partial_download(_remote, _expected, *, download_root):
            download_root.mkdir(parents=True)
            (download_root / "partial").write_bytes(b"partial")
            attempt_directories.append(download_root.parent)
            raise readback.ReadbackError("injected failure")

        with tempfile.TemporaryDirectory() as temporary:
            validation_dir = Path(temporary) / "validation"
            validation_dir.mkdir()
            with mock.patch.object(
                readback,
                "_verify_remote_media",
                side_effect=fail_with_partial_download,
            ):
                with self.assertRaisesRegex(readback.ReadbackError, "injected"):
                    readback._verify_remote_media_attempt(
                        object(),
                        (),
                        validation_dir=validation_dir,
                        attempt=7,
                    )
            self.assertEqual(len(attempt_directories), 1)
            self.assertFalse(attempt_directories[0].exists())

    def test_unprojected_scalar_readback_binds_steps_values_and_rejects_stale_rows(self):
        expected = (
            readback.ScalarPoint("trainer_step", 0, 0, "step0.json"),
            readback.ScalarPoint("loss", 0, 0.5, "step0.json"),
            readback.ScalarPoint("trainer_step", 10, 10, "step10.json"),
            readback.ScalarPoint("loss", 10, 0.25, "step10.json"),
        )
        remote = RemoteRun(
            history=[
                {"trainer_step": 0, "loss": 0.5},
                {"trainer_step": 10, "loss": 0.25},
                {"trainer_step": 20, "loss": 999.0},
            ]
        )

        verified = readback._scan_scalar_points(
            remote, expected, axis="trainer_step", page_size=500
        )

        self.assertEqual(verified, expected)
        self.assertEqual(remote.history_calls, [None])
        self.assertFalse(remote.history_options[0]["use_cache"])

        for bad_value in (None, float("nan"), float("inf")):
            with self.subTest(value=bad_value):
                stale = RemoteRun(
                    history=[
                        {"trainer_step": 0, "loss": 0.5},
                        {"trainer_step": 10, "loss": bad_value},
                    ]
                )
                verified = readback._scan_scalar_points(
                    stale, expected, axis="trainer_step", page_size=500
                )
                self.assertLess(len(verified), len(expected))

    def test_scalar_record_cannot_be_satisfied_across_split_history_rows(self):
        expected = (
            readback.ScalarPoint("trainer_step", 10, 10, "step10.json"),
            readback.ScalarPoint("loss", 10, 0.25, "step10.json"),
            readback.ScalarPoint("accuracy", 10, 0.75, "step10.json"),
        )
        remote = RemoteRun(
            history=[
                {"trainer_step": 10, "loss": 0.25, "accuracy": 0.0},
                {"trainer_step": 10, "loss": 99.0, "accuracy": 0.75},
            ]
        )

        verified = readback._scan_scalar_points(
            remote, expected, axis="trainer_step", page_size=500
        )

        self.assertEqual(verified, ())
        self.assertEqual(remote.history_calls, [None])

    def test_unprojected_scan_avoids_wandb_explicit_key_projection_loss(self):
        keys = tuple(f"metric/{index:02d}" for index in range(20))
        expected = tuple(
            readback.ScalarPoint(key, step, step + index / 100.0, f"{step}.json")
            for step in range(30)
            for index, key in enumerate(keys)
        )

        class ProjectionDefectiveRun(RemoteRun):
            def scan_history(
                self, *, page_size, keys=None, use_cache=True
            ):
                if keys is not None:
                    rows = super().scan_history(
                        page_size=page_size,
                        keys=keys,
                        use_cache=use_cache,
                    )
                    missing = set(keys[-4:])
                    return iter(
                        [
                            {
                                key: (None if key in missing else value)
                                for key, value in row.items()
                            }
                            for row in rows
                        ]
                    )
                return super().scan_history(
                    page_size=page_size,
                    use_cache=use_cache,
                )

        remote = ProjectionDefectiveRun(
            history=[
                {
                    "trainer_step": step,
                    **{
                        key: step + index / 100.0
                        for index, key in enumerate(keys)
                    },
                }
                for step in range(30)
            ]
        )

        verified = readback._scan_scalar_points(
            remote, expected, axis="trainer_step", page_size=500
        )

        self.assertEqual(verified, expected)
        self.assertEqual(len(verified), 600)
        self.assertEqual(remote.history_calls, [None])
        self.assertFalse(remote.history_options[0]["use_cache"])

    def test_trainer_marker_maps_update_one_to_wandb_step_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            marker = run_dir / "validation/dagger_bc_loss_step_000000001.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.dagger_bc_loss.v1",
                        "update": 1,
                        "world_size": 1,
                        "loss": 0.25,
                        "finite_verified": True,
                        "nonnegative_verified": True,
                    }
                ),
                encoding="utf-8",
            )

            points = readback._expected_trainer_evidence(run_dir)

        self.assertEqual(points[0].step, 0)
        remote = RemoteRun(
            history=[{"_step": 0, "production/dagger_bc_loss": 0.25}]
        )
        self.assertEqual(
            readback._scan_scalar_points(
                remote, points, axis="_step", page_size=500
            ),
            points,
        )

    def test_scalar_report_requires_both_sides_of_resume_boundary(self):
        trainer = tuple(
            readback.ScalarPoint(
                "production/dagger_bc_loss",
                step,
                1.0 / (step + 1),
                f"loss_{step}.json",
            )
            for step in range(6)
        )
        trainer_report = readback._scalar_evidence_report(
            trainer,
            trainer,
            axis="_step",
            resume_training=True,
            resume_checkpoint_step=5,
            rollout_steps=300,
        )
        sidecar = (
            readback.ScalarPoint("trainer_step", 1490, 1490, "step1490.json"),
            readback.ScalarPoint("trainer_step", 1500, 1500, "step1500.json"),
        )
        sidecar_report = readback._scalar_evidence_report(
            sidecar,
            sidecar,
            axis="trainer_step",
            resume_training=True,
            resume_checkpoint_step=5,
            rollout_steps=300,
        )

        self.assertEqual(
            (trainer_report["boundary_pre_step"], trainer_report["boundary_post_step"]),
            (4, 5),
        )
        self.assertEqual(
            (sidecar_report["boundary_pre_step"], sidecar_report["boundary_post_step"]),
            (1490, 1500),
        )
        missing_post = readback._scalar_evidence_report(
            sidecar,
            sidecar[:1],
            axis="trainer_step",
            resume_training=True,
            resume_checkpoint_step=5,
            rollout_steps=300,
        )
        self.assertFalse(missing_post["boundary_post_verified"])


if __name__ == "__main__":
    unittest.main()
