import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.scaling import wandb_run_watcher as watcher


class FakeRun:
    def __init__(self, failures_remaining=0):
        self.logs = []
        self.metrics = []
        self.finished = False
        self.failures_remaining = failures_remaining
        self.log_attempts = 0

    def log(self, payload):
        self.log_attempts += 1
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("transient upload failure")
        self.logs.append(payload)

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def finish(self, exit_code=None):
        self.finished = True
        self.exit_code = exit_code


class FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.runs = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = FakeRun()
        self.runs.append(run)
        return run

    @staticmethod
    def Image(path):
        return ("image", path)

    @staticmethod
    def Video(path, format):
        return ("video", path, format)


class WandbRunWatcherTest(unittest.TestCase):
    def test_authorization_scope_is_required_and_team_only(self):
        required = [
            "--run-dir",
            "/tmp/run",
            "--project",
            "project",
            "--expected-email",
            "ldu@nvidia.com",
            "--entity",
            "nvidia",
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                watcher._parser().parse_args(required)
            with self.assertRaises(SystemExit):
                watcher._parser().parse_args(
                    required + ["--authorization-scope", "personal"]
                )
        arguments = watcher._parser().parse_args(
            required + ["--authorization-scope", "team"]
        )
        self.assertEqual(arguments.authorization_scope, "team")

    def test_rank_resolution_and_invalid_rank(self):
        self.assertEqual(watcher.process_rank({}), 0)
        self.assertEqual(watcher.process_rank({"SLURM_PROCID": "7"}), 7)
        self.assertEqual(watcher.process_rank({"RANK": "3", "SLURM_PROCID": "7"}), 3)
        with self.assertRaises(watcher.ArtifactError):
            watcher.process_rank({"RANK": "not-an-integer"})

    def test_scalar_contract_filters_metadata_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "step": 12,
                        "metrics": {
                            "loss/dagger_bc_loss": 0.25,
                            "renderer/errors": 0,
                            "healthy": True,
                            "optional": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            step, metrics = watcher.load_scalar_file(path)
            self.assertEqual(step, 12)
            self.assertEqual(metrics["healthy"], 1)
            self.assertNotIn("optional", metrics)

            path.write_text(
                json.dumps({"step": 13, "metrics": {"bad": float("nan")}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(watcher.ArtifactError, "not finite numeric"):
                watcher.load_scalar_file(path)

    def test_discovers_only_stable_in_tree_media_with_step_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "telemetry" / "images"
            videos = root / "telemetry" / "videos"
            images.mkdir(parents=True)
            videos.mkdir(parents=True)
            image = images / "step_000009_generated_triplet.png"
            video = videos / "step-000010-rollout.mp4"
            ignored = images / "preview.png"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            ignored.write_bytes(b"image")

            now = max(image.stat().st_mtime, video.stat().st_mtime) + 10
            with self.assertRaisesRegex(watcher.ArtifactError, "no step"):
                watcher.discover_artifacts(root, min_age_s=0, now=now)
            ignored.unlink()
            found = watcher.discover_artifacts(root, min_age_s=5, now=now)
            self.assertEqual([(x.kind, x.step) for x in found], [("image", 9), ("video", 10)])
            self.assertEqual(found[0].key, "media/image/generated_triplet")
            self.assertEqual(found[1].key, "media/video/rollout")

            outside = root / "outside_step_000011.png"
            outside.write_bytes(b"outside")
            link = images / "step_000011_link.png"
            try:
                link.symlink_to(outside)
            except OSError:
                pass
            found = watcher.discover_artifacts(root, min_age_s=0, now=now)
            self.assertNotIn(11, [item.step for item in found])

    def test_oversize_media_is_an_error_not_a_silent_drop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "telemetry" / "images"
            images.mkdir(parents=True)
            image = images / "step_000001_generated.png"
            image.write_bytes(b"too-large")
            with self.assertRaisesRegex(watcher.ArtifactError, "exceeds"):
                watcher.discover_artifacts(
                    root, min_age_s=0, max_image_bytes=1, now=10**12
                )

    def test_uploads_scalar_image_and_video_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scalar_dir = root / "telemetry" / "scalars"
            image_dir = root / "telemetry" / "images"
            video_dir = root / "telemetry" / "videos"
            scalar_dir.mkdir(parents=True)
            image_dir.mkdir()
            video_dir.mkdir()
            scalar = scalar_dir / "step_000002.json"
            image = image_dir / "step_000002_triplet.png"
            video = video_dir / "step_000002_rollout.mp4"
            scalar.write_text(
                json.dumps({"step": 2, "metrics": {"loss/dagger_bc_loss": 0.4}}),
                encoding="utf-8",
            )
            image.write_bytes(b"image")
            video.write_bytes(b"video")

            fake_run = FakeRun()
            state_file = root / "telemetry" / ".state.json"
            upload = watcher.WandbArtifactWatcher(
                run_dir=root,
                wandb_module=FakeWandb(),
                wandb_run=fake_run,
                state_file=state_file,
                min_age_s=0,
            )
            self.assertEqual(upload.scan_once(now=10**12), 3)
            self.assertEqual(upload.scan_once(now=10**12), 0)
            self.assertEqual(len(fake_run.logs), 3)
            self.assertTrue(all(entry["trainer_step"] == 2 for entry in fake_run.logs))
            self.assertTrue(any("loss/dagger_bc_loss" in entry for entry in fake_run.logs))
            self.assertTrue(any("media/image/triplet" in entry for entry in fake_run.logs))
            self.assertTrue(any("media/video/rollout" in entry for entry in fake_run.logs))
            self.assertEqual(len(watcher.load_state(state_file)), 3)

            scalar.write_text(
                json.dumps({"step": 2, "metrics": {"loss/dagger_bc_loss": 0.3}}),
                encoding="utf-8",
            )
            os.utime(scalar, ns=(scalar.stat().st_atime_ns, scalar.stat().st_mtime_ns + 1))
            self.assertEqual(upload.scan_once(now=10**12), 1)
            self.assertEqual(fake_run.logs[-1]["loss/dagger_bc_loss"], 0.3)

    def test_transient_upload_failure_retries_without_blocking_other_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scalar_dir = root / "telemetry" / "scalars"
            scalar_dir.mkdir(parents=True)
            for step in (1, 2):
                (scalar_dir / f"step_{step:06d}.json").write_text(
                    json.dumps({"step": step, "metrics": {"loss": 1.0 / step}}),
                    encoding="utf-8",
                )

            fake_run = FakeRun(failures_remaining=1)
            state_file = root / "telemetry" / ".state.json"
            upload = watcher.WandbArtifactWatcher(
                run_dir=root,
                wandb_module=FakeWandb(),
                wandb_run=fake_run,
                state_file=state_file,
                min_age_s=0,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(upload.scan_once(now=10**12), 1)
                self.assertEqual(upload.last_scan_error_count, 1)
                self.assertEqual(upload.last_scan_pending_count, 1)
                self.assertEqual(upload.scan_once(now=10**12), 1)
                self.assertEqual(upload.last_scan_error_count, 0)
                self.assertEqual(upload.last_scan_pending_count, 0)

            self.assertEqual(fake_run.log_attempts, 3)
            self.assertEqual([entry["trainer_step"] for entry in fake_run.logs], [2, 1])
            self.assertEqual(len(watcher.load_state(state_file)), 2)
            self.assertIn("upload deferred", stderr.getvalue())
            self.assertNotIn("transient upload failure", stderr.getvalue())

    def test_bad_artifact_is_quarantined_while_valid_scalar_uploads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scalar_dir = root / "telemetry" / "scalars"
            image_dir = root / "telemetry" / "images"
            scalar_dir.mkdir(parents=True)
            image_dir.mkdir()
            (scalar_dir / "step_000007.json").write_text(
                json.dumps({"step": 7, "metrics": {"loss": 0.2}}),
                encoding="utf-8",
            )
            (image_dir / "missing_step_name.png").write_bytes(b"invalid")

            fake_run = FakeRun()
            upload = watcher.WandbArtifactWatcher(
                run_dir=root,
                wandb_module=FakeWandb(),
                wandb_run=fake_run,
                state_file=root / "telemetry" / ".state.json",
                min_age_s=0,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(upload.scan_once(now=10**12), 1)

            self.assertEqual(fake_run.logs[0]["trainer_step"], 7)
            self.assertEqual(upload.last_scan_error_count, 1)
            self.assertIn("invalid image artifact", stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())

    def test_ready_marker_is_atomic_and_contains_no_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "telemetry" / "READY.json"
            watcher.write_ready_marker(
                marker,
                account_email="ldu@nvidia.com",
                entity="nvidia",
                project="dagger",
                authorization_scope="team",
                team_member=True,
                entity_is_team=True,
                personal=False,
                run_id="stable_run_1",
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["account_email"], "ldu@nvidia.com")
            self.assertEqual(payload["entity"], "nvidia")
            self.assertEqual(payload["authorization_scope"], "team")
            self.assertTrue(payload["team_member"])
            self.assertTrue(payload["entity_is_team"])
            self.assertFalse(payload["personal"])
            self.assertEqual(payload["project"], "dagger")
            self.assertEqual(payload["wandb_run_id"], "stable_run_1")
            self.assertNotIn("key", json.dumps(payload).lower())

    def test_run_id_and_upload_state_are_durably_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id_file = root / "telemetry" / "WANDB_SIDECAR_RUN_ID"
            state_file = root / "telemetry" / ".wandb_upload_state.json"

            first = watcher.load_or_create_run_id(run_id_file)
            second = watcher.load_or_create_run_id(run_id_file)
            self.assertEqual(first, second)
            self.assertEqual(run_id_file.stat().st_mode & 0o777, 0o600)

            # Legacy fingerprints must not suppress uploads into a new run.
            watcher.save_state(state_file, {"telemetry/images/old.png": "1:2"})
            watcher.bind_state_to_run(state_file, first)
            self.assertEqual(
                watcher.load_state(state_file, expected_run_id=first), {}
            )
            payload = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["wandb_run_id"], first)
            self.assertEqual(payload["artifacts"], {})

            watcher.save_state(state_file, {"x": "y"}, run_id="other_run")
            with self.assertRaisesRegex(watcher.ArtifactError, "different sidecar run"):
                watcher.bind_state_to_run(state_file, first)

    def test_malformed_or_symlinked_run_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad-id"
            bad.write_text("contains spaces\n", encoding="utf-8")
            with self.assertRaisesRegex(watcher.ArtifactError, "malformed"):
                watcher.load_or_create_run_id(bad)

            target = root / "target"
            target.write_text("valid_id\n", encoding="utf-8")
            link = root / "link"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are not available")
            with self.assertRaisesRegex(watcher.ArtifactError, "symlink"):
                watcher.load_or_create_run_id(link)

    def test_main_resumes_the_same_sidecar_run_id_after_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_wandb = FakeWandb()
            identity = SimpleNamespace(
                entity="nvidia",
                email="ldu@nvidia.com",
                authorization_scope="team",
                team_member=True,
                entity_is_team=True,
                personal=False,
            )
            arguments = [
                "--run-dir",
                str(root),
                "--project",
                "dagger",
                "--expected-email",
                "ldu@nvidia.com",
                "--entity",
                "nvidia",
                "--authorization-scope",
                "team",
                "--min-age-s",
                "0",
                "--once",
            ]
            with mock.patch.dict(os.environ, {"RANK": "0"}, clear=False), \
                 mock.patch.object(watcher, "_load_wandb", return_value=fake_wandb), \
                 mock.patch.object(
                     watcher, "query_identity", return_value=identity
                 ) as query_identity:
                self.assertEqual(watcher.main(arguments), 0)
                self.assertEqual(watcher.main(arguments), 0)

            first, second = fake_wandb.init_calls
            self.assertEqual(query_identity.call_count, 2)
            self.assertEqual(
                query_identity.call_args.kwargs,
                {
                    "expected_email": "ldu@nvidia.com",
                    "requested_entity": "nvidia",
                    "authorization_scope": "team",
                },
            )
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(first["resume"], "allow")
            self.assertEqual(second["resume"], "must")
            self.assertEqual(
                (root / "telemetry" / "WANDB_SIDECAR_RUN_ID")
                .read_text(encoding="utf-8")
                .strip(),
                first["id"],
            )
            for call in (first, second):
                self.assertEqual(call["config"]["account_email"], "ldu@nvidia.com")
                self.assertEqual(call["config"]["entity"], "nvidia")
                self.assertEqual(call["config"]["authorization_scope"], "team")
                self.assertTrue(call["config"]["team_member"])
                self.assertTrue(call["config"]["entity_is_team"])
                self.assertFalse(call["config"]["personal"])

    def test_final_drain_waits_a_full_stability_window(self):
        deadline, complete = watcher.update_final_drain(
            marker_present=False,
            deadline=None,
            now=10.0,
            min_age_s=5.0,
        )
        self.assertIsNone(deadline)
        self.assertFalse(complete)

        deadline, complete = watcher.update_final_drain(
            marker_present=True,
            deadline=None,
            now=11.0,
            min_age_s=5.0,
        )
        self.assertEqual(deadline, 16.0)
        self.assertFalse(complete)

        deadline, complete = watcher.update_final_drain(
            marker_present=True,
            deadline=deadline,
            now=15.999,
            min_age_s=5.0,
        )
        self.assertFalse(complete)
        deadline, complete = watcher.update_final_drain(
            marker_present=False,
            deadline=deadline,
            now=16.0,
            min_age_s=5.0,
        )
        self.assertTrue(complete)

    def test_nonzero_rank_main_is_a_noop_before_loading_wandb(self):
        with mock.patch.dict(os.environ, {"RANK": "1"}, clear=False), mock.patch.object(
            watcher, "_load_wandb", side_effect=AssertionError("must not load W&B")
        ):
            code = watcher.main(
                [
                    "--run-dir",
                    "/unused",
                    "--project",
                    "unused",
                    "--expected-email",
                    "ldu@nvidia.com",
                    "--entity",
                    "nvidia",
                    "--authorization-scope",
                    "team",
                    "--once",
                ]
            )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
