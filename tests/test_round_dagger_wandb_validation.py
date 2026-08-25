"""Fail-closed tests for bounded actor W&B upload and sidecar-only readback."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.round_dagger import readback_actor_validation_wandb as readback
from scripts.scaling import wandb_run_watcher


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/round_dagger/run_actor_validation_with_wandb.sh"


def _digests(payload: bytes) -> tuple[str, str]:
    md5 = base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")
    return md5, hashlib.sha256(payload).hexdigest()


class FakeRemoteFile:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.payload = payload
        self.size = len(payload)
        self.md5, _ = _digests(payload)

    def download(self, *, root: str, replace: bool):
        destination = Path(root) / self.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload)
        return destination.open("rb")


class FakeRemoteRun:
    def __init__(self, rows, files, summary):
        self.rows = rows
        self._files = files
        self.summary = summary
        self.job_type = "dagger-bounded-validation-sidecar"
        self.config = {
            "telemetry_scope": "bounded-validation",
            "expected_account": "ldu@nvidia.com",
            "account_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "team_member": True,
            "entity_is_team": True,
            "personal": False,
        }

    def scan_history(self, **kwargs):
        return iter(self.rows)

    def files(self):
        return list(self._files)


class FakeApi:
    def __init__(self, remote, *, error_text: str | None = None):
        self.remote = remote
        self.error_text = error_text
        self.viewer = {
            "email": "ldu@nvidia.com",
            "username": "leo",
            "teams": ["nvidia"],
        }

    def run(self, path):
        if self.error_text is not None:
            raise RuntimeError(self.error_text)
        return self.remote


class FakeWandb:
    def __init__(self, remote, *, error_text: str | None = None):
        self.remote = remote
        self.error_text = error_text

    def Api(self, timeout):
        return FakeApi(self.remote, error_text=self.error_text)


class BoundedWandbReadbackTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        telemetry = self.root / "telemetry"
        for kind in ("scalars", "images", "videos"):
            (telemetry / kind).mkdir(parents=True, exist_ok=True)
        self.scalar = telemetry / "scalars" / "step_000000000.json"
        self.image = telemetry / "images" / "step_000000000_conditions_generated.png"
        self.video = telemetry / "videos" / "step_000000001_generated_triplet.mp4"
        self.scalar.write_text(
            json.dumps({"step": 0, "metrics": {"renderer/latency_ms": 12.5}}),
            encoding="utf-8",
        )
        self.image.write_bytes(b"exact-png-evidence")
        self.video.write_bytes(b"exact-mp4-evidence")
        self.run_id = "bounded_sidecar_1"
        (telemetry / "WANDB_SIDECAR_RUN_ID").write_text(
            self.run_id + "\n", encoding="utf-8"
        )
        wandb_run_watcher.write_initialized_marker(
            telemetry / "WANDB_SIDECAR_INITIALIZED.json",
            run_id=self.run_id,
            telemetry_scope="bounded-validation",
            project="bounded-project",
            entity="nvidia",
        )
        wandb_run_watcher.write_ready_marker(
            telemetry / "WANDB_WATCHER_READY.json",
            account_email="ldu@nvidia.com",
            entity="nvidia",
            project="bounded-project",
            authorization_scope="team",
            team_member=True,
            entity_is_team=True,
            personal=False,
            run_id=self.run_id,
            telemetry_scope="bounded-validation",
        )
        self.output = self.root / "validation" / "readback.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _remote(self):
        image_payload = self.image.read_bytes()
        video_payload = self.video.read_bytes()
        image_md5, image_sha = _digests(image_payload)
        video_md5, video_sha = _digests(video_payload)
        image_path = "media/images/conditions_generated.png"
        video_path = "media/videos/generated_triplet.mp4"
        image_meta = {
            "_type": "image-file",
            "path": image_path,
            "size": len(image_payload),
            "sha256": image_sha,
        }
        video_meta = {
            "_type": "video-file",
            "path": video_path,
            "size": len(video_payload),
            "sha256": video_sha,
        }
        files = [
            FakeRemoteFile(image_path, image_payload),
            FakeRemoteFile(video_path, video_payload),
        ]
        self.assertEqual(files[0].md5, image_md5)
        self.assertEqual(files[1].md5, video_md5)
        rows = [
            {"_step": 0, "trainer_step": 0, "renderer/latency_ms": 12.5},
            {
                "_step": 1,
                "trainer_step": 0,
                "media/image/conditions_generated": image_meta,
            },
            {
                "_step": 2,
                "trainer_step": 1,
                "media/video/generated_triplet": video_meta,
            },
        ]
        return FakeRemoteRun(
            rows,
            files,
            {
                "media/image/conditions_generated": image_meta,
                "media/video/generated_triplet": video_meta,
            },
        )

    def _arguments(self):
        return [
            "--run-dir",
            str(self.root),
            "--project",
            "bounded-project",
            "--entity",
            "nvidia",
            "--expected-email",
            "ldu@nvidia.com",
            "--authorization-scope",
            "team",
            "--output",
            str(self.output),
            "--timeout-s",
            "0.05",
            "--poll-interval-s",
            "0.001",
        ]

    def test_sidecar_only_readback_binds_scalar_png_and_mp4_exactly(self):
        fake = FakeWandb(self._remote())
        stdout = io.StringIO()
        with mock.patch.object(readback, "_load_wandb", return_value=fake), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(readback.main(self._arguments()), 0)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["trainer_run_required"])
        self.assertEqual(report["local_artifact_counts"], {
            "scalar": 1, "image": 1, "video": 1
        })
        self.assertEqual(report["verified_remote_media_event_count"], 2)
        self.assertTrue(report["sidecar_scalar_evidence"]["all_points_verified"])
        self.assertNotIn("trainer_run_id", report)

    def test_wrong_remote_scope_and_missing_local_video_fail_closed(self):
        remote = self._remote()
        remote.job_type = "dagger-observability-sidecar"
        with mock.patch.object(readback, "_load_wandb", return_value=FakeWandb(remote)):
            self.assertEqual(readback.main(self._arguments()), 2)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["last_api_error_type"], "ReadbackError")

        self.output.unlink()
        self.video.unlink()
        stderr = io.StringIO()
        with mock.patch.object(
            readback, "_load_wandb", side_effect=AssertionError("must fail locally")
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(readback.main(self._arguments()), 2)
        self.assertIn("failed (ReadbackError)", stderr.getvalue())

    def test_third_party_error_text_and_credentials_are_not_persisted(self):
        secret = "wandb_v1_should_never_be_reported"
        fake = FakeWandb(self._remote(), error_text=secret)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(readback, "_load_wandb", return_value=fake), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(readback.main(self._arguments()), 2)
        report_text = self.output.read_text(encoding="utf-8")
        self.assertNotIn(secret, report_text + stdout.getvalue() + stderr.getvalue())
        self.assertEqual(json.loads(report_text)["last_api_error_type"], "RuntimeError")

    def test_ready_marker_scope_is_required_before_remote_access(self):
        marker = self.root / "telemetry" / "WANDB_WATCHER_READY.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["telemetry_scope"] = "production"
        marker.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(
            readback, "_load_wandb", side_effect=AssertionError("must fail first")
        ):
            self.assertEqual(readback.main(self._arguments()), 2)

    def test_initialized_marker_is_required_before_remote_access(self):
        (self.root / "telemetry" / "WANDB_SIDECAR_INITIALIZED.json").unlink()
        with mock.patch.object(
            readback, "_load_wandb", side_effect=AssertionError("must fail first")
        ):
            self.assertEqual(readback.main(self._arguments()), 2)


class BoundedWandbWrapperTest(unittest.TestCase):
    def test_wrapper_is_validation_scoped_and_forces_rank_zero_media(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('run_actor_validation.sh', source)
        self.assertNotIn('run_actor_cohort.sh', source)
        self.assertIn('--telemetry-scope bounded-validation', source)
        self.assertIn('readback_actor_validation_wandb.py', source)
        self.assertIn('IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1', source)
        self.assertIn('IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY=1', source)
        self.assertIn('IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY=$STEPS', source)

    def test_wrapper_rejects_dry_run_before_loading_wandb(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "TRAIN_RUN_DIR": str(Path(temporary).resolve() / "run"),
                    "ROUND_DAGGER_VALIDATION_RANKS": "1",
                    "ROUND_DAGGER_VALIDATION_STEPS": "2",
                    "DRY_RUN": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("rejects DRY_RUN", result.stderr)
        self.assertNotIn("missing ROUND_DAGGER_WANDB_PYTHON", result.stderr)

    def test_wrapper_accepts_absolute_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            interpreter = root / "python-real"
            interpreter.write_text("#!/usr/bin/env bash\nexit 2\n", encoding="utf-8")
            interpreter.chmod(0o755)
            virtualenv_python = root / "python"
            virtualenv_python.symlink_to(interpreter.name)
            result = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "TRAIN_RUN_DIR": str(root / "run"),
                    "ROUND_DAGGER_VALIDATION_RANKS": "1",
                    "ROUND_DAGGER_VALIDATION_STEPS": "2",
                    "DRY_RUN": "0",
                    "ROUND_DAGGER_WANDB_PYTHON": str(virtualenv_python),
                    "ROUND_DAGGER_WANDB_PROJECT": "bounded-project",
                    "ROUND_DAGGER_WANDB_ENTITY": "nvidia",
                    "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": "ldu@nvidia.com",
                    # Stop after interpreter validation and before any process
                    # is launched; this test never contacts W&B.
                    "ROUND_DAGGER_WANDB_READY_TIMEOUT_SECONDS": "0",
                },
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("timing values must be positive", result.stderr)
        self.assertNotIn("Python is missing", result.stderr)

    def test_wrapper_sanitizes_external_python_for_watcher_and_readback(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("WANDB_RUNTIME_ENV=(", source)
        self.assertEqual(source.count('"${WANDB_RUNTIME_ENV[@]}"'), 2)
        self.assertIn('PYTHONPATH="$ROOT"', source)
        self.assertIn("PYTHONNOUSERSITE=1", source)
        for name in (
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        ):
            self.assertIn(f"-u {name}", source)
        self.assertNotIn(
            'PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \\\n'
            '    "${WATCHER[@]}"',
            source,
        )

    def test_logical_nonzero_rank_bypasses_wandb_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "TRAIN_RUN_DIR": str(Path(temporary).resolve() / "run"),
                    "ROUND_DAGGER_VALIDATION_RANKS": "2",
                    "ROUND_DAGGER_VALIDATION_ACTOR_RANK": "1",
                    "ROUND_DAGGER_VALIDATION_STEPS": "2",
                    "DRY_RUN": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("GR00T_PYTHON", result.stderr)
        self.assertNotIn("ROUND_DAGGER_WANDB", result.stderr)
        self.assertNotIn("rejects DRY_RUN", result.stderr)

    def test_wrapper_rejects_unsafe_rank_and_single_step_video_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = {
                **os.environ,
                "TRAIN_RUN_DIR": str(Path(temporary).resolve() / "run"),
                "DRY_RUN": "0",
            }
            bad_rank = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env={
                    **base,
                    "ROUND_DAGGER_VALIDATION_RANKS": "1",
                    "ROUND_DAGGER_VALIDATION_ACTOR_RANK": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            one_step = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env={
                    **base,
                    "ROUND_DAGGER_VALIDATION_RANKS": "1",
                    "ROUND_DAGGER_VALIDATION_STEPS": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            missing_isolated_rank = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env={
                    key: value
                    for key, value in {
                        **base,
                        "ROUND_DAGGER_VALIDATION_RANKS": "2",
                    }.items()
                    if key != "ROUND_DAGGER_VALIDATION_ACTOR_RANK"
                },
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(bad_rank.returncode, 2)
        self.assertIn("smaller than validation ranks", bad_rank.stderr)
        self.assertEqual(one_step.returncode, 2)
        self.assertIn("at least two steps", one_step.stderr)
        self.assertEqual(missing_isolated_rank.returncode, 2)
        self.assertIn("explicit isolated actor rank", missing_isolated_rank.stderr)

    def test_wrapper_lifecycle_reaches_terminal_validation_report(self):
        # Reuse the exact bounded actor contract fixture, but replace Isaac and
        # W&B with deterministic local executables.  This exercises the shell
        # lifecycle, shard/seal gate, fresh-media gate, drain, and terminal
        # report without touching a GPU or remote account.
        from tests.test_round_dagger_actor_validation import ActorValidationDryRunTest

        fixture = ActorValidationDryRunTest(methodName="test_one_rank_eval_dry_run_binds_exact_action_contract")
        fixture.setUp()
        try:
            actor_source = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from round_dagger.artifacts import load_strict_json, publish_rollout_shard

cohort = Path(os.environ["ROUND_DAGGER_COHORT_DIR"])
contract = load_strict_json(cohort / "contract.json")
steps = contract["expected_steps_per_env"]
env_ids = np.asarray(contract["expected_global_env_ids"], dtype=np.int64)
envs = len(env_ids)
actions = np.zeros((steps, envs, 26), dtype=np.float32)
arrays = {
    "policy_rgb": np.zeros((steps, envs, 324, 192, 3), dtype=np.uint8),
    "expert_actions": actions.copy(),
    "student_actions": actions.copy(),
    "behavior_actions": actions.copy(),
    "expert_mask": np.zeros((steps, envs), dtype=np.bool_),
    "dones": np.zeros((steps, envs), dtype=np.bool_),
    "observation_is_raytraced": np.zeros((steps, envs), dtype=np.bool_),
    "global_env_ids": env_ids,
    "episode_ids": np.zeros((steps, envs), dtype=np.int64),
    "episode_steps": np.repeat(
        np.arange(steps, dtype=np.int64)[:, None], envs, axis=1
    ),
}
publish_rollout_shard(
    cohort,
    rank=int(os.environ.get("ROUND_DAGGER_ACTOR_RANK", "0")),
    rollout_id=0,
    part_index=0,
    arrays=arrays,
    **{
        key: contract[key]
        for key in (
            "campaign_id", "macro_round", "cohort_index", "purpose",
            "policy_sha256", "runtime_policy_sha256", "teacher_sha256",
            "renderer_sha256", "beta_sha256", "config_sha256",
        )
    },
)
root = Path(os.environ["TRAIN_RUN_DIR"]) / "telemetry"
for kind in ("scalars", "images", "videos"):
    (root / kind).mkdir(parents=True, exist_ok=True)
(root / "scalars" / "step_000000000.json").write_text(
    '{"step":0,"metrics":{"renderer/latency_ms":1.0}}\n', encoding="utf-8"
)
(root / "images" / "step_000000000_conditions_generated.png").write_bytes(b"png")
(root / "videos" / "step_000000001_generated_triplet.mp4").write_bytes(b"mp4")
'''
            fixture.fake_python.write_text(actor_source, encoding="utf-8")
            fixture.fake_python.chmod(0o755)

            fake_wandb_python = fixture.root / "wandb-python"
            fake_wandb_python.write_text(
                r'''#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

script = Path(sys.argv[1]).name
args = sys.argv[2:]
expected_root = str(Path(sys.argv[1]).resolve().parents[2])
if os.environ.get("PYTHONPATH") != expected_root:
    raise SystemExit(17)
if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise SystemExit(18)
for inherited in (
    "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
):
    if inherited in os.environ:
        raise SystemExit(19)
def value(flag):
    return args[args.index(flag) + 1]
if script == "wandb_run_watcher.py":
    run_dir = Path(value("--run-dir"))
    ready = Path(value("--ready-marker"))
    run_id_path = run_dir / "telemetry" / "WANDB_SIDECAR_RUN_ID"
    if not run_id_path.exists():
        run_id_path.write_text("local_bounded_run\n", encoding="utf-8")
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("{}\n", encoding="utf-8")
    done = run_dir / value("--finish-marker")
    for _ in range(1000):
        if done.exists():
            raise SystemExit(0)
        time.sleep(0.01)
    raise SystemExit(2)
if script == "readback_actor_validation_wandb.py":
    output = Path(value("--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "fast_roborender.round_dagger.wandb_actor_validation_readback.v1",
        "status": "PASS",
        "telemetry_scope": "bounded-validation",
        "trainer_run_required": False,
    }) + "\n", encoding="utf-8")
    raise SystemExit(0)
raise SystemExit(2)
''',
                encoding="utf-8",
            )
            fake_wandb_python.chmod(0o755)
            environment = fixture._environment(
                ranks=1, envs_per_rank=1, steps=2, dry_run=False
            )
            environment.update(
                {
                    "ROUND_DAGGER_WANDB_PYTHON": str(fake_wandb_python),
                    "ROUND_DAGGER_WANDB_PROJECT": "bounded-project",
                    "ROUND_DAGGER_WANDB_ENTITY": "nvidia",
                    "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": "ldu@nvidia.com",
                    "ROUND_DAGGER_WANDB_READY_TIMEOUT_SECONDS": "5",
                    "ROUND_DAGGER_WANDB_READBACK_TIMEOUT_SECONDS": "5",
                    "ROUND_DAGGER_WANDB_POLL_SECONDS": "1",
                    # Reproduce the actor image entrypoint contamination.  The
                    # actor keeps this path, while both external W&B Python
                    # invocations must receive the exact repository path only.
                    "PYTHONPATH": f"/isaac-sim/kit/python/lib/python3.11:{ROOT}",
                    "VIRTUAL_ENV": "/opt/round-dagger-rtx",
                    "CONDA_PREFIX": "/opt/isaac-conda",
                    "CONDA_DEFAULT_ENV": "isaac",
                }
            )
            result = subprocess.run(
                ["bash", str(WRAPPER)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ROUND_ACTOR_VALIDATION_WANDB_OK", result.stdout)
            report = Path(environment["TRAIN_RUN_DIR"]) / "validation" / "wandb_actor_validation_readback.json"
            self.assertEqual(json.loads(report.read_text())["status"], "PASS")
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
