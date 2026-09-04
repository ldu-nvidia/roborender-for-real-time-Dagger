"""Tests for the root-container to RTX-host telemetry handoff."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from round_dagger.actor_host_access import (
    ActorHostAccessError,
    HOST_MUTABLE_DIRECTORIES,
    normalize_container_outputs,
    prepare_host_layout,
    verify_host_access,
)


ROOT = Path(__file__).resolve().parents[1]
HOST_ACCESS_DOCKER = (
    ROOT / "scripts/round_dagger/run_actor_host_access_docker.sh"
)


class ActorHostAccessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "run"
        self.run.mkdir(mode=0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_host_precreates_every_shared_directory_before_container(self) -> None:
        report = prepare_host_layout(self.run)
        self.assertEqual(report["status"], "HOST_ACCESSIBLE")
        self.assertEqual(report["host_uid"], os.geteuid())
        for relative in HOST_MUTABLE_DIRECTORIES:
            path = self.run / relative
            info = os.lstat(path)
            self.assertTrue(stat.S_ISDIR(info.st_mode))
            self.assertEqual(info.st_uid, os.geteuid())
            self.assertTrue(stat.S_IMODE(info.st_mode) & stat.S_IWUSR)

    def test_normalizer_recovers_container_modes_without_touching_cohort(self) -> None:
        prepare_host_layout(self.run)
        cohort = self.root / "cohort_00"
        cohort.mkdir()
        shard = cohort / "rollout.npz"
        shard.write_bytes(b"immutable-rollout")
        shard.chmod(0o444)
        before = (shard.read_bytes(), shard.stat().st_mode, shard.stat().st_mtime_ns)

        telemetry = self.run / "telemetry"
        scalar = telemetry / "scalars/step_000000511.json"
        scalar.write_text('{"step":511,"metrics":{"loss":1.0}}\n')
        state = telemetry / ".wandb_upload_state.json"
        state.write_text("{}\n")
        state.chmod(0o600)
        (telemetry / "scalars").chmod(0o555)
        private = self.run / ".wandb_media/wandb/private-state"
        private.parent.mkdir(parents=True)
        private.write_text("state\n")
        private.chmod(0o400)

        with self.assertRaises(ActorHostAccessError):
            verify_host_access(self.run)

        report = normalize_container_outputs(
            self.run, host_uid=os.geteuid(), host_gid=os.getegid()
        )
        self.assertEqual(report["status"], "NORMALIZED")
        verified = verify_host_access(self.run)
        self.assertEqual(verified["status"], "HOST_ACCESSIBLE")
        self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(scalar.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
        self.assertEqual(
            (shard.read_bytes(), shard.stat().st_mode, shard.stat().st_mtime_ns),
            before,
        )

    def test_normalizer_rejects_symlink_without_following_it(self) -> None:
        prepare_host_layout(self.run)
        victim = self.root / "victim"
        victim.write_bytes(b"preserve")
        linked = self.run / "telemetry/scalars/linked.json"
        linked.symlink_to(victim)
        with self.assertRaisesRegex(ActorHostAccessError, "symlink|symbolic"):
            normalize_container_outputs(
                self.run, host_uid=os.geteuid(), host_gid=os.getegid()
            )
        self.assertEqual(victim.read_bytes(), b"preserve")
        self.assertTrue(linked.is_symlink())

    def test_normalizer_removes_only_live_shape_wandb_symlinks(self) -> None:
        prepare_host_layout(self.run)
        wandb = self.run / ".wandb_media/wandb"
        run = wandb / "run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD"
        logs = run / "logs"
        logs.mkdir(parents=True)
        debug = logs / "debug.log"
        internal = logs / "debug-internal.log"
        debug.write_text("debug\n", encoding="utf-8")
        internal.write_text("internal\n", encoding="utf-8")
        (wandb / "latest-run").symlink_to(run.name)
        (wandb / "debug.log").symlink_to(f"{run.name}/logs/debug.log")
        (wandb / "debug-internal.log").symlink_to(
            f"{run.name}/logs/debug-internal.log"
        )
        (logs / "debug-core.log").symlink_to(
            "/root/.cache/wandb/logs/core-debug-live.log"
        )

        report = normalize_container_outputs(
            self.run, host_uid=os.geteuid(), host_gid=os.getegid()
        )
        self.assertEqual(report["removed_wandb_symlinks"], 4)
        for path in (
            wandb / "latest-run",
            wandb / "debug.log",
            wandb / "debug-internal.log",
            logs / "debug-core.log",
        ):
            self.assertFalse(os.path.lexists(path))
        self.assertEqual(debug.read_text(encoding="utf-8"), "debug\n")
        self.assertEqual(internal.read_text(encoding="utf-8"), "internal\n")
        self.assertEqual(verify_host_access(self.run)["status"], "HOST_ACCESSIBLE")

    def test_normalizer_rejects_unknown_wandb_symlink_without_unlinking(self) -> None:
        prepare_host_layout(self.run)
        victim = self.root / "victim"
        victim.write_bytes(b"preserve")
        unknown = self.run / ".wandb_media/wandb/unknown-link"
        unknown.parent.mkdir(parents=True)
        unknown.symlink_to(victim)
        with self.assertRaisesRegex(ActorHostAccessError, "unknown symlink"):
            normalize_container_outputs(
                self.run, host_uid=os.geteuid(), host_gid=os.getegid()
            )
        self.assertTrue(unknown.is_symlink())
        self.assertEqual(victim.read_bytes(), b"preserve")

    def test_requested_identity_must_own_run_root(self) -> None:
        prepare_host_layout(self.run)
        with self.assertRaisesRegex(ActorHostAccessError, "does not own"):
            normalize_container_outputs(
                self.run, host_uid=os.geteuid() + 1, host_gid=os.getegid()
            )


class ActorHostAccessDockerIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.campaign = self.root / "campaign"
        self.run = (
            self.campaign
            / "actors/dagger_prod_run1/macro_round_000000/cohort_00_run"
        )
        self.run.mkdir(mode=0o700, parents=True)
        self.argv = self.root / "docker.argv"
        self.docker = self.root / "docker"
        self.docker.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "if [[ ${1:-} == image && ${2:-} == inspect ]]; then\n"
            "  printf 'sha256:%s\\n' \"$ROUND_DAGGER_RTX_IMAGE_ID_SHA256\"\n"
            "  exit 0\n"
            "fi\n"
            f"printf '%s\\n' \"$@\" > {self.argv!s}\n",
            encoding="utf-8",
        )
        self.docker.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "ROUND_DAGGER_TRUSTED_COORDINATOR_HOST_ACCESS": "1",
            "ROUND_DAGGER_COORDINATOR_SOURCE_ROOT": str(ROOT),
            "ROUND_DAGGER_COORDINATOR_SOURCE_INVENTORY_SHA256": "a" * 64,
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(self.campaign),
            "TRAIN_RUN_DIR": str(self.run),
            "ROUND_DAGGER_DOCKER_BIN": str(self.docker),
            "ROUND_DAGGER_RTX_IMAGE": "actor:pinned",
            "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": "b" * 64,
            "ROUND_DAGGER_CAMPAIGN_ID": "dagger_prod_run1",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_COHORT_INDEX": "0",
            "WANDB_API_KEY": "must-not-be-forwarded",
        }

    def test_root_helper_is_pinned_narrow_and_has_no_network_or_gpu(self) -> None:
        result = subprocess.run(
            ["bash", str(HOST_ACCESS_DOCKER)],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv.read_text(encoding="utf-8").splitlines()
        self.assertIn("--user", argv)
        self.assertIn("0:0", argv)
        self.assertIn("--read-only", argv)
        self.assertIn("--cap-drop=ALL", argv)
        self.assertIn("--cap-add=CHOWN", argv)
        self.assertIn("--cap-add=FOWNER", argv)
        self.assertIn("--cap-add=DAC_OVERRIDE", argv)
        self.assertIn("--security-opt=no-new-privileges", argv)
        self.assertIn("--ipc=private", argv)
        self.assertIn("--network=none", argv)
        self.assertNotIn("--gpus", argv)
        self.assertIn(f"type=bind,src={ROOT},dst={ROOT},readonly", argv)
        self.assertIn(
            f"type=bind,src={self.run},dst={self.run}", argv
        )
        self.assertNotIn("WANDB_API_KEY", argv)
        self.assertNotIn("must-not-be-forwarded", argv)
        self.assertIn(f"sha256:{'b' * 64}", argv)
        command_start = argv.index("python3")
        self.assertEqual(
            argv[command_start : command_start + 4],
            ["python3", "-m", "round_dagger.actor_host_access", "normalize"],
        )

    def test_real_docker_root_output_is_handed_back_to_host_identity(self) -> None:
        """Opt-in proof using the same root-container boundary as production."""

        if os.environ.get("ROUND_DAGGER_TEST_DOCKER_ROOT_HANDOFF") != "1":
            self.skipTest("real Docker cross-UID handoff test is opt-in")
        docker = shutil.which("docker")
        image = os.environ.get("ROUND_DAGGER_TEST_RTX_IMAGE")
        if docker is None or image is None:
            self.skipTest("Docker and ROUND_DAGGER_TEST_RTX_IMAGE are required")

        prepare_host_layout(self.run)
        cohort = self.root / "cohort_00"
        cohort.mkdir()
        shard = cohort / "rollout.npz"
        shard.write_bytes(b"sealed-cohort-must-not-change")
        shard.chmod(0o444)
        before = (shard.read_bytes(), shard.stat().st_mode, shard.stat().st_mtime_ns)

        container = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "--user",
                "0:0",
                "--mount",
                f"type=bind,src={self.run},dst=/handoff",
                image,
                "/usr/bin/env",
                "bash",
                "-c",
                (
                    "set -Eeuo pipefail; "
                    "mkdir -p /handoff/.wandb_media/wandb; "
                    "printf '{}\\n' > /handoff/telemetry/.wandb_upload_state.json; "
                    "printf '{}\\n' > /handoff/telemetry/scalars/root.json; "
                    "printf 'private\\n' > /handoff/.wandb_media/wandb/private; "
                    "mkdir -p /handoff/.wandb_media/wandb/"
                    "run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD/logs; "
                    "printf 'debug\\n' > /handoff/.wandb_media/wandb/"
                    "run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD/logs/debug.log; "
                    "printf 'internal\\n' > /handoff/.wandb_media/wandb/"
                    "run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD/logs/"
                    "debug-internal.log; "
                    "ln -s run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD "
                    "/handoff/.wandb_media/wandb/latest-run; "
                    "ln -s run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD/logs/debug.log "
                    "/handoff/.wandb_media/wandb/debug.log; "
                    "ln -s run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD/"
                    "logs/debug-internal.log /handoff/.wandb_media/wandb/"
                    "debug-internal.log; "
                    "ln -s /root/.cache/wandb/logs/core-debug-live.log "
                    "/handoff/.wandb_media/wandb/"
                    "run-20260831_013222-vAnnvrfGIijU3ETLS7mHIXoD/logs/debug-core.log; "
                    "chown -R 0:0 /handoff/telemetry/scalars "
                    "/handoff/.wandb_media; "
                    "chown 0:0 /handoff/telemetry/.wandb_upload_state.json; "
                    "chmod 0755 /handoff/telemetry/scalars; "
                    "chmod 0600 /handoff/telemetry/.wandb_upload_state.json "
                    "/handoff/telemetry/scalars/root.json "
                    "/handoff/.wandb_media/wandb/private"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(container.returncode, 0, container.stderr)
        self.assertEqual((self.run / "telemetry/scalars/root.json").stat().st_uid, 0)
        with self.assertRaises(ActorHostAccessError):
            verify_host_access(self.run)

        image_id = subprocess.run(
            [docker, "image", "inspect", image, "--format", "{{.Id}}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        self.assertRegex(image_id, r"^sha256:[0-9a-f]{64}$")
        environment = {
            **self._environment(),
            "ROUND_DAGGER_DOCKER_BIN": docker,
            "ROUND_DAGGER_RTX_IMAGE": image,
            "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": image_id.removeprefix("sha256:"),
        }
        result = subprocess.run(
            ["bash", str(HOST_ACCESS_DOCKER)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        verified = verify_host_access(self.run)
        self.assertEqual(verified["status"], "HOST_ACCESSIBLE")
        self.assertFalse(
            os.path.lexists(self.run / ".wandb_media/wandb/latest-run")
        )
        self.assertFalse(
            os.path.lexists(self.run / ".wandb_media/wandb/debug.log")
        )
        self.assertFalse(
            os.path.lexists(self.run / ".wandb_media/wandb/debug-internal.log")
        )
        self.assertEqual(
            (shard.read_bytes(), shard.stat().st_mode, shard.stat().st_mtime_ns),
            before,
        )


if __name__ == "__main__":
    unittest.main()
