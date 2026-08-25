"""Focused fail-closed gates for the production actor collect hook."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from round_dagger import artifacts
from scripts.round_dagger.verify_anchor_pool import (
    AnchorPoolError,
    AnchorRecord,
    inventory_sha256,
    verify_anchor_pool,
)
from tests.test_round_dagger_handoff import publish_policy_bundle


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/round_dagger/verify_anchor_pool.py"
HOOK = ROOT / "scripts/round_dagger/run_production_actor_cohort_hook.sh"


def _records(root: Path) -> list[AnchorRecord]:
    result = []
    for path in sorted(root.rglob("*.npz")):
        payload = path.read_bytes()
        result.append(
            AnchorRecord(
                relative_path=path.relative_to(root).as_posix(),
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return result


class ExactAnchorPoolVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.anchors = self.root / "anchors"
        (self.anchors / "nested").mkdir(parents=True)
        (self.anchors / "anchor_000.npz").write_bytes(b"anchor-zero")
        (self.anchors / "nested/anchor_001.npz").write_bytes(b"anchor-one")
        # Regular non-NPZ metadata is deliberately outside the NPZ inventory.
        (self.anchors / "README.md").write_text("metadata\n", encoding="utf-8")
        self.digest = inventory_sha256(_records(self.anchors))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_inventory_passes_cli_and_library(self) -> None:
        report = verify_anchor_pool(
            self.anchors, expected_count=2, expected_sha256=self.digest
        )
        self.assertEqual(report["anchor_count"], 2)
        self.assertEqual(report["inventory_sha256"], self.digest)
        result = subprocess.run(
            [
                "python3",
                str(VERIFIER),
                "--root",
                str(self.anchors),
                "--expected-count",
                "2",
                "--expected-sha256",
                self.digest,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status":"verified"', result.stdout)

    def test_mutation_and_extra_npz_fail_closed(self) -> None:
        (self.anchors / "anchor_000.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(AnchorPoolError, "SHA-256 mismatch"):
            verify_anchor_pool(
                self.anchors, expected_count=2, expected_sha256=self.digest
            )
        (self.anchors / "anchor_002.npz").write_bytes(b"extra")
        with self.assertRaisesRegex(AnchorPoolError, "expected exactly 2"):
            verify_anchor_pool(
                self.anchors, expected_count=2, expected_sha256=self.digest
            )

    def test_symlink_and_special_file_fail_closed(self) -> None:
        link = self.anchors / "alias.npz"
        link.symlink_to(self.anchors / "anchor_000.npz")
        with self.assertRaisesRegex(AnchorPoolError, "symbolic links are forbidden"):
            verify_anchor_pool(
                self.anchors, expected_count=2, expected_sha256=self.digest
            )
        link.unlink()
        fifo = self.anchors / "control.fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(AnchorPoolError, "special files are forbidden"):
            verify_anchor_pool(
                self.anchors, expected_count=2, expected_sha256=self.digest
            )

    def test_root_symlink_is_rejected(self) -> None:
        alias = self.root / "anchor-alias"
        alias.symlink_to(self.anchors, target_is_directory=True)
        with self.assertRaisesRegex(AnchorPoolError, "real directory"):
            verify_anchor_pool(alias, expected_count=2, expected_sha256=self.digest)


class ProductionCollectHookStaticContractTest(unittest.TestCase):
    def test_hook_pins_all_accepted_production_identities(self) -> None:
        source = HOOK.read_text(encoding="utf-8")
        for expected in (
            "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38",
            "a82f639a9038789db220793475dfce3eb5651ff3145287be748f387a20caeb9f",
            "d075c02b3e8aace7c1a4b2ef1e0b694738052da66bea6300299a7c4fd6956be6",
            "ee358a97581512a5d03730d06efd5804e53844f517112dc0a845ed86b6dae46f",
            "sha256:65a0cbe64eb71ee2edb672bed3ec7bdaa8adfeb22298c0e99af8b91d74dc78f8",
            "53234dfe2eab9b1ae02df21e143f996ecde831aadb8aacb2e5f2df632d3d8d27",
            '"256:4:64:300:76800"',
            '"64:19200"',
        ):
            self.assertIn(expected, source)
        self.assertIn('exec "$PAIR_SCRIPT"', source)
        self.assertNotIn("WANDB_API_KEY", source)

    def test_missing_identity_fails_before_any_actor_launch(self) -> None:
        result = subprocess.run(
            ["bash", str(HOOK)],
            env={"PATH": os.environ.get("PATH", "")},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing required environment", result.stderr)
        self.assertNotIn("WANDB", result.stdout + result.stderr)


class ProductionCollectHookRealDryRunTest(unittest.TestCase):
    """Exercise the real accepted assets while guaranteeing no actor starts."""

    TEACHER = ROOT / (
        "runtime/raven_handoff_6f4e682_gr00t/logs_rl/"
        "walk_stand_place_grasp_turn_homie_predrop_project_name-"
        "wsdpt_teacher_BTFv8q9_qben_exppredrop_leanz_hand_predrop_"
        "pos_distance-10.0_penalty_bottle_lean_during_pick--3500.0_"
        "penalty_bottle_non_z_up_velocity_during_pick--1000.0-"
        "20251105_004910/model_step_044500.pt"
    )
    ENDPOINTS = Path(
        "/mnt/data2/ldu/roborender_round_dagger_validation/"
        "fleet_two_gpu_provenance_v2/endpoint_manifest.txt"
    )
    ANCHORS = ROOT / "runtime/hsg_stage_transfer/anchor_pool_2048"
    IMAGE = "roborender-round-actor:isaaclab2.3.0-local-v2"
    CAMERA = Path(
        "/mnt/data2/ldu/roborender_round_dagger_validation/production_inputs_v1/"
        "camera/camera_calibration_acceptance.json"
    )
    WANDB_NETRC = ROOT / "runtime/round_dagger_credentials/.netrc"

    def setUp(self) -> None:
        checks = (
            self.TEACHER.is_file(),
            self.ENDPOINTS.is_file(),
            self.ANCHORS.is_dir(),
            Path("/usr/bin/docker").is_file(),
            self.CAMERA.is_file(),
            self.WANDB_NETRC.is_file(),
        )
        if not all(checks):
            self.skipTest("RTX production assets are not present on this host")
        inspect = subprocess.run(
            ["/usr/bin/docker", "image", "inspect", self.IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode != 0:
            self.skipTest("accepted RTX actor image is not present on this host")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.checkpoint = self.root / "student-v0.pt"
        self.checkpoint.write_bytes(b"fresh-policy")
        self.runtime_sha = "2" * 64
        self.spec, self.manifest = publish_policy_bundle(
            self.checkpoint, self.runtime_sha
        )
        self.policy_sha = artifacts.sha256_file(self.checkpoint)
        self.cohort = self.root / "actors/cohort_00"
        artifacts.create_cohort(
            self.cohort,
            campaign_id="production-collect-dry-run",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            expected_ranks=[0, 1],
            expected_global_env_ids=list(range(64)),
            expected_transitions=19200,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [324, 192, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            action_contract={"dtype": "float32", "shape": [26]},
            policy_sha256=self.policy_sha,
            runtime_policy_sha256=self.runtime_sha,
            teacher_sha256=(
                "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38"
            ),
            renderer_sha256=(
                "a82f639a9038789db220793475dfce3eb5651ff3145287be748f387a20caeb9f"
            ),
            beta_sha256="5" * 64,
            config_sha256="6" * 64,
        )

    def tearDown(self) -> None:
        temporary = getattr(self, "temporary", None)
        if temporary is not None:
            temporary.cleanup()

    def test_full_preflight_reaches_only_pair_dry_run(self) -> None:
        run_dir = self.root / "actors/cohort_00_run"
        environment = {
            **os.environ,
            "DRY_RUN": "1",
            "GR00T_PYTHON": "/usr/bin/python3",
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(self.root),
            "ROUND_DAGGER_CAMPAIGN_ID": "production-collect-dry-run",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_COHORT_INDEX": "0",
            "ROUND_DAGGER_COHORT_DIR": str(self.cohort),
            "ROUND_DAGGER_GLOBAL_ENV_OFFSET": "0",
            "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
            "ROUND_DAGGER_COHORT_COUNT": "4",
            "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
            "ROUND_DAGGER_LOGICAL_ENV_COUNT": "64",
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "300",
            "ROUND_DAGGER_EXPECTED_TRANSITIONS": "19200",
            "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "76800",
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(self.checkpoint),
            "ROUND_DAGGER_POLICY_SHA256": self.policy_sha,
            "ROUND_DAGGER_POLICY_SPEC": str(self.spec),
            "ROUND_DAGGER_POLICY_MANIFEST": str(self.manifest),
            "ROUND_DAGGER_POLICY_SPEC_SHA256": artifacts.sha256_file(self.spec),
            "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(
                self.manifest
            ),
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": self.runtime_sha,
            "ROUND_DAGGER_TEACHER_CHECKPOINT": str(self.TEACHER),
            "ROUND_DAGGER_TEACHER_SHA256": (
                "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38"
            ),
            "ROUND_DAGGER_RENDERER_SHA256": (
                "a82f639a9038789db220793475dfce3eb5651ff3145287be748f387a20caeb9f"
            ),
            "ROUND_DAGGER_BETA": "1.0",
            "ROUND_DAGGER_BETA_SHA256": "5" * 64,
            "ROUND_DAGGER_CONFIG_SHA256": "6" * 64,
            "IMAGE3F_ENDPOINT_MANIFEST": str(self.ENDPOINTS),
            "ANCHORS": str(self.ANCHORS),
            "ROUND_DAGGER_RTX_IMAGE": self.IMAGE,
            "ROUND_DAGGER_DOCKER_BIN": "/usr/bin/docker",
            "ROUND_DAGGER_CAMERA_ACCEPTANCE": str(self.CAMERA),
            "ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256": artifacts.sha256_file(
                self.CAMERA
            ),
            "ROUND_DAGGER_WANDB_NETRC": str(self.WANDB_NETRC),
            "ROUND_DAGGER_ACTOR_RUN_DIR": str(run_dir),
            "TRAIN_RUN_DIR": str(run_dir),
            "ROUND_DAGGER_WANDB_ENABLED": "0",
        }
        result = subprocess.run(
            ["bash", str(HOOK)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("production_collect_preflight.v1", result.stdout)
        self.assertIn("DRY_RUN_ACTOR_PAIR_OK", result.stdout)
        self.assertFalse((self.cohort / "seal.json").exists())


if __name__ == "__main__":
    unittest.main()
