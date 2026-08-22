from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixed = load_module("fixed_seed_gate", ROOT / "scripts/eval/fixed_seed_gate.py")
# The HSG module imports the shared implementation by its normal sibling name.
sys.modules["fixed_seed_gate"] = fixed
hsg = load_module(
    "hsg_gb200_fixed_seed_eval_gate",
    ROOT / "scripts/eval/hsg_gb200_fixed_seed_eval_gate.py",
)
wandb_upload = load_module(
    "hsg_gb200_fixed_seed_wandb",
    ROOT / "scripts/eval/hsg_gb200_fixed_seed_wandb.py",
)
team_contract = sys.modules["wandb_team_contract"]


def write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class HsgGb200FixedSeedGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "fast_roborender"
        self.gr00t = self.root / "GR00T-VisualSim2Real"
        self.fast_revision = "1" * 40
        self.gr00t_revision = "2" * 40
        write(self.root / ".source_revision", (self.fast_revision + "\n").encode())
        write(self.root / "scripts/tool.py", b"value = 1\n")
        write(self.gr00t / ".source_revision", (self.gr00t_revision + "\n").encode())
        write(self.gr00t / "gr00t/rl/eval_agent_trl.py", b"# evaluator\n")
        self.container = self.base / "trainer.sqsh"
        self.container_hash = write(self.container, b"sealed-sm100-trainer")
        self.run_dir = self.base / "train"
        self.baseline = self.run_dir / "initial_policy.pt"
        write(self.baseline, b"baseline")
        for step in (5, 10, 15):
            write(self.run_dir / f"model_step_{step:06d}.pt", f"step-{step}".encode())
        self.config = self.run_dir / "config.yaml"
        write(
            self.config,
            b"policy_view_count: 3\nsimulator:\n  config:\n    cameras:\n"
            b"      policy_views: [head_color, hand_left, hand_right]\n",
        )
        self.manifest = self.base / "eval/manifest.json"
        self.wandb_acceptance = self.base / "eval/wandb_online_smoke_nvidia_acceptance.json"
        write(
            self.wandb_acceptance,
            (
                json.dumps(
                    {
                        "schema": team_contract.ONLINE_SMOKE_SCHEMA,
                        "status": "PASS",
                        **team_contract.team_binding(),
                        "observed_keys": sorted(team_contract.REQUIRED_OBSERVED_KEYS),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        fixed.create_manifest(
            train_run_dir=self.run_dir,
            baseline_checkpoint=self.baseline,
            reference_config=self.config,
            seeds=(11, 29),
            num_envs=4,
            output_path=self.manifest,
            expected_baseline_sha256=fixed.sha256_file(self.baseline),
            expected_reference_config_sha256=fixed.sha256_file(self.config),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def preflight_args(self):
        return argparse.Namespace(
            root=self.root,
            gr00t_root=self.gr00t,
            fast_revision=self.fast_revision,
            gr00t_revision=self.gr00t_revision,
            trainer_container=self.container,
            trainer_container_sha256=self.container_hash,
            fixed_seed_manifest=self.manifest,
            fixed_seed_manifest_sha256=hsg.sha256_file(self.manifest),
            train_run_dir=self.run_dir,
            expected_seeds="11,29",
            expected_num_envs=4,
            wandb_acceptance=self.wandb_acceptance,
            wandb_acceptance_sha256=hsg.sha256_file(self.wandb_acceptance),
        )

    def test_preflight_binds_sources_container_config_and_four_checkpoints(self):
        report = hsg.preflight(self.preflight_args())
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["observation_mode"], "simulator_rgb")
        self.assertFalse(report["image3f_in_the_loop"])
        self.assertFalse(report["production_observation_distribution_validated"])
        self.assertEqual(len(report["fixed_seed"]["roles"]), 4)
        self.assertEqual(report["fixed_seed"]["roles"][-1]["step"], 15)
        self.assertEqual(report["trainer_container_sha256"], self.container_hash)
        self.assertEqual(report["wandb_online_acceptance"]["entity"], "nvidia")
        self.assertEqual(
            report["wandb_online_acceptance"]["authorization_scope"], "team"
        )
        self.assertFalse(report["wandb_online_acceptance"]["personal"])

    def test_preflight_rejects_checkpoint_and_observation_substitution(self):
        middle = self.run_dir / "model_step_000010.pt"
        middle.write_bytes(b"changed")
        with self.assertRaisesRegex(hsg.GateError, "checkpoint changed"):
            hsg.preflight(self.preflight_args())
        middle.write_bytes(b"step-10")
        payload = json.loads(self.manifest.read_text())
        payload["observation_mode"] = "image3f"
        self.manifest.write_text(json.dumps(payload) + "\n")
        args = self.preflight_args()
        args.fixed_seed_manifest_sha256 = hsg.sha256_file(self.manifest)
        with self.assertRaisesRegex(hsg.GateError, "simulator-RGB/raytraced"):
            hsg.preflight(args)

    def make_results_and_summary(self):
        manifest = json.loads(self.manifest.read_text())
        results = self.base / "eval/results"
        success = {"baseline": 1, "early": 2, "mid": 3, "final": 4}
        reward = {"baseline": -5.0, "early": -4.0, "mid": -3.0, "final": -2.0}
        for role in manifest["checkpoint_roles"]:
            label = role["label"]
            for seed in manifest["seeds"]:
                row = fixed.build_result(
                    manifest=manifest,
                    label=label,
                    seed=seed,
                    metrics={
                        "collected_episodes": 4,
                        "goal_reached_rate": success[label] / 4,
                        "reward": reward[label],
                        "episode_length": 100.0,
                    },
                    evaluated_checkpoint_sha256="a" * 64,
                    evaluated_config_sha256=manifest["reference_config_sha256"],
                )
                target = results / label / f"seed_{seed}/result.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        summary = fixed.summarize_results(
            manifest=manifest,
            results_dir=results,
            min_success_delta=0.02,
        )
        summary_path = self.base / "eval/summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return results, summary_path

    def test_postrun_recomputes_paired_gate_and_requires_wandb_marker(self):
        preflight = self.base / "eval/preflight.json"
        preflight.write_text(
            json.dumps(hsg.preflight(self.preflight_args()), indent=2, sort_keys=True) + "\n"
        )
        results, summary = self.make_results_and_summary()
        wandb = self.base / "eval/wandb_upload.json"
        wandb.write_text(
            json.dumps(
                {
                    "schema": hsg.WANDB_SCHEMA,
                    "status": "PASS",
                    **team_contract.team_binding(),
                    "manifest_sha256": hsg.sha256_file(self.manifest),
                    "summary_sha256": hsg.sha256_file(summary),
                    "wandb_online_acceptance_sha256": hsg.sha256_file(
                        self.wandb_acceptance
                    ),
                    "wandb_online_acceptance_observed_keys": sorted(
                        team_contract.REQUIRED_OBSERVED_KEYS
                    ),
                    "observed_checkpoint_ordinals": [0, 1, 2, 3],
                    "observed_history_keys": list(wandb_upload.HISTORY_KEYS),
                    "observed_summary_keys": list(wandb_upload.SUMMARY_KEYS),
                }
            )
            + "\n"
        )
        args = argparse.Namespace(
            preflight=preflight,
            preflight_sha256=hsg.sha256_file(preflight),
            fixed_seed_manifest=self.manifest,
            fixed_seed_manifest_sha256=hsg.sha256_file(self.manifest),
            results_dir=results,
            summary=summary,
            min_success_delta=0.02,
            wandb_upload=wandb,
            expected_wandb_email="ldu@nvidia.com",
            expected_wandb_entity="nvidia",
        )
        report = hsg.postrun(args)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["readiness_scope"], "secondary_policy_mechanics_gate")
        self.assertFalse(report["production_observation_distribution_validated"])
        self.assertEqual(report["wandb_entity"], "nvidia")
        self.assertEqual(report["wandb_authorization_scope"], "team")
        self.assertEqual(report["wandb_authorization"], team_contract.team_binding())
        self.assertGreater(report["paired_success_delta_ci95_low"], 0.0)
        self.assertEqual(len(report["results"]), 8)

        preflight_payload = json.loads(preflight.read_text())
        preflight_payload["fixed_seed"]["manifest_sha256"] = "b" * 64
        preflight.write_text(json.dumps(preflight_payload) + "\n")
        args.preflight_sha256 = hsg.sha256_file(preflight)
        with self.assertRaisesRegex(hsg.GateError, "manifest bound by HSG preflight"):
            hsg.postrun(args)
        preflight.write_text(
            json.dumps(hsg.preflight(self.preflight_args()), indent=2, sort_keys=True) + "\n"
        )
        args.preflight_sha256 = hsg.sha256_file(preflight)

        payload = json.loads(wandb.read_text())
        payload.pop("observed_summary_keys")
        wandb.write_text(json.dumps(payload) + "\n")
        with self.assertRaisesRegex(hsg.GateError, "summary keys"):
            hsg.postrun(args)
        payload["observed_summary_keys"] = list(wandb_upload.SUMMARY_KEYS)
        payload["personal"] = True
        wandb.write_text(json.dumps(payload) + "\n")
        with self.assertRaisesRegex(hsg.GateError, "personal mismatch"):
            hsg.postrun(args)
        payload["personal"] = False
        payload["entity"] = "wrong"
        wandb.write_text(json.dumps(payload) + "\n")
        with self.assertRaisesRegex(hsg.GateError, "entity mismatch"):
            hsg.postrun(args)

    def test_launchers_pin_hsg_resources_team_wandb_and_simulator_rgb(self):
        submit = (ROOT / "scripts/eval/submit_hsg_gb200_fixed_seed_gate.sh").read_text()
        packed = (ROOT / "scripts/eval/hsg_gb200_fixed_seed_policy_eval.sbatch").read_text()
        task = (ROOT / "scripts/eval/hsg_gb200_fixed_seed_task.sh").read_text()
        for path in (
            ROOT / "scripts/eval/submit_hsg_gb200_fixed_seed_gate.sh",
            ROOT / "scripts/eval/hsg_gb200_fixed_seed_policy_eval.sbatch",
            ROOT / "scripts/eval/hsg_gb200_fixed_seed_task.sh",
        ):
            parsed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
        for contract in (
            "--partition=batch_long",
            "--time=08:00:00",
            "--gpus-per-node=4",
            "roborender-isaac-gr00t-arm64-sm100-isaac5.1-r5.sqsh",
            "WANDB_ONLINE_VALIDATION",
            "wandb_online_smoke_nvidia_acceptance.json",
            "authorized-team W&B scalar/image/video acceptance is missing",
            'fixed_seed_gate.py" prepare',
        ):
            self.assertIn(contract, submit)
        self.assertIn("--ntasks=4", packed)
        self.assertIn("#SBATCH --partition=batch_long", packed)
        self.assertIn("#SBATCH --time=08:00:00", packed)
        self.assertGreaterEqual(packed.count("--gpus=0"), 2)
        self.assertIn('hsg_gb200_fixed_seed_wandb.py"', packed)
        self.assertIn('wandb_team_contract.py"', packed)
        self.assertIn('--authorization-scope "$WANDB_AUTHORIZATION_SCOPE"', packed)
        self.assertIn('hsg_gb200_fixed_seed_eval_gate.py" postrun', packed)
        self.assertIn("IMAGE_DAGGER_EVAL_RELABEL=0", task)
        self.assertIn("head_color,hand_left,hand_right", task)
        self.assertIn('HOME="$EVAL_DIR/runtime/$LABEL/home"', task)
        self.assertIn("secondary_policy_mechanics_gate", submit)
        self.assertIn("image3f_in_the_loop=false", packed)
        self.assertNotIn("WANDB_API_KEY", submit + packed + task)

    def test_wandb_team_acceptance_rejects_personal_stale_or_wrong_scope(self):
        payload = json.loads(self.wandb_acceptance.read_text())
        normalized = team_contract.validate_online_acceptance(payload)
        self.assertEqual(normalized["entity"], "nvidia")
        self.assertEqual(normalized["authorization_scope"], "team")
        self.assertFalse(normalized["personal"])

        mutations = (
            ("schema", "fast_roborender.wandb_online_smoke.v1"),
            ("entity", "ldu"),
            ("authorization_scope", "personal"),
            ("personal", True),
            ("team_member", False),
            ("entity_is_team", False),
            ("team_member", 1),
            ("entity_is_team", 1),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                changed = dict(payload)
                changed[key] = value
                with self.assertRaisesRegex(team_contract.TeamContractError, key):
                    team_contract.validate_online_acceptance(changed)

    def test_image3f_eval_uses_the_same_exact_team_binding(self):
        submit = (
            ROOT / "scripts/eval/submit_hsg_gb200_image3f_fixed_seed_gate.sh"
        ).read_text()
        packed = (
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_policy_eval.sbatch"
        ).read_text()
        gate_source = (
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py"
        ).read_text()
        upload_source = (
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_wandb.py"
        ).read_text()
        self.assertIn("wandb_online_smoke_nvidia_acceptance.json", submit)
        self.assertIn("WANDB_ENTITY=${WANDB_ENTITY:-nvidia}", submit)
        self.assertIn("WANDB_AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}", submit)
        self.assertIn('--authorization-scope "$WANDB_AUTHORIZATION_SCOPE"', packed)
        self.assertIn('--wandb-acceptance "$WANDB_ONLINE_VALIDATION"', packed)
        self.assertIn("hsg_image3f_fixed_seed_wandb_upload.v2", gate_source)
        self.assertIn("authorization_scope=team_contract.AUTHORIZED_SCOPE", upload_source)
        self.assertNotIn("require_personal", gate_source + upload_source)

    def test_wandb_readback_requires_every_checkpoint_and_gate_summary(self):
        rows = [
            {
                "eval/checkpoint_ordinal": ordinal,
                "eval/checkpoint_label": label,
                "eval/goal_reached_rate": 0.1 * ordinal,
                "eval/reward": float(ordinal),
                "eval/episodes": 32,
            }
            for ordinal, label in enumerate(fixed.ROLE_ORDER)
        ]
        remote_summary = {
            "policy_improvement_gate/status": "PASS",
            "policy_improvement_gate/paired_success_delta": 0.2,
            "policy_improvement_gate/paired_success_delta_ci95_low": 0.1,
            "policy_improvement_gate/paired_success_delta_ci95_high": 0.3,
        }
        report = wandb_upload.validate_readback(rows, remote_summary)
        self.assertEqual(report["observed_checkpoint_ordinals"], [0, 1, 2, 3])
        rows[-1].pop("eval/reward")
        with self.assertRaisesRegex(wandb_upload.UploadError, "readback incomplete"):
            wandb_upload.validate_readback(rows, remote_summary)


if __name__ == "__main__":
    unittest.main()
