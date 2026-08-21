import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.scaling.resume_continuity_gate import GateError, verify


ROOT = Path(__file__).resolve().parents[1]


def _snapshot(step: int, policy: str, *, max_telemetry: int) -> dict:
    telemetry = {
        str(value): f"digest-{value}"
        for value in range(0, max_telemetry + 1, 10)
    }
    uploads = {
        f"telemetry/scalars/step_{value:09d}.json": f"fingerprint-{value}"
        for value in range(0, max_telemetry + 1, 10)
    }
    return {
        "schema": "fast_roborender.resume_continuity.snapshot.v1",
        "checkpoint": f"/run/model_step_{step:06d}.pt",
        "checkpoint_sha256": f"checkpoint-{step}",
        "checkpoint_step": step,
        "absolute_target": 2,
        "policy_sha256": policy,
        "optimizer_state_present": True,
        "optimizer_state_entry_count": 1,
        "optimizer_param_group_count": 1,
        "scheduler_state_present": True,
        "run_dir": "/run",
        "trainer_wandb_run_id": "trainer-stable",
        "sidecar_wandb_run_id": "sidecar-stable",
        "wandb_required": True,
        "sidecar_upload_artifacts": uploads,
        "telemetry_scalars": telemetry,
        "runtime_ranks": [
            {
                "rank": rank,
                "sha256": f"runtime-{rank}-{step}",
                "telemetry_step": step * 300,
                "rollout_steps": 300,
                "local_num_envs": 32,
            }
            for rank in range(2)
        ],
    }


class ResumeContinuityGateTest(unittest.TestCase):
    def test_accepts_new_process_step_and_telemetry_continuity(self):
        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=590)
        report = verify(
            before,
            after,
            expected_before_step=1,
            expected_after_step=2,
            rollout_steps=300,
            expected_world_size=2,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["steps_advanced"], 1)
        self.assertEqual(report["new_telemetry_steps"][-1], 590)

    def test_rejects_restarted_telemetry_overwrite(self):
        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=290)
        after["telemetry_scalars"]["0"] = "overwritten"
        with self.assertRaisesRegex(GateError, "was overwritten"):
            verify(
                before,
                after,
                expected_before_step=1,
                expected_after_step=2,
                rollout_steps=300,
                expected_world_size=2,
            )

    def test_mechanics_gate_can_run_without_wandb(self):
        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=590)
        for value in (before, after):
            value["wandb_required"] = False
            value["trainer_wandb_run_id"] = None
            value["sidecar_wandb_run_id"] = None
            value["sidecar_upload_artifacts"] = {}
        report = verify(
            before,
            after,
            expected_before_step=1,
            expected_after_step=2,
            rollout_steps=300,
            expected_world_size=2,
            require_wandb=False,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["wandb_required"])

    def test_rejects_new_wandb_run_or_missing_rank_state(self):
        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=590)
        after["trainer_wandb_run_id"] = "different"
        with self.assertRaisesRegex(GateError, "trainer_wandb_run_id changed"):
            verify(
                before,
                after,
                expected_before_step=1,
                expected_after_step=2,
                rollout_steps=300,
                expected_world_size=2,
            )

        after = _snapshot(2, "policy-after", max_telemetry=590)
        after["runtime_ranks"].pop()
        with self.assertRaisesRegex(GateError, "incomplete per-rank"):
            verify(
                before,
                after,
                expected_before_step=1,
                expected_after_step=2,
                rollout_steps=300,
                expected_world_size=2,
            )

    def test_rejects_non_integer_or_inconsistent_rank_runtime_fields(self):
        mutations = (
            ("rank", True),
            ("telemetry_step", "600"),
            ("rollout_steps", 0),
            ("local_num_envs", 0),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                before = _snapshot(1, "policy-before", max_telemetry=290)
                after = _snapshot(2, "policy-after", max_telemetry=590)
                after["runtime_ranks"][0][key] = value
                with self.assertRaises(GateError):
                    verify(
                        before,
                        after,
                        expected_before_step=1,
                        expected_after_step=2,
                        rollout_steps=300,
                        expected_world_size=2,
                    )

        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=590)
        after["runtime_ranks"][1]["local_num_envs"] = 31
        with self.assertRaisesRegex(GateError, "local environment counts differ"):
            verify(
                before,
                after,
                expected_before_step=1,
                expected_after_step=2,
                rollout_steps=300,
                expected_world_size=2,
            )

    def test_rejects_empty_post_update_optimizer_state(self):
        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=590)
        after["optimizer_state_entry_count"] = 0
        with self.assertRaisesRegex(GateError, "empty after step zero"):
            verify(
                before,
                after,
                expected_before_step=1,
                expected_after_step=2,
                rollout_steps=300,
                expected_world_size=2,
            )

    def test_cli_verifier_writes_a_pass_report(self):
        before = _snapshot(1, "policy-before", max_telemetry=290)
        after = _snapshot(2, "policy-after", max_telemetry=590)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before_path = root / "before.json"
            after_path = root / "after.json"
            output = root / "report.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            from scripts.scaling import resume_continuity_gate as gate

            code = gate.main(
                [
                    "verify",
                    "--before",
                    str(before_path),
                    "--after",
                    str(after_path),
                    "--expected-before-step",
                    "1",
                    "--expected-after-step",
                    "2",
                    "--rollout-steps",
                    "300",
                    "--expected-world-size",
                    "2",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())["status"], "PASS")


class ProductionResumeSourceContractTest(unittest.TestCase):
    def test_trainer_uses_absolute_target_and_strict_state(self):
        path = ROOT / (
            "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer.py"
        )
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        constructor = source[source.index("    def __init__(") : source.index("    def _init_trl(")]
        self.assertLess(
            constructor.index("self._checkpoint_loaded = False"),
            constructor.index("self._setup_storage()"),
        )
        self.assertIn(
            "range(completed_step + 1, target_step + 1)", source
        )
        self.assertIn("strict resume checkpoint has malformed optimizer state", source)
        self.assertIn("strict resume optimizer has no state entries after step zero", source)
        self.assertIn("self._checkpoint_loaded = strict_resume", source)
        self.assertIn("Training stopped cleanly after absolute step", source)
        self.assertNotIn(
            "if self.control.should_training_stop:\n            return", source
        )
        callback = (
            ROOT
            / "GR00T-VisualSim2Real/gr00t/rl/trl/callbacks/model_save_callback.py"
        ).read_text(encoding="utf-8")
        ast.parse(callback)
        self.assertIn('"checkpoint_schema": "gr00t.trl.full_state.v2"', callback)
        self.assertIn("os.replace(temporary_name, destination)", callback)
        self.assertIn("test_stop_after_checkpoint_step", callback)
        self.assertIn("control.should_training_stop = True", callback)

    def test_image_dagger_persists_per_rank_runtime_offset(self):
        path = ROOT / (
            "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/"
            "distill_trainer_videogen.py"
        )
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("rank_{self._id_rank:04d}_step_", source)
        self.assertIn("self._id_rng.get_state()", source)
        self.assertIn("self._id_rng.set_state", source)
        self.assertIn("episode_ids + 1", source)
        self.assertIn("telemetry_step", source)
        self.assertIn("EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S", source)
        self.assertIn("require_expected_sha256=production_resume", source)

    def test_wandb_and_launcher_resume_fail_closed(self):
        train_agent = (
            ROOT / "GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py"
        ).read_text(encoding="utf-8")
        ast.parse(train_agent)
        self.assertIn('resume="must" if resume_mode else "allow"', train_agent)
        self.assertIn("W&B resume requires the original experiment metadata", train_agent)
        launcher = (ROOT / "scripts/train/dagger_scale_256x300.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("strict resume must reuse the checkpoint experiment directory", launcher)
        self.assertIn('headless=true "$CKPT_ARG" "$RESUME_ARG"', launcher)
        self.assertIn("CHECKPOINT_MECHANICS_SMOKE=1", launcher)
        self.assertIn("TEST_STOP_AFTER_CHECKPOINT_STEP", launcher)
        self.assertIn('use_wandb="$TRAINER_USE_WANDB"', launcher)
        self.assertIn("production_256x300 requires online W&B telemetry", launcher)


if __name__ == "__main__":
    unittest.main()
