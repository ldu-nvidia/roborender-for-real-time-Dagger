import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from scripts.eval import fixed_seed_gate as gate
from scripts.eval import eval_agent_metrics_adapter as metrics_adapter


class FixedSeedGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "train"
        self.run_dir.mkdir()
        self.baseline = self.root / "baseline.pt"
        self.baseline.write_bytes(b"baseline")
        self.config = self.root / "config.yaml"
        self.config.write_text("seed: 0\n", encoding="utf-8")
        for step in (105, 110, 115, 120, 125):
            (self.run_dir / f"model_step_{step:06d}.pt").write_bytes(str(step).encode())
        self.manifest_path = self.root / "eval" / "manifest.json"
        self.manifest = gate.create_manifest(
            train_run_dir=self.run_dir,
            baseline_checkpoint=self.baseline,
            reference_config=self.config,
            seeds=(11, 22, 33, 44, 55, 66, 77, 88),
            num_envs=32,
            output_path=self.manifest_path,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_selection_is_distinct_and_reproducible(self):
        roles = self.manifest["checkpoint_roles"]
        self.assertEqual([role["label"] for role in roles], list(gate.ROLE_ORDER))
        self.assertEqual([role["checkpoint_step"] for role in roles], [None, 105, 115, 125])
        self.assertEqual(len(self.manifest["tasks"]), 4)
        self.assertEqual(self.manifest["protocol"], "one_episode_per_environment_fixed_seed_blocks")
        self.assertEqual(self.manifest["expected_episodes_per_seed"], 32)

    def test_launch_time_source_hashes_fail_closed(self):
        with self.assertRaisesRegex(gate.GateError, "baseline checkpoint changed"):
            gate.create_manifest(
                train_run_dir=self.run_dir,
                baseline_checkpoint=self.baseline,
                reference_config=self.config,
                seeds=(1, 2),
                num_envs=2,
                output_path=self.root / "bad_baseline.json",
                expected_baseline_sha256="0" * 64,
            )
        with self.assertRaisesRegex(gate.GateError, "reference config changed"):
            gate.create_manifest(
                train_run_dir=self.run_dir,
                baseline_checkpoint=self.baseline,
                reference_config=self.config,
                seeds=(1, 2),
                num_envs=2,
                output_path=self.root / "bad_config.json",
                expected_baseline_sha256=gate.sha256_file(self.baseline),
                expected_reference_config_sha256="0" * 64,
            )

    def test_prepare_requires_three_training_snapshots(self):
        (self.run_dir / "model_step_000115.pt").unlink()
        (self.run_dir / "model_step_000120.pt").unlink()
        (self.run_dir / "model_step_000125.pt").unlink()
        with self.assertRaisesRegex(gate.GateError, "at least three"):
            gate.select_checkpoint_roles(self.baseline, self.run_dir)

    def test_result_rejects_wrong_episode_count_and_fractional_successes(self):
        with self.assertRaisesRegex(gate.GateError, "expected exactly 32"):
            gate.build_result(
                manifest=self.manifest,
                label="baseline",
                seed=11,
                metrics={
                    "collected_episodes": 31,
                    "goal_reached_rate": 0.5,
                    "reward": 1.0,
                    "episode_length": 2.0,
                },
            )
        with self.assertRaisesRegex(gate.GateError, "integer number of successes"):
            gate.build_result(
                manifest=self.manifest,
                label="baseline",
                seed=11,
                metrics={
                    "collected_episodes": 32,
                    "goal_reached_rate": 0.41,
                    "reward": 1.0,
                    "episode_length": 2.0,
                },
            )

    def test_sanitized_checkpoint_drops_environment_and_optimizer_state(self):
        source = self.root / "source.pt"
        output = self.root / "staged.pt"
        saved = {}
        fake_torch = types.SimpleNamespace()
        fake_torch.load = lambda *args, **kwargs: {
            "policy_state_dict": {"weight": 1},
            "env_state_dict": {"rng": 99},
            "optimizer_state_dict": {"momentum": 1},
            "lr_scheduler_state_dict": {"step": 10},
            "state": {"global_step": 10},
        }

        def fake_save(payload, path):
            saved.update(payload)
            Path(path).write_bytes(b"sanitized")

        fake_torch.save = fake_save
        source.write_bytes(b"source")
        previous = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            provenance = gate.sanitize_checkpoint(
                source, output, gate.sha256_file(source)
            )
        finally:
            if previous is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous
        self.assertTrue(output.is_file())
        self.assertIn("policy_state_dict", saved)
        self.assertNotIn("env_state_dict", saved)
        self.assertNotIn("optimizer_state_dict", saved)
        self.assertFalse(provenance["environment_state_restored"])

    def _write_results(self, success_counts, rewards):
        results = self.root / "eval" / "results"
        for label in gate.ROLE_ORDER:
            for seed in self.manifest["seeds"]:
                metrics = {
                    "collected_episodes": 32,
                    "goal_reached_rate": success_counts[label] / 32,
                    "reward": rewards[label],
                    "episode_length": 100.0,
                }
                row = gate.build_result(
                    manifest=self.manifest, label=label, seed=seed, metrics=metrics
                )
                path = results / label / f"seed_{seed}" / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(row), encoding="utf-8")
        return results

    def test_paired_gate_passes_only_with_supported_final_improvement(self):
        results = self._write_results(
            {"baseline": 12, "early": 14, "mid": 18, "final": 22},
            {"baseline": -120.0, "early": -115.0, "mid": -100.0, "final": -90.0},
        )
        summary = gate.summarize_results(
            manifest=self.manifest, results_dir=results, min_success_delta=0.02
        )
        self.assertEqual(summary["gate"]["status"], "PASS")
        self.assertGreater(
            summary["gate"]["paired_success_delta_seed_block_bootstrap95_low"], 0.0
        )
        self.assertEqual(summary["best_checkpoint_by_success"], "final")
        self.assertFalse(summary["protocol"]["training_loss_used_as_evidence"])

    def test_paired_gate_fails_when_ci_does_not_support_improvement(self):
        results = self._write_results(
            {"baseline": 16, "early": 18, "mid": 17, "final": 16},
            {"baseline": -100.0, "early": -90.0, "mid": -95.0, "final": -80.0},
        )
        summary = gate.summarize_results(
            manifest=self.manifest, results_dir=results, min_success_delta=0.02
        )
        self.assertEqual(summary["gate"]["status"], "FAIL")
        self.assertIn("SUCCESS_DELTA_BELOW_MINIMUM", summary["gate"]["reason_codes"])
        self.assertIn("SUCCESS_DELTA_CI_INCLUDES_ZERO", summary["gate"]["reason_codes"])
        self.assertTrue(summary["gate"]["reward_is_diagnostic_not_a_gate"])

    def test_missing_seed_block_is_incomplete(self):
        results = self._write_results(
            {"baseline": 12, "early": 14, "mid": 18, "final": 22},
            {"baseline": -120.0, "early": -115.0, "mid": -100.0, "final": -90.0},
        )
        (results / "mid" / "seed_44" / "result.json").unlink()
        with self.assertRaisesRegex(gate.GateError, "missing 1 fixed-seed result"):
            gate.summarize_results(
                manifest=self.manifest, results_dir=results, min_success_delta=0.02
            )


class EvalMetricsAdapterTest(unittest.TestCase):
    def test_checked_in_evaluator_matches_versioned_patch_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "GR00T-VisualSim2Real/gr00t/rl/eval_agent_trl.py"
        ).read_text(encoding="utf-8")
        patched = metrics_adapter.patch_evaluator_source(source)
        self.assertEqual(patched.count("__roborender_record_eval_metrics"), 1)
        self.assertNotRegex(
            patched,
            metrics_adapter.EVAL_CALL_RE,
        )

    def test_adapter_executes_original_program_and_atomically_records_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = root / "gr00t" / "rl" / "eval_agent_trl.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text(
                '''# @hydra.main
# trainer = custom_instantiate(
import sys
class Logger:
    def info(self, message): pass
logger = Logger()
class Trainer:
    def eval(self):
        return {"collected_episodes": 2, "goal_reached_rate": 0.5,
                "reward": 1.25, "episode_length": 4, "argv": sys.argv[1:]}
def main():
    trainer = Trainer()
    trainer.eval()
    logger.info("Finished evaluation")
if __name__ == "__main__":
    main()
''',
                encoding="utf-8",
            )
            output = root / "result" / "metrics.json"
            metrics_adapter.execute_evaluator(evaluator, output, ["+seed=17"])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["collected_episodes"], 2)
            self.assertEqual(payload["argv"], ["+seed=17"])
            self.assertEqual(
                payload["_roborender_eval_adapter"]["schema_version"],
                metrics_adapter.ADAPTER_SCHEMA_VERSION,
            )

    def test_adapter_fails_closed_on_source_drift_or_missing_metrics(self):
        with self.assertRaisesRegex(metrics_adapter.AdapterError, "contract marker"):
            metrics_adapter.patch_evaluator_source("trainer.eval()\n")

        contract = '''# @hydra.main\n# trainer = custom_instantiate(\ndef main():\n    trainer.eval()\n    trainer.eval()\n    logger.info("Finished evaluation")\nif __name__ == "__main__":\n    main()\n'''
        with self.assertRaisesRegex(metrics_adapter.AdapterError, "found 2"):
            metrics_adapter.patch_evaluator_source(contract)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metrics.json"
            with self.assertRaisesRegex(metrics_adapter.AdapterError, "metrics mapping"):
                metrics_adapter.write_metrics_atomic(
                    None,
                    output,
                    evaluator_path=Path("eval.py"),
                    evaluator_sha256="0" * 64,
                )
            self.assertFalse(output.exists())


class FixedSeedSlurmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.eval_script = (cls.root / "scripts/eval/fixed_seed_policy_eval.sbatch").read_text()
        cls.orchestrator = (
            cls.root / "scripts/eval/orchestrate_fixed_seed_gate.sbatch"
        ).read_text()
        cls.report = (cls.root / "scripts/eval/fixed_seed_report.sbatch").read_text()

    def test_eval_is_fixed_seed_one_episode_per_environment(self):
        self.assertIn("+algo.config.eval.eval_num_envs_episodes=true", self.eval_script)
        self.assertIn('+seed="$SEED"', self.eval_script)
        self.assertIn("WANDB_MODE=disabled", self.eval_script)
        self.assertIn("metrics_eval.json", self.eval_script)
        self.assertIn("eval_agent_metrics_adapter.py", self.eval_script)
        self.assertIn('--metrics-output "$METRICS_PENDING"', self.eval_script)
        self.assertIn("sanitize-checkpoint", self.eval_script)
        self.assertIn("--evaluated-checkpoint-sha256", self.eval_script)
        self.assertIn("--evaluated-config-sha256", self.eval_script)
        self.assertNotIn('if [ ! -e "$STAGED_CKPT" ]', self.eval_script)

    def test_report_runs_after_any_array_outcome_and_fails_closed(self):
        self.assertIn('--dependency="afterany:$EVAL_JOB_ID"', self.orchestrator)
        self.assertIn("wandb_preflight.py", self.report)
        self.assertIn("POLICY_IMPROVEMENT_GATE_FAIL", self.report)
        self.assertIn("ldu@nvidia.com", self.report)
        self.assertIn("--expected-baseline-sha256", self.orchestrator)
        self.assertIn("--expected-reference-config-sha256", self.orchestrator)


if __name__ == "__main__":
    unittest.main()
