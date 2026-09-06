from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from round_dagger import artifacts
from round_dagger.native_h200_validation import (
    CANONICAL_TEST_ROOT,
    CANONICAL_VALIDATION_ROOT,
    CASE_COUNT,
    COHORT_COUNT,
    CollectionConfig,
    ENVIRONMENTS_PER_COHORT,
    ENVIRONMENTS_PER_RANK,
    GPU_COUNT,
    PAIRED_SEEDS,
    STEPS_PER_ENVIRONMENT,
    TRANSITION_COUNT,
    SAFE_EXECUTABLE_PATH,
    VALIDATION_PARTITION,
    _verify_file_inventory,
    build_case_assignments,
    load_validation_sources,
    rank_environment,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeH200ValidationTests(unittest.TestCase):
    def _config(self) -> CollectionConfig:
        physical_inventory = [
            {
                "rank": index,
                "physical_index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012x}",
                "pci_bus_id": f"00000000:{index:02X}:00.0",
                "name": "NVIDIA H200",
                "compute_capability": "9.0",
                "total_memory_mib": 143771,
                "mig_mode": "Disabled",
            }
            for index in range(8)
        ]
        config_sha256 = "6" * 64
        physical_inventory_sha256 = artifacts.canonical_json_sha256(
            physical_inventory
        )
        provenance = {
            "observation_source": "native",
            "isaac_runtime_sha256": "8" * 64,
            "actor_python_freeze_sha256": "9" * 64,
            "actor_source_inventory_sha256": "a" * 64,
            "gr00t_revision": "b" * 40,
            "gr00t_source_tree_sha256": "c" * 64,
            "gpu_name": "NVIDIA H200",
            "compute_capability": "9.0",
            "gpu_count": 8,
            "physical_gpu_inventory": physical_inventory,
            "physical_gpu_inventory_sha256": physical_inventory_sha256,
        }
        provenance["observation_runtime_sha256"] = artifacts.canonical_json_sha256(
            {
                "schema": "fast_roborender.round_dagger.native_h200_observation_runtime.v1",
                "config_sha256": config_sha256,
                **provenance,
            }
        )
        return CollectionConfig(
            repository_root=ROOT,
            gr00t_root=ROOT / "GR00T-VisualSim2Real",
            gr00t_python=Path("/runtime/isaac/python"),
            source_motion_root=ROOT
            / "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775",
            source_generation_id="gen-000-bc",
            macro_round=0,
            collection_id="gen-000-bc__validation-v001",
            policy_checkpoint=Path("/campaign/gen-000-bc/checkpoint.pt"),
            policy_sha256="1" * 64,
            policy_spec=Path("/campaign/gen-000-bc/checkpoint.pt.policy_spec.json"),
            policy_spec_sha256="2" * 64,
            runtime_policy_sha256="3" * 64,
            teacher_checkpoint=Path("/campaign/gen-000-bc/teacher.pt"),
            teacher_sha256="4" * 64,
            beta_sha256="5" * 64,
            config_path=Path("/repo/native.yaml"),
            config_sha256=config_sha256,
            observation_provenance=provenance,
            runtime_environment={
                "ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST": "/runtime/isaac-manifest.json",
                "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST": "/runtime/python-manifest.json",
                "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY": "/runtime/actor-manifest.json",
                "ROUND_DAGGER_GR00T_SOURCE_INVENTORY": "/runtime/gr00t-manifest.json",
                "ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT": "/runtime",
                "ROUND_DAGGER_ISAAC_RUNTIME_ROOT": "/runtime/runtime/isaacsim-5.1",
                "ROUND_DAGGER_PYTHON_RUNTIME_ROOT": "/runtime",
                "ROUND_DAGGER_CUDNN_LIBRARY_DIR": "/runtime/cudnn",
                "ROUND_DAGGER_ISAACLAB_SOURCE_ROOT": "/runtime/runtime/IsaacLab/source",
                "ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT": "/runtime/runtime/actor-python-site-packages",
                "ROUND_DAGGER_SYSTEM_LIBRARY_DIR": "/runtime/packages/root/usr/lib/x86_64-linux-gnu",
                "ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256": "e" * 64,
                "ROUND_DAGGER_ISAAC_SETUP_SCRIPT": "/runtime/setup.sh",
                "ROBORENDER_HEADLESS_EXPERIENCE": "/runtime/headless.kit",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "OMNI_KIT_ALLOW_ROOT": "1",
                "ACCEPT_EULA": "Y",
                "PRIVACY_CONSENT": "Y",
            },
            replay_action_dim=26,
        )

    @unittest.skipUnless(
        VALIDATION_PARTITION.is_file(),
        "sealed validation fixtures are intentionally absent until recollection",
    )
    def test_committed_registry_maps_exactly_64_cases_across_four_paired_seeds(self):
        sources = load_validation_sources()
        cases, cohorts = build_case_assignments(sources)
        self.assertEqual(len(cases), CASE_COUNT)
        self.assertEqual(len(cohorts), COHORT_COUNT)
        self.assertEqual({row["global_env_id"] for row in cases}, set(range(CASE_COUNT)))
        self.assertEqual(len({row["case_sha256"] for row in cases}), CASE_COUNT)
        for seed_index, seed in enumerate(PAIRED_SEEDS):
            selected = [row for row in cases if row["paired_seed"] == seed]
            self.assertEqual(len(selected), ENVIRONMENTS_PER_COHORT)
            self.assertEqual(
                sorted(Counter(row["motion_group_id"] for row in selected).values()),
                [12, 13, 13, 13, 13],
            )
            self.assertEqual(
                {row["physical_gpu"] for row in selected},
                {seed_index * 2, seed_index * 2 + 1},
            )
            for rank in cohorts[seed_index]["ranks"]:
                self.assertEqual(len(rank["global_env_ids"]), ENVIRONMENTS_PER_RANK)
                self.assertEqual(
                    rank["hydra_base_seed"] + rank["cohort_rank"], seed
                )
                self.assertEqual(rank["effective_simulator_seed"], seed)

    @unittest.skipUnless(
        VALIDATION_PARTITION.is_file(),
        "sealed validation fixtures are intentionally absent until recollection",
    )
    def test_all_eight_rank_environments_are_validation_only_and_unique(self):
        config = self._config()
        sources = load_validation_sources()
        _cases, cohorts = build_case_assignments(sources)
        assignment = {
            "cohorts": [
                {
                    **cohort,
                    "cohort_path": str(
                        config.collection_root
                        / "cohorts"
                        / f"seed-{cohort['cohort_index']:02d}-{cohort['paired_seed']}"
                    ),
                }
                for cohort in cohorts
            ],
            "validation_partition": {
                "group_ids": [row["group_id"] for row in sources.groups]
            },
        }
        environments = [
            rank_environment(
                config,
                assignment,
                physical_gpu=gpu,
                base_environment={
                    "ROUND_DAGGER_CAMPAIGN_ROOT": "/forbidden/train",
                    "ROUND_DAGGER_TEST_SNAPSHOT": "/forbidden/test",
                    "WORLD_SIZE": "8",
                    "BASH_ENV": "/forbidden/pre-mask.sh",
                    "PYTHONPATH": "/forbidden/imports",
                    "LD_PRELOAD": "/forbidden/preload.so",
                    "LD_LIBRARY_PATH": "/forbidden/libs",
                    "PATH": "/forbidden/bin",
                    "BASH_FUNC_mount%%": "() { :; }",
                    "SHELLOPTS": "xtrace",
                },
                assignment_file_sha256="d" * 64,
            )
            for gpu in range(GPU_COUNT)
        ]
        self.assertEqual(
            [environment["CUDA_VISIBLE_DEVICES"] for environment in environments],
            [
                f"GPU-00000000-0000-0000-0000-{index:012x}"
                for index in range(GPU_COUNT)
            ],
        )
        for environment in environments:
            self.assertEqual(environment["ROUND_DAGGER_ACTOR_ROLE"], "validation_collect")
            self.assertEqual(environment["ROUND_DAGGER_PURPOSE"], "validation")
            self.assertEqual(environment["ROUND_DAGGER_BETA"], "1.0")
            self.assertEqual(environment["ROUND_DAGGER_ACTOR_WORLD_SIZE"], "2")
            self.assertEqual(environment["ROUND_DAGGER_ENVS_PER_GPU"], "32")
            self.assertEqual(environment["ROUND_DAGGER_STEPS_PER_ENVIRONMENT"], "512")
            self.assertEqual(environment["ROUND_DAGGER_VALIDATION_ROOT"], str(CANONICAL_VALIDATION_ROOT))
            self.assertEqual(environment["ROUND_DAGGER_VALIDATION_TEST_ROOT"], str(CANONICAL_TEST_ROOT))
            self.assertEqual(
                json.loads(environment["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON"]),
                config.observation_provenance["physical_gpu_inventory"],
            )
            self.assertEqual(
                environment["ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256"],
                config.observation_provenance["physical_gpu_inventory_sha256"],
            )
            expected_gpu = json.loads(
                environment["ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON"]
            )
            self.assertEqual(expected_gpu["uuid"], environment["CUDA_VISIBLE_DEVICES"])
            self.assertEqual(
                environment["ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"],
                config.runtime_environment["ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"],
            )
            self.assertNotIn("ROUND_DAGGER_CAMPAIGN_ROOT", environment)
            self.assertNotIn("ROUND_DAGGER_TEST_SNAPSHOT", environment)
            self.assertNotIn("WORLD_SIZE", environment)
            self.assertNotIn("BASH_ENV", environment)
            self.assertNotIn("PYTHONPATH", environment)
            self.assertNotIn("LD_PRELOAD", environment)
            self.assertNotIn("LD_LIBRARY_PATH", environment)
            self.assertNotIn("BASH_FUNC_mount%%", environment)
            self.assertNotIn("SHELLOPTS", environment)
            self.assertEqual(environment["PATH"], SAFE_EXECUTABLE_PATH)
        coverage = [
            global_id
            for cohort in cohorts
            for rank in cohort["ranks"]
            for global_id in rank["global_env_ids"]
        ]
        self.assertEqual(sorted(coverage), list(range(CASE_COUNT)))
        self.assertEqual(TRANSITION_COUNT, CASE_COUNT * STEPS_PER_ENVIRONMENT)

    def test_full_native_inventory_bundle_is_admitted_by_cohort_contract(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as temporary:
            contract = artifacts.create_cohort(
                Path(temporary) / "cohort",
                campaign_id="roborender-h200-v001",
                macro_round=0,
                cohort_index=0,
                purpose="validation",
                expected_ranks=[0, 1],
                expected_global_env_ids=list(range(64)),
                expected_transitions=64 * 512,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [324, 192, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [26]},
                policy_sha256=config.policy_sha256,
                runtime_policy_sha256=config.runtime_policy_sha256,
                teacher_sha256=config.teacher_sha256,
                beta_sha256=config.beta_sha256,
                config_sha256=config.config_sha256,
                **config.observation_provenance,
            )
        self.assertEqual(contract["observation_source"], "native")
        self.assertEqual(
            contract["physical_gpu_inventory_sha256"],
            config.observation_provenance["physical_gpu_inventory_sha256"],
        )

    def test_runtime_inventory_binds_qualified_python_link_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            isaac_root = root / "isaac"
            python_root = root / "python"
            (isaac_root / "bin").mkdir(parents=True)
            (python_root / "bin").mkdir(parents=True)
            target = python_root / "bin/python3.11"
            target.write_bytes(b"qualified-python")
            target.chmod(0o500)
            link = isaac_root / "bin/python"
            link.symlink_to(target)
            link_payload = str(target).encode("utf-8")
            entries = sorted(
                [
                    {
                        "path": str(link),
                        "size_bytes": len(link_payload),
                        "sha256": hashlib.sha256(link_payload).hexdigest(),
                    },
                    {
                        "path": str(target),
                        "size_bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    },
                ],
                key=lambda row: row["path"],
            )
            manifest = root / "isaac.inventory.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.native_actor_file_inventory.v1",
                        "kind": "isaac_runtime",
                        "entries": entries,
                    }
                ),
                encoding="utf-8",
            )
            count = _verify_file_inventory(
                manifest,
                kind="isaac_runtime",
                allowed_roots=(isaac_root, python_root),
                required_paths=(link, target),
                required_symlink_paths=(link,),
            )
        self.assertEqual(count, 2)

    def test_rank_launcher_masks_test_source_and_training_replay(self):
        launcher = (
            ROOT / "scripts/round_dagger/run_native_h200_validation_rank.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("unshare --mount --propagation private", launcher)
        self.assertIn("--pid --fork --kill-child", launcher)
        self.assertIn("--mount-proc=/proc", launcher)
        self.assertIn("ROUND_DAGGER_PARENT_PID_NAMESPACE", launcher)
        self.assertIn("native-validation-workspace-jail", launcher)
        self.assertIn('targets+=("$ROOT/.git" "$GR00T_ROOT/.git")', launcher)
        self.assertIn('"nvidia-smi", "-i", expected["uuid"]', launcher)
        self.assertIn("-L $GR00T_PYTHON", launcher)
        self.assertIn("ROUND_DAGGER_ISAACLAB_SOURCE_ROOT", launcher)
        self.assertIn("ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT", launcher)
        self.assertIn("ROUND_DAGGER_SYSTEM_LIBRARY_DIR", launcher)
        self.assertIn("ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256", launcher)
        self.assertIn("NATIVE_VALIDATION_IMPORT_ROOTS_OK", launcher)
        self.assertIn('"rollout_gen" in entry', launcher)
        self.assertIn('[[ -z ${LD_LIBRARY_PATH:-}', launcher)
        self.assertIn("child PATH was not sanitized by the fleet", launcher)
        self.assertIn("ROUND_DAGGER_VALIDATION_TEST_ROOT", launcher)
        self.assertIn("ROUND_DAGGER_VALIDATION_SOURCE_MOTION_ROOT", launcher)
        self.assertIn("dagger_4/replay", launcher)
        self.assertNotIn("dagger_5/replay", launcher)
        self.assertIn("--bounding-set=-sys_admin,-dac_override,-dac_read_search", launcher)
        self.assertIn("algo.config.native_dagger.role=validation_collect", launcher)
        self.assertIn("algo.config.native_dagger.purpose=validation", launcher)
        self.assertIn("algo.config.native_dagger.beta_start=1.0", launcher)
        self.assertIn("layer_config.pretrained=false", launcher)
        self.assertIn('checkpoint="$ROUND_DAGGER_POLICY_CHECKPOINT"', launcher)
        self.assertIn(
            'export EXPECTED_TRAIN_CHECKPOINT_SHA256="$ROUND_DAGGER_POLICY_SHA256"',
            launcher,
        )
        self.assertIn("rank-preflight", launcher)
        self.assertIn("verify-runtime", launcher)
        self.assertIn("--critical-only", launcher)
        self.assertNotIn("algo.config.native_dagger.role=test", launcher)

    def test_rank_launcher_rejects_loader_state_before_gpu_preflight(self):
        launcher = ROOT / "scripts/round_dagger/run_native_h200_validation_rank.sh"
        environment = {
            "PATH": SAFE_EXECUTABLE_PATH,
            "LD_LIBRARY_PATH": "/tmp/unsealed",
        }
        result = subprocess.run(
            [str(launcher)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("inherited loader/shell injection state", result.stderr)

    def test_mse_evaluator_admits_replay_and_result_only_under_validation_root(self):
        evaluator = (
            ROOT / "scripts/round_dagger/evaluate_validation_mse.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CANONICAL_VALIDATION_ROOT", evaluator)
        self.assertIn(
            '"--artifact-root must be the exact sealed validation root"',
            evaluator,
        )
        self.assertIn("arguments.validation_artifact", evaluator)
        self.assertIn('CANONICAL_VALIDATION_ROOT,\n        "validation artifact"', evaluator)
        self.assertIn(
            "_rooted_path(path, CANONICAL_CAMPAIGN_ROOT, label)", evaluator
        )


if __name__ == "__main__":
    unittest.main()
