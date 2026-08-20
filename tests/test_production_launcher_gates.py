import math
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GR00T = ROOT / "GR00T-VisualSim2Real"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class ProductionLauncherGatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submit = read("scripts/scaling/submit_oci_packed.sh")
        cls.packed = read("scripts/scaling/oci_packed_256x300.sbatch")
        cls.trainer = read("scripts/train/dagger_scale_256x300.sbatch")
        cls.renderer = read("scripts/server/image3f_server_array.sbatch")
        cls.compat = read("scripts/scaling/submit_production.sh")
        cls.eight_hour = read("scripts/scaling/submit_8h_256env.sh")
        cls.smoke = read("scripts/scaling/submit_oci_smoke_32x300.sh")
        cls.calibration = read("scripts/scaling/submit_oci_calibration_3view.sh")
        cls.distributed_smoke = read("scripts/scaling/submit_oci_smoke_256x300.sh")
        cls.config = (
            GR00T
            / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger_3view.yaml"
        ).read_text(encoding="utf-8")
        cls.integrated_trainer = (
            GR00T / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
        ).read_text(encoding="utf-8")

    def test_only_packed_oci_entrypoint_is_reachable(self):
        for wrapper in (
            self.compat,
            self.eight_hour,
            self.smoke,
            self.calibration,
            self.distributed_smoke,
        ):
            self.assertIn("submit_oci_packed.sh", wrapper)
            self.assertNotIn("submit_production.sh\"", wrapper)
        self.assertNotIn("image3f_server_array.sbatch\")", self.submit)
        self.assertIn("oci_packed_256x300.sbatch", self.submit)

    def test_oci_scheduler_contract_is_explicit(self):
        for text in (self.packed, self.trainer, self.renderer):
            self.assertIn("#SBATCH --account=general_cs_infra", text)
        for text in (self.packed, self.trainer):
            self.assertIn("#SBATCH --partition=grizzly", text)
        self.assertIn("#SBATCH --partition=backfill_block1", self.renderer)
        for setting in (
            "#SBATCH --nodes=8",
            "#SBATCH --gpus-per-node=8",
            "#SBATCH --mem=768G",
            "#SBATCH --time=04:00:00",
            "OCI_CONTAINER_IMAGE=${OCI_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}",
        ):
            self.assertIn(setting, self.packed)

    def test_profile_scheduler_and_step_resources_are_immutable(self):
        tuples = (
            "grizzly:10:768G:8:512G:16:128G",
            "backfill_block1:10:768G:8:512G:16:128G",
            "backfill_block1:16:256G:64G:128G",
        )
        for contract in tuples:
            self.assertIn(contract, self.submit)
        for wrapper, expected in (
            (
                self.smoke,
                (
                    "PACKED_PARTITION=backfill_block1",
                    "PACKED_CPUS_PER_GPU=10",
                    "PACKED_MEM=768G",
                    "RENDERER_MEM_PER_NODE=512G",
                    "TRAINER_MEM_PER_NODE=128G",
                ),
            ),
            (
                self.calibration,
                (
                    "PACKED_PARTITION=backfill_block1",
                    "PACKED_CPUS_PER_GPU=16",
                    "PACKED_MEM=256G",
                    "RENDERER_MEM_PER_NODE=64G",
                    "TRAINER_MEM_PER_NODE=128G",
                ),
            ),
            (
                self.distributed_smoke,
                (
                    "PACKED_PARTITION=grizzly",
                    "PACKED_CPUS_PER_GPU=10",
                    "PACKED_MEM=768G",
                    "RENDERER_MEM_PER_NODE=512G",
                    "TRAINER_MEM_PER_NODE=128G",
                ),
            ),
        ):
            for contract in expected:
                self.assertIn(contract, wrapper)

    def test_packed_gpu_topology_matches_256_environment_plan(self):
        for contract in (
            "GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}",
            "ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}",
            "TRAINER_NODES=${TRAINER_NODES:-8}",
            "ENVS_PER_RENDERER=${ENVS_PER_RENDERER:-5}",
            '"8:8:256:300:8:5:52"',
            '[ "$REQUIRED_RENDERERS" -eq "$EXPECTED_RENDERERS" ]',
            "--ntasks-per-node=7",
            "trainer_tasks=$TRAINER_NODES",
        ):
            self.assertIn(contract, self.packed)

    def test_one_node_smoke_uses_the_same_packed_runtime(self):
        for contract in (
            "RUN_PROFILE=smoke_32x300",
            "PACKED_NODES=1",
            "PACKED_GPUS_PER_NODE=8",
            "GLOBAL_NUM_ENVS=32",
            "ROLLOUT_STEPS=300",
            "TRAINER_NODES=1",
            "EXPECTED_RENDERERS=7",
            "NUM_ROLLOUT_ITERATIONS=1",
            "SAVE_FREQUENCY=1",
            "submit_oci_packed.sh",
        ):
            self.assertIn(contract, self.smoke)
        self.assertIn("smoke_32x300", self.submit)
        self.assertIn("smoke_32x300", self.packed)
        self.assertIn('"1:8:32:300:1:5:7:1:1"', self.submit)
        self.assertIn('"1:8:32:300:1:5:7:1:1"', self.packed)
        self.assertIn('--nodes="$PACKED_NODES"', self.submit)
        self.assertIn('--nodes="$PACKED_NODES"', self.packed)
        for contract in (
            "PACKED_PARTITION=backfill_block1",
            "PACKED_CPUS_PER_GPU=10",
            "PACKED_MEM=768G",
            "RENDERER_CPUS_PER_TASK=8",
            "RENDERER_MEM_PER_NODE=512G",
            "TRAINER_CPUS_PER_TASK=16",
            "TRAINER_MEM_PER_NODE=128G",
        ):
            self.assertIn(f"export {contract}", self.smoke)
        self.assertIn(
            '"backfill_block1:10:768G:8:512G:16:128G"', self.submit
        )

    def test_camera_calibration_profile_is_small_and_dump_bounded(self):
        for contract in (
            "RUN_PROFILE=calibration_3view",
            "PACKED_NODES=1",
            "PACKED_GPUS_PER_NODE=2",
            "PACKED_PARTITION=backfill_block1",
            "PACKED_CPUS_PER_GPU=16",
            "PACKED_MEM=256G",
            "GLOBAL_NUM_ENVS=1",
            "ROLLOUT_STEPS=2",
            "TRAINER_NODES=1",
            "EXPECTED_RENDERERS=1",
            "NUM_ROLLOUT_ITERATIONS=1",
            "RENDERER_MEM_PER_NODE=64G",
            "TRAINER_MEM_PER_NODE=128G",
            "IMAGE_DAGGER_DUMP_SIM_RGB=1",
            "IMAGE_DAGGER_DUMP_MAX_STEPS=2",
            "IMAGE_DAGGER_DUMP_MAX_ENVS=1",
            "IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1",
        ):
            self.assertIn(contract, self.calibration)
        self.assertIn("calibration_3view", self.submit)
        self.assertIn("calibration_3view", self.packed)
        self.assertIn("PROFILE_TECHNICALLY_VALIDATED", self.trainer)
        self.assertIn("awaiting_camera_visual_acceptance", self.trainer)
        self.assertIn('"columns": ["head_color", "hand_left", "hand_right"]', self.trainer)
        self.assertIn(
            '"rows": ["depth_control", "simulator_rgb", "generated_rgb"]',
            self.trainer,
        )
        self.assertIn('"panel_shape_hwc": [720, 1248, 3]', self.trainer)
        self.assertIn("calibration panel contract mismatch", self.trainer)

    def test_human_accepted_camera_calibration_is_a_release_prerequisite(self):
        for text in (self.submit, self.packed):
            self.assertIn("CAMERA_CALIBRATION_ACCEPTANCE", text)
            self.assertIn("EXPECTED_CAMERA_CALIBRATION_ACCEPTANCE_SHA256", text)
            self.assertIn("camera_calibration_gate.py", text)
            self.assertIn("camera calibration acceptance validation failed", text)
        self.assertIn(
            "$STAGE_ROOT/runtime/validation/camera_calibration_acceptance.json",
            self.submit,
        )
        self.assertIn("camera_calibration_acceptance=", self.submit)
        self.assertIn("camera_calibration_acceptance=", self.packed)
        self.assertIn("camera_calibration_acceptance=", self.trainer)
        self.assertIn(
            '"status": "awaiting_visual_acceptance"', self.trainer
        )

    def test_distributed_smoke_enables_sync_gate_and_one_update(self):
        for contract in (
            "RUN_PROFILE=smoke_256x300",
            "PACKED_NODES=8",
            "PACKED_GPUS_PER_NODE=8",
            "GLOBAL_NUM_ENVS=256",
            "ROLLOUT_STEPS=300",
            "TRAINER_NODES=8",
            "EXPECTED_RENDERERS=52",
            "NUM_ROLLOUT_ITERATIONS=1",
            "SAVE_FREQUENCY=1",
            "IMAGE_DAGGER_VERIFY_DDP_SYNC=1",
            "IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1",
            "PROJ=roborender_dagger_3view_validation",
        ):
            self.assertIn(contract, self.distributed_smoke)
        self.assertIn("smoke_256x300", self.submit)
        self.assertIn("smoke_256x300", self.packed)

    def test_production_rechecks_ddp_sync_on_its_own_first_update(self):
        self.assertIn("IMAGE_DAGGER_VERIFY_DDP_SYNC=1", self.compat)
        self.assertIn("IMAGE_DAGGER_VERIFY_DDP_SYNC=1", self.eight_hour)
        self.assertIn('"0:0:0:1:1"', self.submit)
        self.assertIn('"0:0:0:1:1"', self.packed)

    def test_smoke_profiles_are_machine_enforced_release_predecessors(self):
        for text in (self.submit, self.packed):
            self.assertIn("PREREQUISITE_PROFILE_VALIDATION", text)
            self.assertIn("EXPECTED_PREREQUISITE_PROFILE_VALIDATION_SHA256", text)
            self.assertIn("PREREQUISITE_PROFILE_EXPECTED", text)
            self.assertIn("profile_validation_gate.py", text)
            self.assertIn("predecessor profile validation failed", text)
        self.assertIn(
            "$STAGE_ROOT/runtime/validation/smoke_32x300_profile_validation_manifest.txt",
            self.submit,
        )
        self.assertIn(
            "$STAGE_ROOT/runtime/validation/smoke_256x300_profile_validation_manifest.txt",
            self.submit,
        )
        self.assertIn("smoke_256x300:smoke_32x300", self.packed)
        self.assertIn("production_256x300:smoke_256x300", self.packed)
        self.assertIn("prerequisite_profile_validation=", self.submit)
        self.assertIn("prerequisite_profile_validation=", self.packed)
        self.assertIn("prerequisite_profile_validation=", self.trainer)

    def test_single_rank_profile_omits_distributed_accelerate_flags(self):
        self.assertIn('if [ "$TRAINER_NODES" -eq 1 ]; then', self.trainer)
        self.assertIn("ACCELERATE_ARGS=(", self.trainer)
        self.assertIn('exec accelerate launch "${ACCELERATE_ARGS[@]}"', self.trainer)
        single_rank_branch = self.trainer.split(
            'if [ "$TRAINER_NODES" -eq 1 ]; then', 1
        )[1].split("else", 1)[0]
        self.assertNotIn("--multi_gpu", single_rank_branch)
        self.assertNotIn("--num_machines", single_rank_branch)
        self.assertNotIn("--machine_rank", single_rank_branch)

    def test_student_initialization_is_fresh_and_checkpoint_free(self):
        for contract in (
            "CKPT=${CKPT:-none}",
            "EXPECTED_TRAIN_CHECKPOINT_SHA256=${EXPECTED_TRAIN_CHECKPOINT_SHA256:-NONE}",
            "initialization=fresh_student",
            "checkpoint=none",
        ):
            self.assertIn(contract, self.packed)
        self.assertIn("CKPT=none", self.submit)
        self.assertIn("EXPECTED_TRAIN_CHECKPOINT_SHA256=NONE", self.submit)
        self.assertIn("CKPT_ARG=checkpoint=null", self.trainer)

    def test_three_view_policy_contract_is_fail_closed(self):
        ordered = "head_color:hand_left:hand_right"
        for text in (self.submit, self.packed, self.trainer, self.eight_hour):
            self.assertIn(ordered, text)
            self.assertIn("VG_POLICY_VIEW_COUNT", text)
            self.assertIn("VG_POLICY_VIEW_LAYOUT", text)
            self.assertIn("VG_POLICY_IMAGE_SHAPE", text)
            self.assertIn("324x192x3", text)
        self.assertIn("+exp=loco_manip/wsdpt_image_dagger_3view", self.trainer)
        self.assertNotRegex(
            self.trainer,
            r"\+exp=loco_manip/wsdpt_image_dagger(?:\s|\\)",
        )
        for gate in (
            "IMAGE_DAGGER_EXPECT_POLICY_VIEWS",
            "IMAGE_DAGGER_EXPECT_POLICY_VIEW_COUNT",
            "IMAGE_DAGGER_EXPECT_POLICY_VIEW_LAYOUT",
            "IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE",
        ):
            self.assertIn(gate, self.trainer)
        self.assertIn(
            'IMAGE_DAGGER_EXPECT_POLICY_IMAGE_SHAPE="${VG_POLICY_IMAGE_SHAPE//x/,}"',
            self.trainer,
        )
        self.assertIn(
            "IMAGE_DAGGER_DUMP_SIM_RGB=${IMAGE_DAGGER_DUMP_SIM_RGB:-0}",
            self.trainer,
        )
        self.assertIn("IMAGE_DAGGER_DUMP_SIM_RGB=1", self.calibration)
        self.assertIn("IMAGE_DAGGER_DUMP_MAX_STEPS=2", self.calibration)
        self.assertIn("IMAGE_DAGGER_DUMP_MAX_ENVS=1", self.calibration)
        for manifest in (
            "policy_views=",
            "policy_view_count=",
            "policy_view_layout=",
            "policy_image_shape=",
        ):
            self.assertIn(manifest, self.submit)
            self.assertIn(manifest, self.packed)
            self.assertIn(manifest, self.trainer)
        self.assertRegex(
            self.config,
            r"policy_views:\s*\[head_color, hand_left, hand_right\]",
        )
        self.assertRegex(self.config, r"policy_view_count:\s*3")
        self.assertIn("image_dagger_policy_view_count: ${policy_view_count}", self.config)
        self.assertIn(
            '++env.config.image_dagger_policy_view_count="$VG_POLICY_VIEW_COUNT"',
            self.trainer,
        )

    def test_absent_beta_fields_use_hydra_append_override(self):
        for field in ("beta_schedule", "beta_start", "beta_end", "beta_decay_iters"):
            self.assertIn(f"++algo.config.videogen.{field}=", self.trainer)
            self.assertNotRegex(
                self.trainer,
                rf"(?<!\+)algo\.config\.videogen\.{field}=",
            )

    def test_full_production_hydra_overrides_compose(self):
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("hydra-core/omegaconf unavailable in this test environment")

        sys.path.insert(0, str(GR00T))
        try:
            # Mirror register_rl_resolvers() without importing the training
            # stack: config composition itself should not require torch,
            # Isaac, or even the optional loguru package.
            resolvers = {
                "eval": eval,
                "if": lambda pred, a, b: a if pred else b,
                "eq": lambda x, y: x.lower() == y.lower(),
                "sqrt": lambda x: math.sqrt(float(x)),
                "sum": lambda x: sum(x),
                "ceil": lambda x: math.ceil(x),
                "int": lambda x: int(x),
                "len": lambda x: len(x),
                "sum_list": lambda values: sum(values),
            }
            for name, resolver in resolvers.items():
                if not OmegaConf.has_resolver(name):
                    OmegaConf.register_new_resolver(name, resolver)
            overrides = [
                "+exp=loco_manip/wsdpt_image_dagger_3view",
                "num_envs=32",
                "algo.config.num_steps_per_env=300",
                "algo.config.num_mini_batches=1",
                "algo.config.num_learning_epochs=1",
                "algo.config.rollout_with_teacher_num_steps=0",
                "algo.config.enforce_teacher_rollout=true",
                "algo.config.ratio_teacher_rollout=0.0",
                "algo.config.value_loss_coef=0.0",
                "algo.config.entropy_coef=0.0",
                "algo.config.videogen.rollout_action_source=beta",
                "++algo.config.videogen.beta_schedule=linear",
                "++algo.config.videogen.beta_start=1.0",
                "++algo.config.videogen.beta_end=0.0",
                "++algo.config.videogen.beta_decay_iters=500",
                "algo.trl.per_device_train_batch_size=1",
                "algo.trl.gradient_accumulation_steps=32",
                "algo.trl.num_total_batches=1",
                "algo.trl.total_episodes=null",
                "callbacks.model_save.save_frequency=5",
                "callbacks.model_save.save_initial_checkpoint=true",
                "headless=true",
                "checkpoint=null",
                "teacher_actor_path=/tmp/expert.pt",
                "algo.config.videogen.raytraced_mix_prob=0.0",
                "++algo.config.videogen.seed=42",
                "++algo.config.videogen.reference_lag=0",
                "++algo.config.videogen.primary_view=head_color",
                "simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]",
                "policy_view_count=3",
                "++env.config.image_dagger_policy_view_count=3",
                "algo.config.videogen.image3f_endpoint_files=[/tmp/endpoint.json]",
                "algo.config.videogen.anchor_pool_dir=/tmp/anchors",
                "algo.config.videogen.anchor_index=0",
                "simulator.config.render_results=false",
                "use_wandb=true",
                "wandb.wandb_entity=ldu",
                "wandb.wandb_group=null",
                "project_name=roborender_dagger_base_scale_4h",
                "experiment_name=composition_regression",
                "experiment_dir=/tmp/roborender-composition",
                "hydra.run.dir=/tmp/roborender-composition/.hydra/rank_0",
                "output_dir=/tmp/roborender-composition/output/rank_0",
            ]
            with initialize_config_dir(
                config_dir=str(GR00T / "gr00t/rl/config"),
                version_base="1.1",
            ):
                cfg = compose(config_name="base", overrides=overrides)
        finally:
            sys.path.remove(str(GR00T))

        self.assertEqual(
            list(cfg.simulator.config.cameras.policy_views),
            ["head_color", "hand_left", "hand_right"],
        )
        self.assertEqual(cfg.policy_view_count, 3)
        self.assertEqual(cfg.env.config.image_dagger_policy_view_count, 3)
        self.assertTrue(cfg.callbacks.model_save.save_initial_checkpoint)
        self.assertEqual(cfg.algo.config.videogen.rollout_action_source, "beta")
        self.assertEqual(cfg.algo.config.videogen.beta_schedule, "linear")
        self.assertEqual(cfg.algo.config.videogen.beta_decay_iters, 500)

    def test_initial_and_post_update_checkpoints_are_release_gates(self):
        self.assertIn("callbacks.model_save.save_initial_checkpoint=true", self.trainer)
        for text in (self.submit, self.packed, self.trainer):
            self.assertIn("initial_policy_checkpoint=", text)
        self.assertIn('model_step_000001.pt', self.trainer)
        self.assertIn("profile_validation_manifest.txt", self.trainer)
        self.assertIn("PROFILE_VALIDATED", self.trainer)
        self.assertIn("ddp_policy_sync_step_000000001.json", self.trainer)
        self.assertIn('"world_size": 8', self.trainer)
        self.assertIn("initial_policy_sha256=", self.trainer)
        self.assertIn("final_checkpoint_sha256=", self.trainer)
        self.assertIn("EXPECTED_FINAL_CHECKPOINT", self.trainer)
        self.assertIn("NUM_ROLLOUT_ITERATIONS % SAVE_FREQUENCY", self.trainer)
        self.assertIn("successful trainer did not create exact final checkpoint", self.trainer)

    def test_all_profiles_require_finite_nonnegative_bc_loss(self):
        for wrapper in (
            self.compat,
            self.eight_hour,
            self.smoke,
            self.calibration,
            self.distributed_smoke,
        ):
            self.assertIn("IMAGE_DAGGER_REQUIRE_FINITE_BC_LOSS=1", wrapper)
        for contract in (
            "dagger_bc_loss_step_",
            '"schema": "fast_roborender.dagger_bc_loss.v1"',
            '"finite_verified": True',
            '"nonnegative_verified": True',
            "finite_bc_loss_marker_sha256=",
        ):
            self.assertIn(contract, self.trainer)

    def test_non_calibration_profiles_forbid_camera_dumps(self):
        for wrapper in (
            self.compat,
            self.eight_hour,
            self.smoke,
            self.distributed_smoke,
        ):
            self.assertIn("IMAGE_DAGGER_DUMP_SIM_RGB=0", wrapper)
            self.assertIn("IMAGE_DAGGER_DUMP_MAX_STEPS=0", wrapper)
            self.assertIn("IMAGE_DAGGER_DUMP_MAX_ENVS=0", wrapper)

    def test_hydra_contract_is_built_and_checked_before_submission(self):
        setup = read("scripts/scaling/oci_h100_trainer_setup.sbatch")
        preflight = read("scripts/setup/check_env.sh")
        validator = read("scripts/scaling/validate_production_hydra.py")
        self.assertIn("validate_production_hydra.py", setup)
        self.assertIn("production_hydra_contract_sha256=", setup)
        self.assertIn("production_hydra_contract", preflight)
        self.assertIn("fast_roborender.production_hydra_contract.v1", preflight)
        self.assertIn("+exp=loco_manip/wsdpt_image_dagger_3view", validator)
        self.assertIn('"fresh_student": cfg.checkpoint is None', validator)

    def test_expert_labeler_is_a_pinned_cluster_asset(self):
        expected_hash = "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38"
        for text in (self.submit, self.packed, self.trainer):
            self.assertIn("assets/policies/expert_labeler/model_step_044500.pt", text)
            self.assertIn(expected_hash, text)
            self.assertIn("sha256sum", text)
        self.assertIn('teacher_actor_path="$TEACHER_CKPT"', self.trainer)

    def test_renderer_and_sampler_provenance_fail_closed(self):
        for contract in (
            "EXPECTED_IMAGE3F_RELEASE_ID",
            "EXPECTED_IMAGE3F_ADAPTER_SHA256",
            "EXPECTED_IMAGE3F_INFERENCE_STEPS",
            "EXPECTED_IMAGE3F_TEACACHE_THRESHOLD",
            "EXPECTED_IMAGE3F_CFG_SCALE",
            "EXPECTED_IMAGE3F_CUDA_GRAPH",
            "--require-tensorrt-engine",
            "engine_build_manifest.json",
            "--expected-output-views",
            "--expected-output-layout",
            "--expected-output-tile-shape",
            "--expected-output-stacked-shape",
        ):
            self.assertIn(contract, self.submit + self.packed + self.trainer)

    def test_foundation_root_is_not_double_nested(self):
        sources = self.submit + self.packed + self.renderer + read(
            "scripts/scaling/oci_h100_image3f_setup.sbatch"
        )
        self.assertGreaterEqual(
            sources.count("ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models}"),
            4,
        )
        self.assertNotIn(
            "ROBORENDER_MODEL_BASE=${ROBORENDER_MODEL_BASE:-$STAGE_ROOT/assets/models/PAI/",
            sources,
        )

    def test_target_environments_match_oci_driver_and_gitless_container(self):
        image_setup = read("scripts/scaling/oci_h100_image3f_setup.sbatch")
        trainer_setup = read("scripts/scaling/oci_h100_trainer_setup.sbatch")
        wandb_setup = read("scripts/scaling/oci_host_wandb_setup.sbatch")
        for contract in (
            "runtime/envs/image3f-cu128",
            "https://download.pytorch.org/whl/cu128",
            "tensorrt-cu12-libs",
            "torch.cuda.is_available()",
            "expected an H100 (sm90)",
        ):
            self.assertIn(contract, image_setup)
        self.assertIn("codeload.github.com/isaac-sim/IsaacLab/tar.gz", trainer_setup)
        self.assertIn("#SBATCH --mem=224G", image_setup)
        self.assertIn("#SBATCH --mem=224G", trainer_setup)
        self.assertIn(".oci_source_revision", trainer_setup)
        self.assertNotIn("git clone", trainer_setup)
        self.assertNotIn("git -C", trainer_setup)
        self.assertIn("EXPECTED_GR00T_REVISION", trainer_setup)
        self.assertIn("runtime/envs/wandb-host", wandb_setup)
        self.assertIn("HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}", wandb_setup)
        self.assertIn("NUMPY_VERSION=${NUMPY_VERSION:-1.24.4}", wandb_setup)
        self.assertIn("#SBATCH --partition=cpu_short", wandb_setup)
        self.assertNotIn("#SBATCH --gpus-per-node=", wandb_setup)
        self.assertNotIn("#SBATCH --container-image=", wandb_setup)

    def test_host_coordinator_never_executes_container_python(self):
        self.assertIn("HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}", self.packed)
        self.assertIn("runtime/envs/wandb-host", self.submit + self.packed + self.trainer)
        self.assertNotIn('"$VIRAL_PYTHON" - "$ANCHORS"', self.packed)
        self.assertNotIn(
            '"$VIRAL_PYTHON" "$ROOT/scripts/scaling/scale_plan.py"',
            self.packed,
        )

    def test_direct_sbatch_jobs_do_not_depend_on_spool_location(self):
        for base in (ROOT / "scripts", ROOT / "tools"):
            for path in base.rglob("*.sbatch"):
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertNotIn("BASH_SOURCE", path.read_text(encoding="utf-8"))

    def test_standalone_container_python_jobs_reexec_through_srun(self):
        jobs = (
            "scripts/anchors/anchor_pool.sbatch",
            "scripts/anchors/teacher_anchor_render.sbatch",
            "scripts/data/collect_data.sbatch",
            "scripts/eval/eval_latest.sbatch",
            "scripts/eval/eval_policy.sbatch",
            "scripts/eval/eval_student.sbatch",
            "scripts/eval/fixed_seed_policy_eval.sbatch",
            "scripts/scaling/oci_h100_container_probe.sbatch",
            "scripts/scaling/oci_h100_image3f_setup.sbatch",
            "scripts/scaling/oci_h100_trainer_setup.sbatch",
            "scripts/server/image3f_ft_eval.sbatch",
            "scripts/server/image3f_ft_setup.sbatch",
            "scripts/server/image3f_server.sbatch",
            "scripts/server/image3f_server_array.sbatch",
            "scripts/server/raven_server.sbatch",
            "scripts/train/dagger_train.sbatch",
            "scripts/train/dagger_train_3view.sbatch",
            "scripts/train/image_dagger_r4.sbatch",
            "scripts/train/image_dagger_run.sbatch",
            "scripts/train/image_dagger_smoke.sbatch",
            "scripts/train/offline_bc_train.sbatch",
            "tools/student3v_bench.sbatch",
        )
        for job in jobs:
            text = read(job)
            with self.subTest(job=job):
                self.assertIn("reexec_in_oci_container.sh", text)
                self.assertNotIn("#SBATCH --container-image=", text)
                self.assertNotIn("#SBATCH --container-mounts=", text)

    def test_oci_container_interface_is_srun_only(self):
        helper = read("scripts/setup/reexec_in_oci_container.sh")
        self.assertIn('exec srun --nodes=1 --ntasks=1', helper)
        self.assertIn('--container-image="$OCI_CONTAINER_IMAGE"', helper)
        self.assertIn('--container-mounts="$ROBORENDER_STAGE_ROOT:$ROBORENDER_STAGE_ROOT"', helper)
        for base in (ROOT / "scripts", ROOT / "tools"):
            for path in base.rglob("*.sbatch"):
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertNotIn("#SBATCH --container-image=", text)
                    self.assertNotIn("#SBATCH --container-mounts=", text)

    def test_optional_legacy_renderer_uses_staged_asset_contract(self):
        sources = read("scripts/setup/oci_env.sh") + read("run_raven_server.sh")
        self.assertIn("ROBORENDER_ACCEL_ROOT", sources)
        self.assertIn("$ROBORENDER_STAGE_ROOT/assets/roborender-acceleration", sources)
        self.assertNotIn("$ROOT/roborender-acceleration", sources)

    def test_wandb_identity_and_media_are_fail_fast(self):
        for contract in (
            "WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}",
            "WANDB_MODE=${WANDB_MODE:-online}",
            '--expected-email "$WANDB_EXPECTED_EMAIL"',
            "IMAGE_DAGGER_TELEMETRY_REQUIRED=1",
            "IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1",
            "WANDB_UPLOAD_STOP",
        ):
            self.assertIn(contract, self.submit + self.packed + self.trainer)
        self.assertIn("owner-only permissions", self.submit)
        self.assertIn('stat -c %u "$NETRC"', self.submit)

    def test_submission_uses_an_explicit_environment_allowlist(self):
        self.assertIn('EXPORTS="STAGE_ROOT=', self.submit)
        self.assertNotIn('EXPORTS="ALL,', self.submit)
        self.assertNotIn("--export=ALL", self.submit)
        self.assertIn('EXPORTS+=",WANDB_API_KEY"', self.submit)
        self.assertIn("export SLURM_EXPORT_ENV=ALL", self.packed)

    def test_packed_step_owns_cleanup_and_pre_timeout_forwarding(self):
        for contract in (
            "trap cleanup EXIT TERM INT",
            "trap forward_pre_timeout USR1",
            'kill -TERM "$TRAIN_PID"',
            'kill -TERM "$RENDER_STEP_PID"',
            'kill -USR1 "$TRAIN_PID"',
            "--signal=B:USR1@300",
        ):
            self.assertIn(contract, self.packed + self.submit)

    def test_gr00t_config_has_no_site_fallbacks(self):
        self.assertRegex(self.config, r"image3f_endpoint_files:\s*\[\]")
        self.assertRegex(self.config, r"anchor_pool_dir:\s*null")
        self.assertRegex(self.config, r"dump_dir:\s*null")
        self.assertIn("teacher_actor_path: ???", read(
            "GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/"
            "wsdpt_student_for_teacher_v8q8.002_resnet_rgb_delay.yaml"
        ))

    def test_gr00t_trainer_requires_renderer_inputs(self):
        for message in (
            "algo.config.videogen.image3f_endpoint_files is required",
            "algo.config.videogen.anchor_pool_dir is required",
            "no Image3F anchor .npz files",
        ):
            self.assertIn(message, self.integrated_trainer)

    def test_all_active_batch_headers_target_authorized_oci_profile(self):
        for base in (ROOT / "scripts", ROOT / "tools"):
            for path in base.rglob("*.sbatch"):
                text = path.read_text(encoding="utf-8")
                accounts = re.findall(r"^#SBATCH --account=(\S+)", text, re.MULTILINE)
                partitions = re.findall(r"^#SBATCH --partition=(\S+)", text, re.MULTILINE)
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertTrue(accounts, "missing explicit OCI account")
                    self.assertTrue(partitions, "missing explicit OCI partition")
                    if path.name.startswith("hsg_gb200_"):
                        expected_account = "llmservice_nemotron_ultra"
                        expected_partition = "batch"
                    elif path.name in {
                        "oci_packed_256x300.sbatch",
                        "dagger_scale_256x300.sbatch",
                    }:
                        expected_account = "general_cs_infra"
                        expected_partition = "grizzly"
                    elif re.search(r"^#SBATCH --gpus", text, re.MULTILINE):
                        expected_account = "general_cs_infra"
                        expected_partition = "backfill_block1"
                    else:
                        expected_account = "general_cs_infra"
                        expected_partition = "cpu_short"
                    self.assertEqual({expected_account}, set(accounts))
                    self.assertEqual({expected_partition}, set(partitions))
                    time_match = re.search(
                        r"^#SBATCH --time=(\d+):(\d+):(\d+)$", text, re.MULTILINE
                    )
                    self.assertIsNotNone(time_match, "missing HH:MM:SS time limit")
                    hours, minutes, seconds = map(int, time_match.groups())
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    max_seconds = {
                        "cpu_short": 4 * 3600,
                        "grizzly": 4 * 3600,
                        "backfill_block1": 16 * 3600,
                        "batch": 4 * 3600,
                    }[expected_partition]
                    self.assertLessEqual(total_seconds, max_seconds)


class NoRetiredSiteReferencesTest(unittest.TestCase):
    def test_repository_cannot_reintroduce_retired_site_paths(self):
        suffixes = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".md", ".patch"}
        paths = []
        for repo in (ROOT, GR00T):
            tracked = subprocess.run(
                ["git", "ls-files"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            paths.extend(repo / path for path in tracked if Path(path).suffix in suffixes)

        forbidden = {
            # Split the retired tokens so this regression test does not itself
            # look like a deployable reference when the repository is audited.
            "retired user mount": re.compile(r"\b" + "sv" + r"l\b", re.IGNORECASE),
            "retired shared mount": re.compile("/" + "vis" + "ion/", re.IGNORECASE),
            "retired external mount": re.compile("/mnt/" + "aml" + "fs", re.IGNORECASE),
            "retired home mount": re.compile(r"\b" + "sail" + r"home\b", re.IGNORECASE),
            "retired scheduler": re.compile(r"\b" + "vis" + r"cam\b", re.IGNORECASE),
            "retired username": re.compile(r"\b" + "raven" + r"h\b", re.IGNORECASE),
            "retired site name": re.compile(r"\b" + "stan" + r"ford\b", re.IGNORECASE),
        }
        failures = []
        for path in sorted(set(paths)):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in forbidden.items():
                match = pattern.search(text)
                if match:
                    line = text.count("\n", 0, match.start()) + 1
                    failures.append(f"{path.relative_to(ROOT)}:{line}: {label}")
        self.assertEqual([], failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
