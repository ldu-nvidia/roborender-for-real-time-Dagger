import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionLauncherGatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = (
            ROOT / "scripts/train/dagger_scale_256x300.sbatch"
        ).read_text(encoding="utf-8")
        cls.submitter = (
            ROOT / "scripts/scaling/submit_production.sh"
        ).read_text(encoding="utf-8")
        cls.eight_hour_submitter = (
            ROOT / "scripts/scaling/submit_8h_256env.sh"
        ).read_text(encoding="utf-8")
        cls.eval_orchestrator = (
            ROOT / "scripts/eval/orchestrate_fixed_seed_gate.sbatch"
        ).read_text(encoding="utf-8")
        cls.cleanup_watchdog = (
            ROOT / "scripts/scaling/renderer_cleanup_watchdog.sbatch"
        ).read_text(encoding="utf-8")
        cls.oci_submitter = (
            ROOT / "scripts/scaling/submit_oci_packed.sh"
        ).read_text(encoding="utf-8")
        cls.oci_packed = (
            ROOT / "scripts/scaling/oci_packed_256x300.sbatch"
        ).read_text(encoding="utf-8")
        cls.image3f_server = (
            ROOT / "scripts/server/image3f_server.py"
        ).read_text(encoding="utf-8")
        cls.image3f_server_task = (
            ROOT / "scripts/server/image3f_server_array.sbatch"
        ).read_text(encoding="utf-8")

    def test_oci_packed_topology_and_base_initialization_are_explicit(self):
        for contract in (
            "#SBATCH --nodes=8",
            "#SBATCH --gpus-per-node=8",
            "GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}",
            "ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}",
            "CKPT=${CKPT:-none}",
            "EXPECTED_TRAIN_CHECKPOINT_SHA256=${EXPECTED_TRAIN_CHECKPOINT_SHA256:-NONE}",
            'REQUIRED_RENDERERS" -eq 52',
            "renderer_tasks=$REQUIRED_RENDERERS",
            "trainer_tasks=$TRAINER_NODES",
            "spare_gpus=4",
            "reference_policy=FT-FIXED",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, self.oci_packed)
        self.assertNotIn("last.pt", self.oci_packed)
        self.assertNotIn("last.pt", self.oci_submitter)

    def test_oci_renderers_and_trainers_share_one_allocation_safely(self):
        for renderer_contract in (
            '--ntasks="$REQUIRED_RENDERERS"',
            "--ntasks-per-node=7",
            "--distribution=cyclic",
            "--gpus-per-task=1",
            "--exact --exclusive --gpu-bind=closest",
            "SLURM_ARRAY_TASK_ID=$SLURM_PROCID",
            "IMAGE3F_ADVERTISE_HOST=${SLURMD_NODENAME:-$(hostname -f)}",
        ):
            with self.subTest(renderer_contract=renderer_contract):
                self.assertIn(renderer_contract, self.oci_packed)
        for trainer_contract in (
            '--ntasks="$TRAINER_NODES" --ntasks-per-node=1',
            '--cpus-per-task="$TRAINER_CPUS_PER_TASK"',
            '--mem="$TRAINER_MEM_PER_NODE" --exact --exclusive',
            "--gpu-bind=closest",
        ):
            with self.subTest(trainer_contract=trainer_contract):
                self.assertIn(trainer_contract, self.launcher)
        self.assertNotIn("scancel", self.oci_packed)
        self.assertEqual(self.oci_submitter.count("sbatch --parsable"), 1)

    def test_oci_packed_launch_has_asset_identity_and_observability_gates(self):
        for gate in (
            "ANCHOR_MIN_FILES=${ANCHOR_MIN_FILES:-2048}",
            "gen__head_color",
            "gen__hand_left",
            "gen__hand_right",
            "d14dd7bda9292a6f148c7d0a68d60f75",
            "engine_build_manifest.json",
            "fast_roborender source is dirty",
            "GR00T source is dirty",
            "WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}",
        ):
            with self.subTest(gate=gate):
                self.assertIn(gate, self.oci_packed)
        self.assertIn("STARTUP_STAGGER_SCOPE", self.image3f_server_task)
        self.assertIn('os.environ.get("IMAGE3F_ADVERTISE_HOST"', self.image3f_server)

    def test_oci_wrapper_owns_step_cleanup_and_signal_forwarding(self):
        for lifecycle in (
            "trap cleanup EXIT TERM INT",
            "trap forward_pre_timeout USR1",
            'kill -USR1 "$TRAIN_PID"',
            'kill -TERM "$RENDER_STEP_PID"',
            "packed renderer step exited while trainer was live",
        ):
            with self.subTest(lifecycle=lifecycle):
                self.assertIn(lifecycle, self.oci_packed)

    def test_runtime_behavior_is_pinned(self):
        required_overrides = (
            "algo.config.rollout_with_teacher_num_steps=0",
            'algo.config.enforce_teacher_rollout="$ENFORCE_TEACHER_ROLLOUT"',
            'algo.config.ratio_teacher_rollout="$TEACHER_ROLLOUT_RATIO"',
            'algo.config.videogen.rollout_action_source="$VG_ROLLOUT_ACTION_SOURCE"',
            'algo.config.videogen.beta_schedule="$VG_BETA_SCHEDULE"',
            'algo.config.videogen.beta_start="$VG_BETA_START"',
            'algo.config.videogen.beta_end="$VG_BETA_END"',
            'algo.config.videogen.beta_decay_iters="$VG_BETA_DECAY_ITERS"',
            'algo.config.num_learning_epochs="$NUM_LEARNING_EPOCHS"',
            '++algo.config.videogen.primary_view="$VG_PRIMARY_VIEW"',
        )
        for override in required_overrides:
            with self.subTest(override=override):
                self.assertIn(override, self.launcher)

        for runtime_assertion in (
            'IMAGE_DAGGER_EXPECT_PRIMARY_VIEW="$VG_PRIMARY_VIEW"',
            'IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE="$VG_ROLLOUT_ACTION_SOURCE"',
            'IMAGE_DAGGER_EXPECT_BETA_SCHEDULE="$VG_BETA_SCHEDULE"',
            'IMAGE_DAGGER_EXPECT_BETA_START="$VG_BETA_START"',
            'IMAGE_DAGGER_EXPECT_BETA_END="$VG_BETA_END"',
            "IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS=0",
            'IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER="$ENFORCE_TEACHER_ROLLOUT"',
            'IMAGE_DAGGER_EXPECT_TEACHER_RATIO="$TEACHER_ROLLOUT_RATIO"',
            'IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS="$NUM_LEARNING_EPOCHS"',
            "IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF=0.0",
            "IMAGE_DAGGER_EXPECT_ENTROPY_COEF=0.0",
        ):
            with self.subTest(runtime_assertion=runtime_assertion):
                self.assertIn(runtime_assertion, self.launcher)

    def test_production_defaults_are_fresh_policy_safe(self):
        for setting in (
            "BEHAVIOR_SOURCE=${BEHAVIOR_SOURCE:-annealed}",
            "TEACHER_ROLLOUT_RATIO=${TEACHER_ROLLOUT_RATIO:-0.0}",
            "BETA_SCHEDULE=${BETA_SCHEDULE:-linear}",
            "BETA_START=${BETA_START:-1.0}",
            "BETA_END=${BETA_END:-0.0}",
            "BETA_DECAY_ITERS=${BETA_DECAY_ITERS:-500}",
            "NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}",
            "VG_PRIMARY_VIEW=${VG_PRIMARY_VIEW:-head_color}",
        ):
            with self.subTest(setting=setting):
                self.assertIn(setting, self.launcher)
                self.assertIn(setting, self.submitter)

        for pure_bc_override in (
            "algo.config.value_loss_coef=0.0",
            "algo.config.entropy_coef=0.0",
        ):
            with self.subTest(pure_bc_override=pure_bc_override):
                self.assertIn(pure_bc_override, self.launcher)

    def test_external_run_length_uses_rollout_iteration_terminology(self):
        self.assertIn(
            "NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-${NUM_TOTAL_BATCHES:-1}}",
            self.launcher,
        )
        self.assertIn(
            'algo.trl.num_total_batches="$NUM_ROLLOUT_ITERATIONS"', self.launcher
        )
        self.assertIn(
            "transitions_per_rollout_iteration=$TRANSITIONS_PER_ROLLOUT_ITERATION",
            self.launcher,
        )

    def test_wandb_identity_and_media_are_fail_fast(self):
        for setting in (
            "WANDB_EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}",
            'use_wandb=true',
            'wandb.wandb_entity="$WANDB_ENTITY"',
            'wandb.wandb_group=null',
            'scripts/scaling/wandb_preflight.py',
            'scripts/scaling/wandb_run_watcher.py',
            'ROBORENDER_TELEMETRY_DIR="$TRAIN_RUN_DIR/telemetry"',
        ):
            with self.subTest(setting=setting):
                self.assertIn(setting, self.launcher)
        self.assertIn('WANDB_MODE="$WANDB_MODE"', self.launcher)
        self.assertIn('--expected-email "$WANDB_EXPECTED_EMAIL"', self.submitter)

    def test_wandb_final_drain_uses_pre_timeout_budget_and_is_pinned(self):
        self.assertIn(
            "WANDB_WATCHER_DRAIN_TIMEOUT_S=${WANDB_WATCHER_DRAIN_TIMEOUT_S:-240}",
            self.launcher,
        )
        self.assertIn(
            "watcher_deadline=$((SECONDS + WANDB_WATCHER_DRAIN_TIMEOUT_S))",
            self.launcher,
        )
        self.assertIn("export WANDB_WATCHER_DRAIN_TIMEOUT_S=240", self.eight_hour_submitter)
        self.assertIn("unset WANDB_WATCHER_DRAIN_TIMEOUT_S", self.eight_hour_submitter)
        self.assertNotIn("for _ in $(seq 1 30); do\n            kill -0 \"$WANDB_WATCHER_PID\"", self.launcher)

    def test_eight_hour_submission_and_checkpoint_cadence_are_explicit(self):
        self.assertIn("RUN_WALLTIME=${RUN_WALLTIME:-08:00:00}", self.submitter)
        self.assertIn("RENDER_WALLTIME=${RENDER_WALLTIME:-24:00:00}", self.submitter)
        self.assertIn('--time="$RUN_WALLTIME"', self.submitter)
        self.assertIn('--time="$RENDER_WALLTIME"', self.submitter)
        self.assertIn("SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}", self.launcher)
        self.assertIn(
            'callbacks.model_save.save_frequency="$SAVE_FREQUENCY"', self.launcher
        )
        for pin in (
            "GLOBAL_NUM_ENVS=256",
            "ROLLOUT_STEPS=300",
            "TRAINER_NODES=8",
            "NUM_LEARNING_EPOCHS=1",
            "BEHAVIOR_SOURCE=annealed",
            "BETA_SCHEDULE=linear",
            "BETA_START=1.0",
            "BETA_END=0.0",
            "BETA_DECAY_ITERS=500",
            "TEACHER_ROLLOUT_RATIO=0.0",
            "VG_PRIMARY_VIEW=head_color",
            "RUN_WALLTIME=08:00:00",
            "RENDER_WALLTIME=24:00:00",
            "WANDB_EXPECTED_EMAIL=ldu@nvidia.com",
            "FIXED_EVAL_NUM_ENVS=32",
            "FIXED_EVAL_MAX_PARALLEL=4",
            "FIXED_EVAL_MIN_SUCCESS_DELTA=0.02",
            "ENABLE_FIXED_EVAL_GATE=0",
            "CKPT=none",
        ):
            with self.subTest(pin=pin):
                self.assertIn(pin, self.eight_hour_submitter)

    def test_renderer_model_sampler_and_engine_provenance_are_gated(self):
        for pin in (
            "EXPECTED_IMAGE3F_RELEASE_ID=roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0",
            "EXPECTED_IMAGE3F_ADAPTER_SHA256=c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588",
            "EXPECTED_IMAGE3F_INFERENCE_STEPS=5",
            "EXPECTED_IMAGE3F_TEACACHE_THRESHOLD=0.3",
            "EXPECTED_IMAGE3F_CFG_SCALE=1.0",
            "EXPECTED_IMAGE3F_CUDA_GRAPH=disabled",
        ):
            with self.subTest(pin=pin):
                self.assertIn(pin, self.eight_hour_submitter)
        for gate in (
            '--expected-release-id "$EXPECTED_IMAGE3F_RELEASE_ID"',
            '--expected-adapter-sha256 "$EXPECTED_IMAGE3F_ADAPTER_SHA256"',
            '--expected-inference-steps "$EXPECTED_IMAGE3F_INFERENCE_STEPS"',
            '--expected-teacache-threshold "$EXPECTED_IMAGE3F_TEACACHE_THRESHOLD"',
            '--expected-cfg-scale "$EXPECTED_IMAGE3F_CFG_SCALE"',
            '--expected-cuda-graph "$EXPECTED_IMAGE3F_CUDA_GRAPH"',
            "--require-tensorrt-engine",
            '--write-report "$FLEET_DIR/renderer_capabilities.json"',
        ):
            with self.subTest(gate=gate):
                self.assertIn(gate, self.launcher)
        self.assertIn('--dependency="after:$RENDER_JOB_ID"', self.submitter)

    def test_renderer_fleet_ownership_is_transactional(self):
        for lifecycle_gate in (
            '"submission_status": "$submission_status"',
            "write_job_manifest renderer_submitted",
            "write_job_manifest trainer_submission_failed",
            "write_job_manifest trainer_submitted",
            "SUBMISSION_OWNS_RENDERER=1",
            'scancel "$RENDER_JOB_ID"',
            'TRAIN_EXPORTS="$EXPORTS,RENDER_JOB_ID=$RENDER_JOB_ID"',
        ):
            with self.subTest(lifecycle_gate=lifecycle_gate):
                self.assertIn(lifecycle_gate, self.submitter)
        self.assertLess(
            self.submitter.index("write_job_manifest renderer_submitted"),
            self.submitter.index("TRAIN_JOB_ID=$(\"${TRAIN_CMD[@]}\")"),
        )

    def test_fixed_seed_policy_improvement_gate_is_transactional(self):
        for contract in (
            'EVAL_ORCHESTRATOR_CMD=(sbatch --parsable',
            '--dependency="afterany:$TRAIN_JOB_ID"',
            "SUBMISSION_OWNS_TRAINER=1",
            "cancel_new_trainer_job",
            "FAILURE_STATUS=eval_orchestrator_submission_failed",
            '"eval_orchestrator_job_id": $eval_orchestrator_json',
            '"eval_array_job_id": $eval_array_json',
            '"eval_report_job_id": $eval_report_json',
            '"renderer_cleanup_watchdog_job_id": $cleanup_watchdog_json',
            '"training_loss_used_as_evidence": false',
            '"min_absolute_success_delta": $FIXED_EVAL_MIN_SUCCESS_DELTA',
            "EXPECTED_BASELINE_CKPT_SHA256=$BASELINE_CKPT_SHA256",
            "EXPECTED_REFERENCE_CONFIG_SHA256=$REFERENCE_CONFIG_SHA256",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, self.submitter)
        transfer = self.submitter.index(
            "# Only after trainer, cleanup watchdog, and dependent evaluation gate are"
        )
        eval_submit = self.submitter.index(
            'EVAL_ORCHESTRATOR_JOB_ID=$("${EVAL_ORCHESTRATOR_CMD[@]}")'
        )
        self.assertGreater(transfer, eval_submit)
        self.assertIn(
            "update_job_manifest eval_and_report_submitted", self.eval_orchestrator
        )
        self.assertIn('scancel "$EVAL_JOB_ID"', self.eval_orchestrator)
        self.assertIn('scancel "$REPORT_JOB_ID"', self.eval_orchestrator)
        self.assertIn("OWNS_REPORT_JOB=1", self.eval_orchestrator)
        self.assertIn("eval_submission_rolled_back", self.eval_orchestrator)
        self.assertLess(
            self.eval_orchestrator.index('scancel "$REPORT_JOB_ID"'),
            self.eval_orchestrator.index('scancel "$EVAL_JOB_ID"'),
        )
        ownership_transfer = self.eval_orchestrator.index(
            "# Ownership transfers only after both the shared job manifest"
        )
        self.assertGreater(
            ownership_transfer,
            self.eval_orchestrator.rindex("os.replace(temporary, target)"),
        )
        self.assertGreater(
            self.eval_orchestrator.index("OWNS_REPORT_JOB=0", ownership_transfer),
            ownership_transfer,
        )
        self.assertGreaterEqual(self.eval_orchestrator.count("os.fsync"), 4)

    def test_renderer_cleanup_watchdog_covers_pending_trainer_cancellation(self):
        for contract in (
            'RENDER_CLEANUP_CMD=(sbatch --parsable',
            '--dependency="afterany:$TRAIN_JOB_ID"',
            "SUBMISSION_OWNS_WATCHDOG=1",
            "renderer_cleanup_watchdog_submission_failed",
            "# Only after trainer, cleanup watchdog, and dependent evaluation gate are",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, self.submitter)
        self.assertIn('scancel "$RENDER_JOB_ID"', self.cleanup_watchdog)
        self.assertIn("renderer_already_terminal", self.cleanup_watchdog)
        self.assertLess(
            self.submitter.index('RENDER_CLEANUP_WATCHDOG_JOB_ID=$("${RENDER_CLEANUP_CMD[@]}")'),
            self.submitter.index('EVAL_ORCHESTRATOR_JOB_ID=$("${EVAL_ORCHESTRATOR_CMD[@]}")'),
        )

    def test_trainer_exit_cleans_fleet_and_preserves_outcome_markers(self):
        for lifecycle_gate in (
            "#SBATCH --signal=B:USR1@300",
            "trap _production_cleanup EXIT",
            "trap _handle_pre_timeout USR1",
            "TRAIN_RC=124",
            '_cancel_owned_renderer_fleet || true',
            ': > "$TRAIN_RUN_DIR/RUN_DONE"',
            'printf \'%s\\n\' "$rc" > "$TRAIN_RUN_DIR/RUN_FAILED"',
            "--finish-marker WANDB_UPLOAD_STOP",
            'while kill -0 "$SRUN_PID" 2>/dev/null; do',
            "W&B media watcher exited during training",
        ):
            with self.subTest(lifecycle_gate=lifecycle_gate):
                self.assertIn(lifecycle_gate, self.launcher)
        self.assertNotIn('touch "$TRAIN_RUN_DIR/RUN_DONE"', self.launcher)

    def test_eight_hour_wrapper_removes_ambient_escape_hatches(self):
        for pin in (
            "export NUM_ROLLOUT_ITERATIONS=1000",
            "export BASE_RENDER_SEED=42",
            "export SAVE_FREQUENCY=5",
            "export PROJ=roborender_dagger_base_scale_8h",
            "export RUN_ID=dagger_base_256x300_8h_$(date +%Y%m%d_%H%M%S)",
            "unset ALLOW_DIRTY_FAST_ROBORENDER ALLOW_DIRTY_GR00T",
            "unset NUM_TOTAL_BATCHES FLEET_DIR TRAIN_RUN_DIR RUN_DIR",
            "unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_GROUP",
            "unset FIXED_EVAL_SEEDS FIXED_EVAL_NUM_ENVS FIXED_EVAL_MAX_PARALLEL",
            "export FIXED_EVAL_SEEDS=1729,3253,4519,5987,7411,8863,10193,12203",
            "export ENABLE_FIXED_EVAL_GATE=0",
            "export CKPT=none",
        ):
            with self.subTest(pin=pin):
                self.assertIn(pin, self.eight_hour_submitter)

    def test_training_checkpoint_is_hash_pinned_at_execution(self):
        self.assertIn(
            "EXPECTED_TRAIN_CHECKPOINT_SHA256=$BASELINE_CKPT_SHA256", self.submitter
        )
        self.assertIn(
            "ACTUAL_TRAIN_CHECKPOINT_SHA256=$(sha256sum", self.launcher
        )
        self.assertIn(
            '"$ACTUAL_TRAIN_CHECKPOINT_SHA256" = "$EXPECTED_TRAIN_CHECKPOINT_SHA256"',
            self.launcher,
        )

    def test_fresh_initialization_omits_round4_and_is_manifested(self):
        self.assertIn("CKPT=${CKPT:-none}", self.submitter)
        self.assertIn("CKPT=${CKPT:-none}", self.launcher)
        self.assertIn("INITIALIZATION_MODE=base", self.submitter)
        self.assertIn("ACTUAL_TRAIN_CHECKPOINT_SHA256=NONE", self.launcher)
        self.assertIn("CKPT_ARG=checkpoint=null", self.launcher)
        self.assertIn('headless=true "$CKPT_ARG"', self.launcher)
        self.assertIn('"initialization_mode": "$INITIALIZATION_MODE"', self.submitter)
        self.assertNotIn("image_dagger_r4", self.eight_hour_submitter)


if __name__ == "__main__":
    unittest.main()
