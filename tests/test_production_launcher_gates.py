import re
import subprocess
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
        cls.config = (
            GR00T
            / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml"
        ).read_text(encoding="utf-8")
        cls.integrated_trainer = (
            GR00T / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
        ).read_text(encoding="utf-8")

    def test_only_packed_oci_entrypoint_is_reachable(self):
        for wrapper in (self.compat, self.eight_hour):
            self.assertIn("submit_oci_packed.sh", wrapper)
            self.assertNotIn("submit_production.sh\"", wrapper)
        self.assertNotIn("image3f_server_array.sbatch\")", self.submit)
        self.assertIn("oci_packed_256x300.sbatch", self.submit)

    def test_oci_scheduler_contract_is_explicit(self):
        for text in (self.packed, self.trainer, self.renderer):
            self.assertIn("#SBATCH --account=general_cs_infra", text)
            self.assertIn("#SBATCH --partition=backfill_block1", text)
        for setting in (
            "#SBATCH --nodes=8",
            "#SBATCH --gpus-per-node=8",
            "#SBATCH --time=08:15:00",
            "OCI_CONTAINER_IMAGE=${OCI_CONTAINER_IMAGE:-nvcr.io#nvidia/isaac-sim:5.1.0}",
        ):
            self.assertIn(setting, self.packed)

    def test_packed_gpu_topology_matches_256_environment_plan(self):
        for contract in (
            "GLOBAL_NUM_ENVS=${GLOBAL_NUM_ENVS:-256}",
            "ROLLOUT_STEPS=${ROLLOUT_STEPS:-300}",
            "TRAINER_NODES=${TRAINER_NODES:-8}",
            "ENVS_PER_RENDERER=${ENVS_PER_RENDERER:-5}",
            "[ \"$REQUIRED_RENDERERS\" -eq 52 ]",
            "[ $((REQUIRED_RENDERERS + TRAINER_NODES)) -le 64 ]",
            "--ntasks-per-node=7",
            "trainer_tasks=$TRAINER_NODES",
        ):
            self.assertIn(contract, self.packed)

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
        self.assertIn(".oci_source_revision", trainer_setup)
        self.assertNotIn("git clone", trainer_setup)
        self.assertNotIn("git -C", trainer_setup)
        self.assertIn("EXPECTED_GR00T_REVISION", trainer_setup)
        self.assertIn("runtime/envs/wandb-host", wandb_setup)
        self.assertIn("HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}", wandb_setup)
        self.assertIn("NUMPY_VERSION=${NUMPY_VERSION:-1.24.4}", wandb_setup)
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

    def test_standalone_container_python_jobs_are_containerized(self):
        jobs = (
            "scripts/anchors/anchor_pool.sbatch",
            "scripts/anchors/teacher_anchor_render.sbatch",
            "scripts/data/collect_data.sbatch",
            "scripts/data/offline_render.sbatch",
            "scripts/data/preview_render_3view.sbatch",
            "scripts/data/render_shard.sbatch",
            "scripts/data/rerender_3view.sbatch",
            "scripts/eval/eval_latest.sbatch",
            "scripts/eval/eval_policy.sbatch",
            "scripts/eval/eval_student.sbatch",
            "scripts/server/image3f_ft_eval.sbatch",
            "scripts/server/image3f_ft_setup.sbatch",
            "scripts/server/image3f_server.sbatch",
            "scripts/server/raven_server.sbatch",
            "scripts/train/dagger_train.sbatch",
            "scripts/train/dagger_train_3view.sbatch",
            "scripts/train/image_dagger_r4.sbatch",
            "scripts/train/image_dagger_run.sbatch",
            "scripts/train/image_dagger_smoke.sbatch",
            "scripts/train/offline_bc_train.sbatch",
            "tools/make_previews.sbatch",
            "tools/student3v_bench.sbatch",
        )
        for job in jobs:
            text = read(job)
            with self.subTest(job=job):
                self.assertIn('#SBATCH --container-image="nvcr.io#nvidia/isaac-sim:5.1.0"', text)
                self.assertIn("#SBATCH --container-mounts=/lustre/fs11/", text)

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

    def test_all_active_batch_headers_target_oci(self):
        for base in (ROOT / "scripts", ROOT / "tools"):
            for path in base.rglob("*.sbatch"):
                text = path.read_text(encoding="utf-8")
                accounts = re.findall(r"^#SBATCH --account=(\S+)", text, re.MULTILINE)
                partitions = re.findall(r"^#SBATCH --partition=(\S+)", text, re.MULTILINE)
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertTrue(accounts, "missing explicit OCI account")
                    self.assertTrue(partitions, "missing explicit OCI partition")
                    self.assertEqual({"general_cs_infra"}, set(accounts))
                    self.assertEqual({"backfill_block1"}, set(partitions))


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
            "retired user mount": re.compile("/" + "sv" + "l/", re.IGNORECASE),
            "retired shared mount": re.compile("/" + "vis" + "ion/", re.IGNORECASE),
            "retired home mount": re.compile("/" + "sail" + "home/", re.IGNORECASE),
            "retired scheduler": re.compile("vis" + "cam", re.IGNORECASE),
            "retired username": re.compile("raven" + "h", re.IGNORECASE),
            "retired site name": re.compile("stan" + "ford", re.IGNORECASE),
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
