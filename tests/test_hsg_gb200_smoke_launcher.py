from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_module("hsg_gb200_smoke_gate", ROOT / "scripts/scaling/hsg_gb200_smoke_gate.py")
profile_gate = load_module("profile_validation_gate", ROOT / "scripts/scaling/profile_validation_gate.py")


def write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class HsgGb200SmokeGateTests(unittest.TestCase):
    def run_ffmpeg_gate(self, value: str | None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        if value is None:
            environment.pop("IMAGE_DAGGER_FFMPEG", None)
        else:
            environment["IMAGE_DAGGER_FFMPEG"] = value
        return subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/scaling/check_image_dagger_ffmpeg.sh"),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_ffmpeg_runtime_gate_rejects_unset_missing_and_nonexecutable_paths(self):
        unset = self.run_ffmpeg_gate(None)
        self.assertEqual(unset.returncode, 2)
        self.assertIn("IMAGE_DAGGER_FFMPEG is unset", unset.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            missing = self.run_ffmpeg_gate(str(base / "missing-ffmpeg"))
            self.assertEqual(missing.returncode, 2)
            self.assertIn("missing or not a regular file", missing.stderr)

            nonexecutable = base / "ffmpeg"
            nonexecutable.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
            nonexecutable.chmod(0o644)
            rejected = self.run_ffmpeg_gate(str(nonexecutable))
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("not executable", rejected.stderr)

    def test_ffmpeg_runtime_gate_attests_the_exact_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            encoder = Path(temporary) / "ffmpeg"
            encoder.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  '-version') printf '%s\\n' 'ffmpeg version fixture-7.0.2' ;;\n"
                "  '-hide_banner -encoders') printf '%s\\n' ' V....D libx264 H.264 fixture encoder' ;;\n"
                "  *) exit 64 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            encoder.chmod(0o755)
            accepted = self.run_ffmpeg_gate(str(encoder))
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn(f"IMAGE_DAGGER_FFMPEG={encoder}\n", accepted.stdout)
            self.assertIn(
                "IMAGE_DAGGER_FFMPEG_VERSION=ffmpeg version fixture-7.0.2\n",
                accepted.stdout,
            )
            self.assertIn(
                "IMAGE_DAGGER_FFMPEG_ENCODER=libx264\n", accepted.stdout
            )

    def test_ffmpeg_runtime_gate_rejects_version_only_binary_without_libx264(self):
        with tempfile.TemporaryDirectory() as temporary:
            encoder = Path(temporary) / "ffmpeg"
            encoder.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  '-version') printf '%s\\n' 'ffmpeg version fixture-7.0.2' ;;\n"
                "  '-hide_banner -encoders') printf '%s\\n' ' V....D h264 fixture encoder' ;;\n"
                "  *) exit 64 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            encoder.chmod(0o755)
            rejected = self.run_ffmpeg_gate(str(encoder))
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("does not advertise the required libx264 encoder", rejected.stderr)

    def test_preflight_authenticates_dual_images_engine_assets_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "fast_roborender"
            gr00t = root / "GR00T-VisualSim2Real"
            fast_revision = "1" * 40
            gr00t_revision = "2" * 40
            write(root / ".source_revision", (fast_revision + "\n").encode())
            write(gr00t / ".source_revision", (gr00t_revision + "\n").encode())
            renderer = base / "renderer.sqsh"
            trainer = base / "trainer.sqsh"
            renderer_hash = write(renderer, b"renderer-sm100")
            trainer_hash = write(trainer, b"trainer-sm100")

            image_root = base / "image3f"
            adapter_hash = write(image_root / "adapter/model.safetensors", b"adapter")
            model_root = base / "models"
            foundation_hash = write(model_root / "foundation/weights.bin", b"foundation")
            deployment = {
                "release_id": "release",
                "adapter": {"path": "adapter/model.safetensors", "sha256": adapter_hash},
                "generation_contract": {"steps": 5},
                "acceleration": {},
                "foundation": {
                    "required_files": [
                        {
                            "relative_path": "foundation/weights.bin",
                            "bytes": len(b"foundation"),
                            "sha256": foundation_hash,
                        }
                    ]
                },
            }
            write(
                image_root / "config/deployment.json",
                (json.dumps(deployment) + "\n").encode(),
            )
            generated = base / "generated"
            engine_dir = generated / "engines/sm100"
            engine_hash = write(engine_dir / "model.engine", b"engine")
            manifest = {
                "status": "built_microbenchmarked_and_correctness_checked",
                "gpu": "NVIDIA GB200",
                "compute_capability": "10.0",
                "precision": "bf16_fp16",
                "adapter_sha256": adapter_hash,
                "cuda_graph": {"enabled_for_runtime": False},
                "engine": {"file": "model.engine", "sha256": engine_hash},
            }
            write(engine_dir / "engine_build_manifest.json", (json.dumps(manifest) + "\n").encode())
            (generated / "engines/current").symlink_to("sm100")

            anchors = base / "anchors"
            anchors.mkdir()
            for index in range(2):
                with zipfile.ZipFile(anchors / f"{index}.npz", "w") as archive:
                    for view in gate.VIEWS:
                        archive.writestr(f"gen__{view}.npy", b"fixture")
            teacher = base / "expert.pt"
            teacher_hash = write(teacher, b"teacher")
            args = argparse.Namespace(
                profile="minimal_e2e",
                root=root,
                gr00t_root=gr00t,
                fast_revision=fast_revision,
                gr00t_revision=gr00t_revision,
                renderer_container=renderer,
                renderer_sha256=renderer_hash,
                trainer_container=trainer,
                trainer_sha256=trainer_hash,
                image_root=image_root,
                model_root=model_root,
                generated_root=generated,
                anchors=anchors,
                anchor_min_files=2,
                teacher=teacher,
                teacher_sha256=teacher_hash,
                release_id="release",
                adapter_sha256=adapter_hash,
            )
            report = gate.preflight(args)
            self.assertEqual(report["status"], "validated")
            self.assertEqual(report["engine_sha256"], engine_hash)
            self.assertEqual(report["anchor_count"], 2)
            self.assertEqual(
                report["gr00t_source_tree_sha256"], gate.source_tree_sha256(gr00t)
            )

    def test_source_tree_digest_covers_patched_code_but_ignores_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "gr00t/module.py", b"value = 1\n")
            baseline = gate.source_tree_sha256(root)
            write(root / "gr00t/__pycache__/module.pyc", b"volatile-bytecode")
            self.assertEqual(gate.source_tree_sha256(root), baseline)
            write(root / "gr00t/module.py", b"value = 2\n")
            self.assertNotEqual(gate.source_tree_sha256(root), baseline)

    def make_postrun(
        self, run_dir: Path, world_size: int, *, final_step: int = 1
    ) -> None:
        write(run_dir / "initial_policy.pt", b"initial")
        write(run_dir / f"model_step_{final_step:06d}.pt", b"final")
        marker = {
            "schema": "fast_roborender.dagger_bc_loss.v1",
            "update": 1,
            "world_size": world_size,
            "loss": 0.25,
            "finite_verified": True,
            "nonnegative_verified": True,
        }
        write(
            run_dir / "validation/dagger_bc_loss_step_000000001.json",
            (json.dumps(marker) + "\n").encode(),
        )
        if world_size > 1:
            ddp = {
                "schema": "fast_roborender.ddp_policy_sync.v1",
                "verified": True,
                "optimizer_update": 1,
                "world_size": world_size,
                "policy_parameter_count": 10,
                "policy_parameters_sha256": "3" * 64,
            }
            write(
                run_dir / "validation/ddp_policy_sync_step_000000001.json",
                (json.dumps(ddp) + "\n").encode(),
            )
        write(run_dir / "telemetry/scalars/step.json", b"{}\n")
        write(run_dir / "telemetry/images/step.png", b"png")
        write(run_dir / "telemetry/videos/step.mp4", b"mp4")

    def test_resume_profile_reports_the_post_restart_checkpoint_as_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(run_dir, 1, final_step=2)
            args = argparse.Namespace(
                profile="resume_minimal",
                run_dir=run_dir,
                global_envs=1,
                rollout_steps=2,
                world_size=1,
                fast_revision="8" * 40,
                gr00t_revision="9" * 40,
            )
            report = gate.postrun(args)
            self.assertEqual(report["final_checkpoint_step"], 2)
            self.assertEqual(
                report["final_policy_sha256"],
                hashlib.sha256(b"final").hexdigest(),
            )

    def test_32_profile_postrun_is_accepted_by_existing_profile_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(run_dir, 1)
            fast_revision = "4" * 40
            gr00t_revision = "5" * 40
            args = argparse.Namespace(
                profile="smoke_32x300",
                run_dir=run_dir,
                global_envs=32,
                rollout_steps=300,
                world_size=1,
                fast_revision=fast_revision,
                gr00t_revision=gr00t_revision,
            )
            report = gate.postrun(args)
            validated = profile_gate.validate_profile(
                Path(report["profile_validation_manifest"]),
                "smoke_32x300",
                fast_revision,
                gr00t_revision,
            )
            self.assertEqual(validated["status"], "verified")

    def test_256_profile_postrun_requires_and_attests_eight_rank_ddp(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(run_dir, 8)
            fast_revision = "6" * 40
            gr00t_revision = "7" * 40
            args = argparse.Namespace(
                profile="smoke_256x300",
                run_dir=run_dir,
                global_envs=256,
                rollout_steps=300,
                world_size=8,
                fast_revision=fast_revision,
                gr00t_revision=gr00t_revision,
            )
            report = gate.postrun(args)
            validated = profile_gate.validate_profile(
                Path(report["profile_validation_manifest"]),
                "smoke_256x300",
                fast_revision,
                gr00t_revision,
            )
            self.assertEqual(validated["status"], "verified")
            self.assertEqual(validated["trainer_nodes"], 8)
            self.assertTrue(validated["ddp_policy_parameters_sha256"])

    def test_launcher_declares_four_profiles_and_dual_container_mounts(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for profile in (
            "minimal_e2e",
            "resume_minimal",
            "calibration_3view",
            "smoke_32x300",
            "smoke_256x300",
        ):
            self.assertIn(profile, submit)
            self.assertIn(profile, packed)
        self.assertIn("PACKED_NODES=15", submit)
        self.assertIn("TRAINER_RANKS=8", submit)
        self.assertIn("EXPECTED_RENDERERS=52", submit)
        self.assertIn("PACKED_QOS=normal", submit)
        self.assertGreaterEqual(submit.count("PACKED_QOS=interactive"), 4)
        self.assertIn('--qos="$PACKED_QOS"', submit)
        self.assertIn("qos=$PACKED_QOS", submit)
        self.assertIn('--container-image="$RENDERER_CONTAINER"', packed)
        self.assertIn('--container-image="$TRAINER_CONTAINER"', packed)
        self.assertIn('--gpus="$EXPECTED_RENDERERS" --gpus-per-task=1', packed)
        self.assertIn('--gpus="$TRAINER_RANKS" --gpus-per-task=1', packed)
        self.assertIn("$IMAGE3F_GENERATED_ROOT:/opt/roborender/assets/generated:ro", packed)
        self.assertGreaterEqual(
            packed.count("$GR00T_ROOT:/opt/GR00T-VisualSim2Real:ro"), 2
        )
        self.assertIn("EXPECTED_GR00T_SOURCE_TREE_SHA256", submit)
        self.assertIn("EXPECTED_GR00T_SOURCE_TREE_SHA256", packed)
        self.assertIn("EXPECTED_RESNET18_SHA256", submit)
        self.assertIn("EXPECTED_RESNET18_SHA256", packed)
        self.assertIn("resnet18-f37072fd.pth", submit)
        self.assertIn("export TORCH_HOME", packed)
        self.assertIn(
            'ACTUAL_QOS=$(scontrol show job "$SLURM_JOB_ID" -o', packed
        )
        self.assertIn(
            'ACTUAL_ACCOUNT=$(scontrol show job "$SLURM_JOB_ID" -o', packed
        )
        for contract in (
            "1:1:2:1:1:0:0:interactive",
            "1:1:2:1:1:1:0:interactive",
            "2:32:300:1:7:0:1:interactive",
            "15:256:300:8:52:0:1:normal",
        ):
            self.assertIn(contract, packed)
        self.assertIn("run_train_phase phase_a null 0 1", packed)
        self.assertIn(
            'run_train_phase phase_b "$TRAIN_RUN_DIR/model_step_000001.pt" 1 0',
            packed,
        )
        self.assertIn("resume_continuity_gate.py", packed)
        self.assertIn("--expected-before-step 1 --expected-after-step 2", packed)
        self.assertIn("++env.config.image_dagger_policy_view_count=3", packed)
        wrapper = (
            ROOT / "scripts/scaling/submit_hsg_gb200_resume_smoke.sh"
        ).read_text()
        self.assertIn("export RUN_PROFILE=resume_minimal", wrapper)
        self.assertIn("DRY_RUN", submit)

    def test_resume_boundary_is_a_durable_checkpoint_stop_not_a_changed_target(self):
        callback_path = ROOT / (
            "GR00T-VisualSim2Real/gr00t/rl/trl/callbacks/"
            "model_save_callback.py"
        )
        callback = callback_path.read_text()
        ast.parse(callback)
        self.assertIn("control.should_training_stop = True", callback)
        self.assertIn("test_stop_after_checkpoint_step", callback)
        self.assertIn("checkpoint_path.is_file()", callback)
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        self.assertIn('TRAIN_ABSOLUTE_TARGET=2', packed)
        self.assertIn(
            'algo.trl.num_total_batches="$TRAIN_ABSOLUTE_TARGET"', packed
        )
        self.assertIn(
            'callbacks.model_save.test_stop_after_checkpoint_step="$STOP_AFTER_STEP"',
            packed,
        )

    def test_hsg_wandb_media_watcher_is_fail_closed_and_drained(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for contract in (
            "WANDB_WATCHER_POLL_S",
            "WANDB_WATCHER_READY_TIMEOUT_S",
            "WANDB_WATCHER_DRAIN_TIMEOUT_S",
        ):
            self.assertIn(contract, submit)
            self.assertIn(contract, packed)
            self.assertIn(f"{contract}=${contract}", submit)
        for contract in (
            "wandb_run_watcher.py",
            "WANDB_WATCHER_READY.json",
            "WANDB_UPLOAD_STOP",
            "WANDB_SIDECAR_RUN_ID",
            ".wandb_upload_state.json",
            "drain_wandb_watcher",
            'while kill -0 "$TRAIN_STEP_PID"',
            "return 70",
        ):
            self.assertIn(contract, packed)
        self.assertIn(
            'env HOME="$STAGE_ROOT/runtime/home" WANDB_MODE="$WANDB_MODE"',
            packed,
        )
        self.assertIn("WANDB_WATCHER_DRAIN_TIMEOUT_S must be at most 270", packed)
        self.assertLess(
            packed.index("if [ \"$USE_WANDB\" -eq 1 ]; then\n    if ! kill -0"),
            packed.index('hsg_gb200_smoke_gate.py" postrun'),
        )
        self.assertNotIn('rm -f "$TRAIN_RUN_DIR/telemetry/WANDB_SIDECAR_RUN_ID"', packed)
        self.assertNotIn('rm -f "$TRAIN_RUN_DIR/telemetry/.wandb_upload_state.json"', packed)
        self.assertIn("PACKED_QOS=interactive IMAGE_DAGGER_DUMP_SIM_RGB=0 USE_WANDB=0", submit)
        self.assertIn("--wandb-mode disabled", packed)

    def test_hsg_signal_handlers_cannot_publish_success(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        self.assertIn("trap cleanup EXIT", packed)
        self.assertIn("trap 'terminate_with_status 143' TERM", packed)
        self.assertIn("trap 'terminate_with_status 130' INT", packed)
        self.assertNotIn("trap cleanup EXIT TERM INT", packed)

    def test_hsg_sanitized_path_preserves_slurm_commands(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        self.assertIn(
            "SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/local/apps/slurm/current/bin}",
            submit,
        )
        self.assertIn(
            "PATH=$SLURM_BIN_DIR:/usr/local/bin:/usr/bin:/bin,SLURM_BIN_DIR=$SLURM_BIN_DIR",
            submit,
        )
        self.assertIn('for slurm_command in scontrol srun; do', submit)
        self.assertIn('for slurm_command in scontrol srun; do', packed)
        self.assertIn("export SLURM_EXPORT_ENV=ALL", packed)

    def test_hsg_ffmpeg_path_is_pinned_preflighted_and_propagated(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        encoder = (
            "/opt/viral-sm100/lib/python3.11/site-packages/imageio_ffmpeg/"
            "binaries/ffmpeg-linux-aarch64-v7.0.2"
        )
        self.assertIn(f"HSG_TRAINER_FFMPEG={encoder}", submit)
        self.assertIn("IMAGE_DAGGER_FFMPEG=$IMAGE_DAGGER_FFMPEG", submit)
        self.assertIn("image_dagger_ffmpeg=$IMAGE_DAGGER_FFMPEG", submit)
        self.assertIn("IMAGE_DAGGER_FFMPEG", packed.split("required=(", 1)[1])
        self.assertIn(
            'env IMAGE_DAGGER_FFMPEG="$IMAGE_DAGGER_FFMPEG"', packed
        )
        self.assertIn("check_image_dagger_ffmpeg.sh", packed)
        self.assertIn("export IMAGE_DAGGER_FFMPEG", packed)
        preflight = packed.index('FFMPEG_PREFLIGHT="$RUN_ROOT/trainer_ffmpeg_preflight.txt"')
        renderer_launch = packed.index('srun --nodes="$PACKED_NODES"')
        trainer_launch = packed.index('srun --nodes="$TRAINER_RANKS"')
        self.assertLess(preflight, renderer_launch)
        self.assertLess(preflight, trainer_launch)
        self.assertLess(
            packed.index("export SLURM_EXPORT_ENV=ALL"),
            preflight,
        )

    def test_srun_steps_receive_variables_computed_after_sbatch_submission(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        production = (ROOT / "scripts/train/dagger_scale_256x300.sbatch").read_text()
        for launcher in (packed, production):
            self.assertIn("export SLURM_EXPORT_ENV=ALL", launcher)
            first_srun = re.search(r"(?m)^\s*srun\s", launcher)
            self.assertIsNotNone(first_srun)
            self.assertLess(
                launcher.index("export SLURM_EXPORT_ENV=ALL"),
                first_srun.start(),
            )
        self.assertLess(
            packed.index("export SLURM_EXPORT_ENV=ALL"),
            packed.index("RANK_RENDER_SEED=$((BASE_RENDER_SEED + SLURM_PROCID))"),
        )


if __name__ == "__main__":
    unittest.main()
