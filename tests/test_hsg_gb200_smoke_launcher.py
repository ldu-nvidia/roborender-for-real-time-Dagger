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
wandb_promotion = load_module(
    "promote_wandb_acceptance",
    ROOT / "scripts/scaling/promote_wandb_acceptance.py",
)


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
            self.assertEqual(
                report["fast_roborender_source_tree_sha256"],
                gate.source_tree_sha256(root),
            )
            with zipfile.ZipFile(anchors / "2.npz", "w") as archive:
                for view in gate.VIEWS:
                    archive.writestr(f"gen__{view}.npy", b"new-anchor")
            mutated = gate.preflight(args)
            self.assertEqual(mutated["anchor_count"], 3)
            self.assertNotEqual(
                mutated["asset_contract_sha256"], report["asset_contract_sha256"]
            )

    def test_source_tree_digest_covers_patched_code_and_rejects_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "gr00t/module.py", b"value = 1\n")
            baseline = gate.source_tree_sha256(root)
            cache = root / "gr00t/__pycache__/module.pyc"
            write(cache, b"volatile-bytecode")
            with self.assertRaisesRegex(gate.GateError, "executable bytecode"):
                gate.source_tree_sha256(root)
            cache.unlink()
            cache.parent.rmdir()
            for suffix in ("pyc", "pyo"):
                legacy = root / f"legacy.{suffix}"
                write(legacy, b"legacy-bytecode")
                with self.assertRaisesRegex(gate.GateError, "executable bytecode"):
                    gate.source_tree_sha256(root)
                legacy.unlink()
            write(root / "gr00t/module.py", b"value = 2\n")
            self.assertNotEqual(gate.source_tree_sha256(root), baseline)

    def test_source_tree_digest_attests_directory_symlink_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "target_a").mkdir()
            (root / "target_b").mkdir()
            link = root / "selected"
            link.symlink_to("target_a", target_is_directory=True)
            baseline = gate.source_tree_sha256(root)
            link.unlink()
            link.symlink_to("target_b", target_is_directory=True)
            self.assertNotEqual(gate.source_tree_sha256(root), baseline)

    def test_sealed_source_snapshot_attestation_and_inventory_are_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            fast_revision = "1" * 40
            gr00t_revision = "2" * 40
            name = f"fast_roborender_{fast_revision[:7]}_{gr00t_revision[:7]}_sealed"
            root = parent / name
            gr00t = root / "GR00T-VisualSim2Real"
            write(root / ".source_revision", (fast_revision + "\n").encode())
            write(gr00t / ".source_revision", (gr00t_revision + "\n").encode())
            write(root / "scripts/tool.py", b"value = 1\n")
            write(gr00t / "gr00t/module.py", b"value = 2\n")
            internal = {
                "schema": "fast_roborender.hsg_source_snapshot.v2",
                "status": "sealed_tracked_hydrated_sources",
                "snapshot_name": name,
                "fast_roborender_revision": fast_revision,
                "gr00t_revision": gr00t_revision,
                "fast_roborender_index_tree": "3" * 40,
                "gr00t_index_tree": "4" * 40,
                "export_method": "stage0-git-blobs-plus-verified-lfs-v2",
                "lfs_file_count": 0,
                "lfs_total_bytes": 0,
                "tracked_only": True,
                "lfs_pointer_oids_verified": True,
                "symlinks_rejected": True,
                "bytecode_rejected": True,
            }
            write(
                root / ".source_snapshot_manifest.json",
                (json.dumps(internal, indent=2, sort_keys=True) + "\n").encode(),
            )
            for path in root.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            root.chmod(0o555)
            fast_tree, records = gate.source_tree_evidence(root)
            gr00t_tree = gate.source_tree_sha256(gr00t)
            inventory = {
                "schema": "fast_roborender.hsg_source_inventory.v1",
                "file_count": len(records),
                "total_file_bytes": sum(record["bytes"] for record in records),
                "files": records,
            }
            inventory_path = parent / f"{name}.inventory.json"
            write(
                inventory_path,
                (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode(),
            )
            inventory_path.chmod(0o444)
            attestation = {
                "schema": "fast_roborender.hsg_source_snapshot_attestation.v2",
                "status": "PASS",
                "snapshot_name": name,
                "fast_roborender_revision": fast_revision,
                "gr00t_revision": gr00t_revision,
                "fast_roborender_index_tree": "3" * 40,
                "gr00t_index_tree": "4" * 40,
                "fast_roborender_source_tree_sha256": fast_tree,
                "gr00t_source_tree_sha256": gr00t_tree,
                "inventory_sha256": gate.sha256_file(inventory_path),
                "deterministic_archive_sha256": "5" * 64,
                "file_count": len(records),
                "total_file_bytes": inventory["total_file_bytes"],
            }
            attestation_path = parent / f"{name}.snapshot_attestation.json"
            write(
                attestation_path,
                (json.dumps(attestation, indent=2, sort_keys=True) + "\n").encode(),
            )
            attestation_path.chmod(0o444)
            evidence = gate.validate_source_snapshot(
                root,
                gr00t,
                fast_revision,
                gr00t_revision,
                fast_tree,
                gr00t_tree,
                records,
            )
            self.assertEqual(
                evidence["source_snapshot_attestation_sha256"],
                gate.sha256_file(attestation_path),
            )
            self.assertEqual(
                evidence["source_snapshot_inventory_sha256"],
                gate.sha256_file(inventory_path),
            )
            attestation_path.chmod(0o644)
            attestation["fast_roborender_source_tree_sha256"] = "f" * 64
            attestation_path.write_text(
                json.dumps(attestation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            attestation_path.chmod(0o444)
            with self.assertRaisesRegex(
                gate.GateError, "attestation fast_roborender_source_tree_sha256 mismatch"
            ):
                gate.validate_source_snapshot(
                    root,
                    gr00t,
                    fast_revision,
                    gr00t_revision,
                    fast_tree,
                    gr00t_tree,
                    records,
                )

    def make_postrun(
        self,
        run_dir: Path,
        world_size: int,
        *,
        final_step: int = 1,
        renderer_count: int = 1,
        renderer_hosts: int = 1,
        wandb: bool = False,
        profile: str = "minimal_e2e",
    ) -> None:
        initial = run_dir / "initial_policy.pt"
        final = run_dir / f"model_step_{final_step:06d}.pt"
        initial_checkpoint_hash = write(initial, b"initial")
        final_checkpoint_hash = write(final, b"final")
        initial_policy_hash = "2" * 64
        final_policy_hash = "3" * 64
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
                "policy_parameters_sha256": final_policy_hash,
            }
            write(
                run_dir / "validation/ddp_policy_sync_step_000000001.json",
                (json.dumps(ddp) + "\n").encode(),
            )
        write(run_dir / "telemetry/scalars/step.json", b"{}\n")
        write(run_dir / "telemetry/images/step.png", b"png")
        write(run_dir / "telemetry/videos/step.mp4", b"mp4")
        policy_change = {
            "schema": "fast_roborender.checkpoint_policy_change.v1",
            "status": "PASS",
            "initial": {
                "path": str(initial.resolve()),
                "checkpoint_step": 0,
                "checkpoint_sha256": initial_checkpoint_hash,
                "policy_sha256": initial_policy_hash,
                "policy_tensor_count": 10,
            },
            "final": {
                "path": str(final.resolve()),
                "checkpoint_step": final_step,
                "checkpoint_sha256": final_checkpoint_hash,
                "policy_sha256": final_policy_hash,
                "policy_tensor_count": 10,
            },
        }
        write(
            run_dir / "validation/checkpoint_policy_change.json",
            (json.dumps(policy_change) + "\n").encode(),
        )
        base, remainder = divmod(renderer_count, renderer_hosts)
        host_counts = {
            f"host{index:02d}": base + (1 if index < remainder else 0)
            for index in range(renderer_hosts)
        }
        placement = {
            "schema": "fast_roborender.renderer_placement_attestation.v1",
            "status": "PASS",
            "renderer_count": renderer_count,
            "unique_endpoint_count": renderer_count,
            "expected_host_counts": host_counts,
            "actual_host_counts": host_counts,
        }
        write(
            run_dir / "validation/renderer_placement.json",
            (json.dumps(placement) + "\n").encode(),
        )
        fleet = run_dir.parent / "fleet"
        (fleet / "endpoints").mkdir(parents=True)
        (fleet / "queues").mkdir()
        ack_dir = fleet / "stop_acks"
        ack_dir.mkdir()
        stop_path = fleet / "STOP"
        run_id = "fixture-run"
        attempt_id = "NONE"
        job_id = "1234"
        stop_hash = write(
            stop_path,
            (
                json.dumps(
                    {
                        "schema": "fast_roborender.renderer_stop_request.v1",
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "job_id": job_id,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        for index in range(renderer_count):
            ack = {
                "schema": "fast_roborender.image3f_stop_ack.v1",
                "status": "STOP_observed",
                "server_id": f"{run_id}:{index:03d}",
                "pid": 1000 + index,
                "stop_sentinel_path": str(stop_path.resolve()),
                "stop_sentinel_sha256": stop_hash,
                "observed_at": "2026-08-22T00:00:00+00:00",
            }
            write(
                ack_dir / f"renderer_{index:03d}.json",
                (json.dumps(ack, indent=2, sort_keys=True) + "\n").encode(),
            )
        gate.renderer_clean_stop(
            argparse.Namespace(
                profile=profile,
                run_id=run_id,
                attempt_id=attempt_id,
                job_id=job_id,
                fleet_dir=fleet,
                evidence=run_dir.parent / "renderer_clean_stop.json",
                expected_renderers=renderer_count,
                renderer_srun_exit_code=0,
                srun_was_live_before_stop=True,
            )
        )
        if wandb:
            drain = {
                "schema": "fast_roborender.wandb_media_drain.v1",
                "status": "PASS",
                "wandb_run_id": "sidecar-run",
                "artifact_count": 3,
                "artifact_kinds": {"scalar": 1, "image": 1, "video": 1},
            }
            scalar_evidence = {
                "axis": "trainer_step",
                "required_point_count": 1,
                "verified_point_count": 1,
                "required_points_sha256": "4" * 64,
                "verified_points_sha256": "4" * 64,
                "required_step_count": 1,
                "required_steps_sha256": "5" * 64,
                "min_step": 0,
                "max_step": 0,
                "all_points_verified": True,
                "resume_training": False,
                "resume_checkpoint_step": 0,
                "rollout_steps": 300,
                "boundary_pre_step": None,
                "boundary_post_step": None,
                "boundary_pre_verified": True,
                "boundary_post_verified": True,
            }
            trainer_scalar_evidence = dict(scalar_evidence)
            trainer_scalar_evidence["axis"] = "_step"
            readback = {
                "schema": "fast_roborender.wandb_training_readback.v3",
                "status": "PASS",
                "account_email": "ldu@nvidia.com",
                "entity": "nvidia",
                "authorization_scope": "team",
                "team_member": True,
                "entity_is_team": True,
                "personal": False,
                "project": (
                    "roborender_dagger_hsg_gb200_production"
                    if profile == "production_256x300"
                    else "roborender_dagger_hsg_gb200_smoke"
                ),
                "trainer_run_id": "trainer-run",
                "sidecar_run_id": "sidecar-run",
                "required_trainer_keys": ["production/dagger_bc_loss"],
                "observed_trainer_keys": ["production/dagger_bc_loss"],
                "trainer_scalar_evidence": trainer_scalar_evidence,
                "required_sidecar_keys": ["trainer_step", "media/image/x", "media/video/x"],
                "observed_sidecar_keys": ["trainer_step", "media/image/x", "media/video/x"],
                "required_sidecar_scalar_keys": ["trainer_step"],
                "observed_sidecar_scalar_keys": ["trainer_step"],
                "sidecar_scalar_evidence": scalar_evidence,
                "required_sidecar_media_keys": ["media/image/x", "media/video/x"],
                "observed_sidecar_media_keys": ["media/image/x", "media/video/x"],
                "required_remote_media_event_count": 2,
                "verified_remote_media_event_count": 2,
                "downloaded_remote_media_file_count": 2,
                "remote_media_history_steps_unique": True,
                "remote_media_paths_unique": True,
                "verified_remote_media_events": [
                    {
                        "key": "media/image/x",
                        "kind": "image",
                        "trainer_step": 0,
                        "remote_history_step": 1,
                        "local_relative_path": "telemetry/images/step.png",
                        "remote_path": "media/images/image.png",
                        "size": 3,
                        "local_md5": "RlLAPdQBNu3+TxhyUlxpaQ==",
                        "remote_md5": "RlLAPdQBNu3+TxhyUlxpaQ==",
                        "local_sha256": "6" * 64,
                        "remote_sha256": "6" * 64,
                        "summary_current": True,
                    },
                    {
                        "key": "media/video/x",
                        "kind": "video",
                        "trainer_step": 1,
                        "remote_history_step": 2,
                        "local_relative_path": "telemetry/videos/step.mp4",
                        "remote_path": "media/videos/video.mp4",
                        "size": 3,
                        "local_md5": "3xDDe4dm2RJU69/PVnPFIA==",
                        "remote_md5": "3xDDe4dm2RJU69/PVnPFIA==",
                        "local_sha256": "7" * 64,
                        "remote_sha256": "7" * 64,
                        "summary_current": True,
                    },
                ],
                "local_artifact_counts": {"scalar": 1, "image": 1, "video": 1},
                "last_api_error_type": "",
            }
            write(
                run_dir / "validation/wandb_media_drain.json",
                (json.dumps(drain) + "\n").encode(),
            )
            write(
                run_dir / "validation/wandb_training_readback.json",
                (json.dumps(readback) + "\n").encode(),
            )

    def test_resume_profile_reports_the_post_restart_checkpoint_as_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(
                run_dir, 1, final_step=2, profile="resume_minimal"
            )
            args = argparse.Namespace(
                profile="resume_minimal",
                run_dir=run_dir,
                global_envs=1,
                rollout_steps=2,
                world_size=1,
                fast_revision="8" * 40,
                gr00t_revision="9" * 40,
                fast_source_tree_sha256="a" * 64,
                gr00t_source_tree_sha256="b" * 64,
                source_snapshot_attestation_sha256="NONE",
                source_snapshot_inventory_sha256="NONE",
                asset_contract_sha256="c" * 64,
                renderer_clean_stop_evidence=run_dir.parent
                / "renderer_clean_stop.json",
                wandb_expected_email="",
                wandb_entity="",
            )
            report = gate.postrun(args)
            self.assertEqual(report["final_checkpoint_step"], 2)
            self.assertEqual(
                report["final_policy_sha256"],
                "3" * 64,
            )

    def test_wandb_readback_v2_and_noninjective_v3_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(
                run_dir,
                1,
                renderer_count=7,
                renderer_hosts=2,
                wandb=True,
                profile="smoke_32x300",
            )
            path = run_dir / "validation/wandb_training_readback.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["schema"] = "fast_roborender.wandb_training_readback.v2"
            path.write_text(json.dumps(report) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "schema mismatch"):
                gate.validate_wandb_evidence(
                    run_dir,
                    "ldu@nvidia.com",
                    "nvidia",
                    "roborender_dagger_hsg_gb200_smoke",
                )

            report["schema"] = "fast_roborender.wandb_training_readback.v3"
            passing = json.loads(json.dumps(report))
            cases = []
            noninjective = json.loads(json.dumps(passing))
            noninjective["verified_remote_media_events"][1][
                "remote_history_step"
            ] = noninjective["verified_remote_media_events"][0][
                "remote_history_step"
            ]
            cases.append(("history is not injective", noninjective))
            split = json.loads(json.dumps(passing))
            split["observed_sidecar_media_keys"] = []
            cases.append(("observed sidecar media", split))
            count = json.loads(json.dumps(passing))
            count["verified_remote_media_event_count"] = 1
            cases.append(("event count", count))
            digest = json.loads(json.dumps(passing))
            digest["verified_remote_media_events"][0]["remote_sha256"] = "8" * 64
            cases.append(("fingerprints differ", digest))
            scalar = json.loads(json.dumps(passing))
            scalar["sidecar_scalar_evidence"]["verified_point_count"] = 0
            cases.append(("scalar point evidence", scalar))
            project = json.loads(json.dumps(passing))
            project["project"] = "unrelated-project"
            cases.append(("project mismatch", project))
            persisted_float_size = json.loads(json.dumps(passing))
            persisted_float_size["verified_remote_media_events"][0]["size"] = 3.0
            cases.append(("remote media record is malformed", persisted_float_size))
            for message, candidate in cases:
                with self.subTest(message=message):
                    path.write_text(
                        json.dumps(candidate) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(gate.GateError, message):
                        gate.validate_wandb_evidence(
                            run_dir,
                            "ldu@nvidia.com",
                            "nvidia",
                            "roborender_dagger_hsg_gb200_smoke",
                        )

            deduplicated = json.loads(json.dumps(passing))
            duplicate_event = json.loads(
                json.dumps(deduplicated["verified_remote_media_events"][0])
            )
            duplicate_event.update(
                {
                    "trainer_step": 10,
                    "remote_history_step": 3,
                    "local_relative_path": "telemetry/images/step2.png",
                    "summary_current": True,
                }
            )
            deduplicated["verified_remote_media_events"][0][
                "summary_current"
            ] = False
            deduplicated["verified_remote_media_events"].append(duplicate_event)
            deduplicated["required_remote_media_event_count"] = 3
            deduplicated["verified_remote_media_event_count"] = 3
            deduplicated["remote_media_paths_unique"] = False
            deduplicated["local_artifact_counts"]["image"] = 2
            path.write_text(json.dumps(deduplicated) + "\n", encoding="utf-8")
            drain_path = run_dir / "validation/wandb_media_drain.json"
            drain = json.loads(drain_path.read_text(encoding="utf-8"))
            drain["artifact_count"] = 4
            drain["artifact_kinds"]["image"] = 2
            drain_path.write_text(json.dumps(drain) + "\n", encoding="utf-8")
            evidence = gate.validate_wandb_evidence(
                run_dir,
                "ldu@nvidia.com",
                "nvidia",
                "roborender_dagger_hsg_gb200_smoke",
            )
            self.assertEqual(evidence["sidecar_run_id"], "sidecar-run")

            stale_summary = json.loads(json.dumps(deduplicated))
            stale_summary["verified_remote_media_events"][0][
                "summary_current"
            ] = True
            stale_summary["verified_remote_media_events"][2][
                "summary_current"
            ] = False
            reversed_local_order = json.loads(json.dumps(deduplicated))
            reversed_local_order["verified_remote_media_events"][0][
                "trainer_step"
            ] = 20
            for candidate in (stale_summary, reversed_local_order):
                path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    gate.GateError, "summary event is stale"
                ):
                    gate.validate_wandb_evidence(
                        run_dir,
                        "ldu@nvidia.com",
                        "nvidia",
                        "roborender_dagger_hsg_gb200_smoke",
                    )

    def test_wandb_readback_rejects_unrelated_resume_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(
                run_dir,
                1,
                renderer_count=7,
                renderer_hosts=2,
                wandb=True,
                profile="smoke_32x300",
            )
            path = run_dir / "validation/wandb_training_readback.json"
            passing = json.loads(path.read_text(encoding="utf-8"))
            trainer = passing["trainer_scalar_evidence"]
            trainer.update(
                {
                    "required_point_count": 2,
                    "verified_point_count": 2,
                    "min_step": 4,
                    "max_step": 5,
                    "required_step_count": 2,
                    "resume_training": True,
                    "resume_checkpoint_step": 5,
                    "boundary_pre_step": 4,
                    "boundary_post_step": 5,
                }
            )
            sidecar = passing["sidecar_scalar_evidence"]
            sidecar.update(
                {
                    "required_point_count": 2,
                    "verified_point_count": 2,
                    "min_step": 1490,
                    "max_step": 1500,
                    "required_step_count": 2,
                    "resume_training": True,
                    "resume_checkpoint_step": 5,
                    "boundary_pre_step": 1490,
                    "boundary_post_step": 1500,
                }
            )

            path.write_text(json.dumps(passing) + "\n", encoding="utf-8")
            gate.validate_wandb_evidence(
                run_dir,
                "ldu@nvidia.com",
                "nvidia",
                "roborender_dagger_hsg_gb200_smoke",
            )

            cases = []
            wrong_trainer = json.loads(json.dumps(passing))
            wrong_trainer["trainer_scalar_evidence"].update(
                {"min_step": 3, "boundary_pre_step": 3}
            )
            cases.append(("_step scalar restart boundary", wrong_trainer))
            split_sidecar = json.loads(json.dumps(passing))
            split_sidecar["sidecar_scalar_evidence"].update(
                {
                    "max_step": 1501,
                    "boundary_pre_step": 1500,
                    "boundary_post_step": 1501,
                }
            )
            cases.append(("trainer_step scalar restart boundary", split_sidecar))
            outside_range = json.loads(json.dumps(passing))
            outside_range["sidecar_scalar_evidence"].update(
                {"boundary_pre_step": 1490, "boundary_post_step": 1501}
            )
            cases.append(("trainer_step scalar restart boundary", outside_range))
            for message, candidate in cases:
                with self.subTest(message=message):
                    path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(gate.GateError, message):
                        gate.validate_wandb_evidence(
                            run_dir,
                            "ldu@nvidia.com",
                            "nvidia",
                            "roborender_dagger_hsg_gb200_smoke",
                        )

    def test_32_profile_postrun_is_accepted_by_existing_profile_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(
                run_dir,
                1,
                renderer_count=7,
                renderer_hosts=2,
                wandb=True,
                profile="smoke_32x300",
            )
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
                fast_source_tree_sha256="a" * 64,
                gr00t_source_tree_sha256="b" * 64,
                source_snapshot_attestation_sha256="d" * 64,
                source_snapshot_inventory_sha256="e" * 64,
                asset_contract_sha256="c" * 64,
                renderer_clean_stop_evidence=run_dir.parent
                / "renderer_clean_stop.json",
                wandb_expected_email="ldu@nvidia.com",
                wandb_entity="nvidia",
            )
            report = gate.postrun(args)
            validated = profile_gate.validate_profile(
                Path(report["profile_validation_manifest"]),
                "smoke_32x300",
                fast_revision,
                gr00t_revision,
                expected_fast_source_tree_sha256="a" * 64,
                expected_gr00t_source_tree_sha256="b" * 64,
                expected_source_snapshot_attestation_sha256="d" * 64,
                expected_source_snapshot_inventory_sha256="e" * 64,
                expected_asset_contract_sha256="c" * 64,
                expected_wandb_email="ldu@nvidia.com",
                expected_wandb_entity="nvidia",
            )
            self.assertEqual(validated["status"], "verified")
            self.assertEqual(validated["renderer_stop_ack_count"], 7)
            self.assertTrue(validated["renderer_stop_ack_set_sha256"])
            ack = run_dir.parent / "fleet/stop_acks/renderer_000.json"
            ack.write_text(ack.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                profile_gate.GateError, "renderer 0 STOP ACK changed"
            ):
                profile_gate.validate_profile(
                    Path(report["profile_validation_manifest"]),
                    "smoke_32x300",
                    fast_revision,
                    gr00t_revision,
                    expected_fast_source_tree_sha256="a" * 64,
                    expected_gr00t_source_tree_sha256="b" * 64,
                    expected_source_snapshot_attestation_sha256="d" * 64,
                    expected_source_snapshot_inventory_sha256="e" * 64,
                    expected_asset_contract_sha256="c" * 64,
                    expected_wandb_email="ldu@nvidia.com",
                    expected_wandb_entity="nvidia",
                )

    def test_256_profile_postrun_requires_and_attests_eight_rank_ddp(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(
                run_dir,
                8,
                renderer_count=52,
                renderer_hosts=15,
                wandb=True,
                profile="smoke_256x300",
            )
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
                fast_source_tree_sha256="a" * 64,
                gr00t_source_tree_sha256="b" * 64,
                source_snapshot_attestation_sha256="d" * 64,
                source_snapshot_inventory_sha256="e" * 64,
                asset_contract_sha256="c" * 64,
                renderer_clean_stop_evidence=run_dir.parent
                / "renderer_clean_stop.json",
                wandb_expected_email="ldu@nvidia.com",
                wandb_entity="nvidia",
            )
            report = gate.postrun(args)
            validated = profile_gate.validate_profile(
                Path(report["profile_validation_manifest"]),
                "smoke_256x300",
                fast_revision,
                gr00t_revision,
                expected_fast_source_tree_sha256="a" * 64,
                expected_gr00t_source_tree_sha256="b" * 64,
                expected_source_snapshot_attestation_sha256="d" * 64,
                expected_source_snapshot_inventory_sha256="e" * 64,
                expected_asset_contract_sha256="c" * 64,
                expected_wandb_email="ldu@nvidia.com",
                expected_wandb_entity="nvidia",
            )
            self.assertEqual(validated["status"], "verified")
            self.assertEqual(validated["trainer_nodes"], 8)
            self.assertTrue(validated["ddp_policy_parameters_sha256"])

    def test_production_postrun_requires_exact_step_500(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "train"
            self.make_postrun(
                run_dir,
                8,
                final_step=500,
                renderer_count=52,
                renderer_hosts=15,
                wandb=True,
                profile="production_256x300",
            )
            args = argparse.Namespace(
                profile="production_256x300",
                run_dir=run_dir,
                global_envs=256,
                rollout_steps=300,
                world_size=8,
                final_step=500,
                fast_revision="a" * 40,
                gr00t_revision="b" * 40,
                fast_source_tree_sha256="a" * 64,
                gr00t_source_tree_sha256="b" * 64,
                source_snapshot_attestation_sha256="d" * 64,
                source_snapshot_inventory_sha256="e" * 64,
                asset_contract_sha256="c" * 64,
                renderer_clean_stop_evidence=run_dir.parent
                / "renderer_clean_stop.json",
                wandb_expected_email="ldu@nvidia.com",
                wandb_entity="nvidia",
            )
            report = gate.postrun(args)
            self.assertEqual(report["final_checkpoint_step"], 500)
            args.final_step = 499
            with self.assertRaisesRegex(gate.GateError, "final step changed"):
                gate.postrun(args)

    def test_launcher_declares_four_profiles_and_dual_container_mounts(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for profile in (
            "minimal_e2e",
            "resume_minimal",
            "calibration_3view",
            "smoke_32x300",
            "smoke_256x300",
            "production_256x300",
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
        self.assertEqual(
            packed.count('--gpus=0 --cpus-per-task=2 --mem=4G --exact --exclusive'),
            7,
        )
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
        self.assertIn("IMMUTABLE_RO_MOUNTS=", packed)
        self.assertIn("$ASSET_ROOT:$ASSET_ROOT:ro", packed)
        self.assertEqual(
            packed.count('$STAGE_ROOT:$STAGE_ROOT,$IMMUTABLE_RO_MOUNTS'), 9
        )
        self.assertIn("FINAL_ASSET_PREFLIGHT=", packed)
        self.assertIn("immutable asset contract changed during the allocation", packed)
        self.assertIn("TensorRT engine changed during the allocation", packed)
        self.assertIn(
            'ACTUAL_QOS=$(scontrol show job "$SLURM_JOB_ID" -o', packed
        )
        self.assertIn(
            'ACTUAL_ACCOUNT=$(scontrol show job "$SLURM_JOB_ID" -o', packed
        )
        for contract in (
            "1:1:2:1:1:0:0:interactive:batch",
            "1:1:2:1:1:1:0:interactive:batch",
            "2:32:300:1:7:0:1:interactive:batch",
            "15:256:300:8:52:0:1:normal:batch",
            "15:256:300:8:52:0:1:normal:batch_long",
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
        self.assertIn(
            "SMOKE_MOTION_FILE=gr00t/rl/data/motions/g1_wsdpt/33demos_675_775/"
            "record_2025-10-14_21-28-47_0.675m.npz",
            packed,
        )
        self.assertIn(
            "minimal_e2e|resume_minimal|calibration_3view", packed
        )
        self.assertIn(
            "env.config.reset_from_dataset.use_motion_file_dir=false", packed
        )
        self.assertIn(
            '"env.config.reset_from_dataset.motion_files=[$SMOKE_MOTION_FILE]"',
            packed,
        )
        self.assertLess(
            packed.index('"${SMOKE_DATASET_ARGS[@]}"'),
            packed.index("algo.config.videogen.raytraced_mix_prob=0.0"),
        )
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
        self.assertIn('TRAIN_ABSOLUTE_TARGET=$NUM_ROLLOUT_ITERATIONS', packed)
        self.assertIn('EXPECTED_TRAINING_CONTRACT="2:1:0:0:none:NONE:train"', packed)
        self.assertIn(
            'algo.trl.num_total_batches="$TRAIN_ABSOLUTE_TARGET"', packed
        )
        self.assertIn(
            'callbacks.model_save.test_stop_after_checkpoint_step="$STOP_AFTER_STEP"',
            packed,
        )

    def test_checkpoint_hashes_are_pinned_at_load_and_final_acceptance(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        trainer = (ROOT / (
            "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer.py"
        )).read_text()
        checkpoint_io = (ROOT / (
            "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/checkpoint_io.py"
        )).read_text()
        videogen_trainer = (ROOT / (
            "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/"
            "distill_trainer_videogen.py"
        )).read_text()

        # Production resume is pinned by the submitter-provided digest both in
        # the pre-resume gate and at the trainer's actual deserialization.
        self.assertIn(
            '--expected-checkpoint-sha256 "$EXPECTED_TRAIN_CHECKPOINT_SHA256"',
            packed,
        )
        self.assertIn(
            'expected_sha256=os.environ.get("EXPECTED_TRAIN_CHECKPOINT_SHA256")',
            trainer,
        )
        self.assertIn("require_expected_sha256=production_resume", trainer)
        self.assertIn('getattr(os, "O_NOFOLLOW", 0)', checkpoint_io)
        self.assertIn("handle.seek(0)", checkpoint_io)
        self.assertLess(
            checkpoint_io.index("actual_sha256 = digest.hexdigest()"),
            checkpoint_io.index("payload = torch_module.load("),
        )

        # The pre-resume snapshot also pins an ordered digest for each rank's
        # Image3F stream/RNG state.  Only a production resume exports it, and
        # every rank selects its own digest before same-descriptor loading.
        self.assertIn("unset EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S", packed)
        self.assertIn(
            "EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S=$(python3 -", packed
        )
        production_resume_block = packed.split(
            'if [ "$RUN_PROFILE" = production_256x300 ] && '
            '[ "$RESUME_TRAINING" = 1 ]; then',
            1,
        )[1].split("\nfi", 1)[0]
        self.assertIn(
            "EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S=$(python3 -",
            production_resume_block,
        )
        self.assertIn('ranks = payload.get("runtime_ranks")', packed)
        self.assertIn('record["rank"] != expected_rank', packed)
        self.assertIn("export EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S", packed)
        self.assertIn(
            'os.environ.get("EXPECTED_IMAGE_DAGGER_RUNTIME_SHA256S")',
            videogen_trainer,
        )
        self.assertIn("select_rank_checkpoint_sha256(", videogen_trainer)
        self.assertIn("expected_sha256=expected_runtime_sha256", videogen_trainer)
        self.assertIn("require_expected_sha256=production_resume", videogen_trainer)

        # The newly produced step-500 checkpoint is pinned once, then the same
        # digest is required by both resumability and policy-change gates.
        self.assertIn(
            'FINAL_PRODUCTION_CHECKPOINT_SHA256=$(sha256sum '
            '"$FINAL_PRODUCTION_CHECKPOINT"',
            packed,
        )
        self.assertIn(
            '"$FINAL_CHECKPOINT_SNAPSHOT" "$TRAINER_RANKS" required \\\n'
            '        "$FINAL_PRODUCTION_CHECKPOINT_SHA256"',
            packed,
        )
        self.assertIn(
            '--expected-initial-sha256 "$INITIAL_POLICY_CHECKPOINT_SHA256"',
            packed,
        )
        self.assertIn(
            '--expected-final-sha256 "$FINAL_POLICY_CHECKPOINT_SHA256"',
            packed,
        )
        self.assertIn(
            'final checkpoint changed between resumability and policy gates',
            packed,
        )

    def test_hsg_production_profile_is_long_fresh_resumable_and_fail_closed(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        wrapper = (
            ROOT / "scripts/scaling/submit_hsg_gb200_production.sh"
        ).read_text()
        for contract in (
            "PACKED_NODES=15 GLOBAL_NUM_ENVS=256 ROLLOUT_STEPS=300",
            "TRAINER_RANKS=8 EXPECTED_RENDERERS=52 RUN_WALLTIME=08:00:00",
            "PACKED_PARTITION=batch_long PACKED_QOS=normal",
            "NUM_ROLLOUT_ITERATIONS=${NUM_ROLLOUT_ITERATIONS:-500}",
            "SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}",
            "PREREQUISITE_PROFILE_EXPECTED=smoke_256x300",
            "fresh production must use CKPT=none, checkpoint hash NONE, and no resume candidate",
            "production resume requires the original RUN_ID",
            "production resume requires the original RUN_ROOT",
            "production RUN_ROOT must be exactly",
            "production resume checkpoint must be inside the original training directory",
            "production resume is missing trainer W&B metadata",
            "production resume is missing the media W&B run ID",
            "refusing to overwrite production run identity",
            "refusing to overwrite submission manifest",
        ):
            self.assertIn(contract, submit)
        self.assertIn("export RUN_PROFILE=production_256x300", wrapper)
        for contract in (
            'EXPECTED_CONTRACT="15:256:300:8:52:0:1:normal:batch_long"',
            'EXPECTED_TRAINING_CONTRACT="500:5:',
            'TRAIN_ABSOLUTE_TARGET=$NUM_ROLLOUT_ITERATIONS',
            'callbacks.model_save.save_frequency="$SAVE_FREQUENCY"',
            "run_train_phase production_fresh null 0 0",
            'run_train_phase production_resume "$CKPT" 1 0',
            "pre_resume_snapshot.json",
            "final_checkpoint_snapshot.json",
            "resume_continuity_report.json",
            '--wandb-mode required',
            'printf \'%s\\n\' "$rc" > "$RUN_ATTEMPT_DIR/ATTEMPT_FAILED"',
            ': > "$RUN_ROOT/RUN_DONE"',
            "resume_candidate.txt",
        ):
            self.assertIn(contract, packed)
        self.assertIn(
            'elif [ "$RUN_PROFILE" = production_256x300 ]; then\n'
            '        printf \'%s\\n\' "$rc" > "$RUN_ATTEMPT_DIR/ATTEMPT_FAILED"',
            packed,
        )
        self.assertIn("schema=fast_roborender.hsg_production_run_identity.v1", submit)
        self.assertIn("fast_roborender_source_tree_sha256", submit)
        self.assertIn("gr00t_source_tree_sha256", submit)

    def test_hsg_provenance_binds_both_source_trees(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for source in (submit, packed):
            self.assertIn("EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256", source)
            self.assertIn("EXPECTED_GR00T_SOURCE_TREE_SHA256", source)
        self.assertIn('tree-hash --root "$ROOT"', packed)
        self.assertIn('tree-hash --root "$GR00T_ROOT"', packed)
        for source in (submit, packed):
            self.assertIn("EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256", source)
            self.assertIn("EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256", source)
            self.assertIn("source_snapshot_attestation_sha256", source)
            self.assertIn("source_snapshot_inventory_sha256", source)
        self.assertGreaterEqual(
            packed.count('payload.get("source_snapshot_attestation_sha256", "")'),
            2,
        )
        self.assertGreaterEqual(
            packed.count('payload.get("source_snapshot_inventory_sha256", "")'),
            2,
        )

    def test_hsg_calibration_postrun_binds_complete_sealed_provenance(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        calibration_publish = packed.rsplit("    calibration_3view)\n", 1)[1].split(
            "\n        ;;", 1
        )[0]
        for contract in (
            '--expected-run-id "$RUN_ID"',
            '"$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256"',
            '"$EXPECTED_GR00T_SOURCE_TREE_SHA256"',
            '"$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256"',
            '"$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256"',
            '"$EXPECTED_ASSET_CONTRACT_SHA256"',
        ):
            self.assertIn(contract, calibration_publish)

    def test_hsg_calibration_manifest_destination_exactly_binds_run_id(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        exact_destination = (
            "camera_calibration_manifest_${RUN_ID}.json"
        )
        self.assertIn(
            "EXPECTED_CAMERA_CALIBRATION_MANIFEST_DESTINATION="
            "$STAGE_ROOT/runtime/validation/" + exact_destination,
            submit,
        )
        self.assertIn(
            "expected_destination=$validation_dir/" + exact_destination,
            packed,
        )
        for launcher in (submit, packed):
            self.assertIn("must be a direct validation child", launcher)
            self.assertIn("calibration validation directory must be canonical", launcher)
            self.assertIn(exact_destination, launcher)

    def test_hsg_calibration_uses_exclusive_fresh_run_root_reservation(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        self.assertIn("calibration RUN_ROOT must be exactly", submit)
        self.assertIn('camera_calibration_gate.py" reserve-run-root', submit)
        self.assertIn('--run-root "$RUN_ROOT" --run-id "$RUN_ID"', submit)

    def test_queued_calibration_reasserts_exact_reserved_run_directories(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for contract in (
            "validate_calibration_run_directories",
            'expected_run_root=$runtime_dir/$RUN_ID',
            '[ "$RUN_ROOT" = "$expected_run_root" ]',
            '[ "$RUN_ATTEMPT_DIR" = "$RUN_ROOT" ]',
            '[ "$FLEET_DIR" = "$RUN_ROOT/fleet" ]',
            '[ "$TRAIN_RUN_DIR" = "$RUN_ROOT/train" ]',
            '[ ! -L "$run_directory" ]',
            'canonical=$(realpath -e "$run_directory")',
            '[ "$canonical" = "$run_directory" ]',
        ):
            self.assertIn(contract, packed)

    def test_hsg_calibration_manifest_destination_is_exported_and_bound(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        self.assertIn(
            ",CAMERA_CALIBRATION_MANIFEST_DESTINATION="
            "$CAMERA_CALIBRATION_MANIFEST_DESTINATION,",
            submit,
        )
        self.assertIn(
            "camera_calibration_manifest_destination="
            "$CAMERA_CALIBRATION_MANIFEST_DESTINATION",
            submit,
        )
        required = packed.split("required=(", 1)[1].split(")", 1)[0]
        self.assertIn("CAMERA_CALIBRATION_MANIFEST_DESTINATION", required)
        self.assertIn('camera_calibration_gate.py" publish', packed)
        self.assertIn(
            '--destination "$CAMERA_CALIBRATION_MANIFEST_DESTINATION"', packed
        )
        self.assertNotIn(
            'publish_once "$TECHNICAL" '
            '"$STAGE_ROOT/runtime/validation/camera_calibration_manifest.json"',
            packed,
        )
        self.assertNotIn("publish_once()", packed)

    def test_hsg_calibration_manifest_destination_rejects_symlinks_twice(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for launcher in (submit, packed):
            self.assertIn(
                '[ ! -L "$CAMERA_CALIBRATION_MANIFEST_DESTINATION" ]',
                launcher,
            )
            self.assertIn(
                "calibration manifest destination cannot be a symlink",
                launcher,
            )
        self.assertEqual(
            packed.count("validate_camera_calibration_manifest_destination"),
            3,
        )
        self.assertLess(
            packed.rindex("validate_camera_calibration_manifest_destination"),
            packed.index(
                'python3 "$ROOT/scripts/scaling/camera_calibration_gate.py" publish'
            ),
        )

    def test_hsg_noncalibration_profiles_force_destination_none(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        self.assertIn(
            "CAMERA_CALIBRATION_MANIFEST_DESTINATION="
            "${CAMERA_CALIBRATION_MANIFEST_DESTINATION:-NONE}",
            submit,
        )
        for launcher in (submit, packed):
            self.assertIn(
                'must use CAMERA_CALIBRATION_MANIFEST_DESTINATION=NONE',
                launcher,
            )

    def test_hsg_calibration_docs_use_the_versioned_destination(self):
        readme = (ROOT / "README.md").read_text()
        runbook = (ROOT / "docs/HSG_GB200_PRODUCTION_RUNBOOK.md").read_text()
        for document in (readme, runbook):
            self.assertIn(
                'camera_calibration_manifest_${RUN_ID}.json', document
            )
            self.assertIn(
                'camera_calibration_acceptance_${RUN_ID}.json', document
            )
            self.assertIn('--output "$CALIBRATION_ACCEPTANCE"', document)
            self.assertIn(
                'export CAMERA_CALIBRATION_ACCEPTANCE="$CALIBRATION_ACCEPTANCE"',
                document,
            )
            self.assertGreaterEqual(
                document.count(
                    'CAMERA_CALIBRATION_ACCEPTANCE="$CALIBRATION_ACCEPTANCE"'
                ),
                3,
            )
            self.assertNotIn(
                "export CALIBRATION_MANIFEST=/absolute/run/path/validation/"
                "camera_calibration_manifest.json",
                document,
            )
            self.assertNotIn(
                '--output "$STAGE_ROOT/runtime/validation/'
                'camera_calibration_acceptance.json"',
                document,
            )

    def test_hsg_acceptance_verification_binds_versioned_calibration_run(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for launcher in (submit, packed):
            self.assertIn("CAMERA_CALIBRATION_RUN_ID", launcher)
            self.assertIn(
                "camera_calibration_acceptance_${CAMERA_CALIBRATION_RUN_ID}.json",
                launcher,
            )
            self.assertIn('--expected-run-id "$CAMERA_CALIBRATION_RUN_ID"', launcher)
        self.assertIn(
            "CAMERA_CALIBRATION_RUN_ID=$CAMERA_CALIBRATION_RUN_ID,", submit
        )
        self.assertIn("camera_calibration_run_id=", submit)

    def test_hsg_profile_validation_calls_bind_complete_sealed_provenance(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        ordered_submit_contract = (
            '--expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \\\n'
            '        --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \\\n'
            "        --expected-source-snapshot-attestation-sha256 \\\n"
            '          "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \\\n'
            "        --expected-source-snapshot-inventory-sha256 \\\n"
            '          "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \\\n'
            '        --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \\'
        )
        ordered_packed_contract = (
            '--expected-fast-source-tree-sha256 "$EXPECTED_FAST_ROBORENDER_SOURCE_TREE_SHA256" \\\n'
            '            --expected-gr00t-source-tree-sha256 "$EXPECTED_GR00T_SOURCE_TREE_SHA256" \\\n'
            "            --expected-source-snapshot-attestation-sha256 \\\n"
            '              "$EXPECTED_SOURCE_SNAPSHOT_ATTESTATION_SHA256" \\\n'
            "            --expected-source-snapshot-inventory-sha256 \\\n"
            '              "$EXPECTED_SOURCE_SNAPSHOT_INVENTORY_SHA256" \\\n'
            '            --expected-asset-contract-sha256 "$EXPECTED_ASSET_CONTRACT_SHA256" \\'
        )

        submit_predecessor_gate = submit.split(
            'python3 "$ROOT/scripts/scaling/profile_validation_gate.py"', 1
        )[1].split("\nfi", 1)[0]
        packed_postrun_gate = packed.split(
            "    smoke_32x300|smoke_256x300)\n", 1
        )[1].split("\n        ;;", 1)[0]
        self.assertIn(ordered_submit_contract, submit_predecessor_gate)
        self.assertIn(ordered_packed_contract, packed_postrun_gate)

    def test_hsg_production_rejects_source_bytecode_before_and_after_queueing(self):
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        for launcher in (submit, packed):
            self.assertIn("-name __pycache__", launcher)
            self.assertIn("-name '*.pyc'", launcher)
            self.assertIn("-name '*.pyo'", launcher)
            self.assertIn("contains executable bytecode", launcher)

    def test_hsg_wandb_validation_uses_writable_runtime_scratch(self):
        validator = (
            ROOT / "scripts/scaling/hsg_gb200_wandb_validate.sbatch"
        ).read_text()
        self.assertIn(
            "WANDB_SCRATCH=${WANDB_SCRATCH:-$STAGE_ROOT/runtime/wandb-validation/$SLURM_JOB_ID}",
            validator,
        )
        self.assertIn('--container-workdir="$WANDB_SCRATCH/work"', validator)
        self.assertIn('WANDB_DIR="$WANDB_SCRATCH/wandb"', validator)
        self.assertIn('WANDB_CACHE_DIR="$WANDB_SCRATCH/cache"', validator)
        self.assertIn('WANDB_CONFIG_DIR="$WANDB_SCRATCH/config"', validator)
        self.assertIn('WANDB_DATA_DIR="$WANDB_SCRATCH/data"', validator)
        self.assertNotIn('--container-workdir="$ROOT/scripts/scaling"', validator)

    def test_hsg_wandb_validation_promotes_job_record_without_clobbering(self):
        validator = (
            ROOT / "scripts/scaling/hsg_gb200_wandb_validate.sbatch"
        ).read_text()
        self.assertIn(
            "WANDB_ACCEPTANCE=${WANDB_ACCEPTANCE:-$STAGE_ROOT/runtime/validation/"
            "wandb_online_smoke_nvidia_acceptance.json}",
            validator,
        )
        self.assertIn("promote_wandb_acceptance.py", validator)
        self.assertLess(
            validator.index("grep -q '\"status\": \"PASS\"'"),
            validator.index("promote_wandb_acceptance.py"),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wandb_online_smoke_123.json"
            destination = root / "wandb_online_smoke_nvidia_acceptance.json"
            document = {
                "schema": "fast_roborender.wandb_online_smoke.v2",
                "status": "PASS",
                "account_email": "ldu@nvidia.com",
                "entity": "nvidia",
                "authorization_scope": "team",
                "team_member": True,
                "entity_is_team": True,
                "personal": False,
                "observed_keys": [
                    "validation/scalar",
                    "validation/image",
                    "validation/video",
                ],
                "run_id": "first",
            }
            source.write_text(json.dumps(document) + "\n", encoding="utf-8")
            digest = wandb_promotion.promote(
                source,
                destination,
                expected_email="ldu@nvidia.com",
                expected_entity="nvidia",
                authorization_scope="team",
            )
            self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            self.assertEqual(
                wandb_promotion.promote(
                    source,
                    destination,
                    expected_email="ldu@nvidia.com",
                    expected_entity="nvidia",
                    authorization_scope="team",
                ),
                digest,
            )

            document["run_id"] = "different"
            other = root / "wandb_online_smoke_456.json"
            other.write_text(json.dumps(document) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                wandb_promotion.PromotionError, "refusing to replace different"
            ):
                wandb_promotion.promote(
                    other,
                    destination,
                    expected_email="ldu@nvidia.com",
                    expected_entity="nvidia",
                    authorization_scope="team",
                )
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), digest)

    def test_wandb_acceptance_promotion_rejects_nonpassing_or_wrong_account(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "failed.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.wandb_online_smoke.v2",
                        "status": "FAIL",
                        "account_email": "other@example.com",
                        "entity": "nvidia",
                        "authorization_scope": "team",
                        "team_member": True,
                        "entity_is_team": True,
                        "personal": False,
                        "observed_keys": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                wandb_promotion.PromotionError, "status mismatch"
            ):
                wandb_promotion.promote(
                    source,
                    root / "acceptance.json",
                    expected_email="ldu@nvidia.com",
                    expected_entity="nvidia",
                    authorization_scope="team",
                )

    def test_hsg_runbook_uses_sealed_tools_and_documents_wandb_promotion(self):
        runbook = (ROOT / "docs/HSG_GB200_PRODUCTION_RUNBOOK.md").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn(
            'WANDB_ACCEPTANCE="$STAGE_ROOT/runtime/validation/'
            'wandb_online_smoke_nvidia_acceptance.json"',
            runbook,
        )
        self.assertIn("A PASS atomically creates $WANDB_ACCEPTANCE", runbook)
        self.assertIn("different existing acceptance is never overwritten", runbook)
        self.assertIn(
            'python "$ROOT/scripts/scaling/camera_calibration_gate.py" accept',
            runbook,
        )
        self.assertNotIn(
            "bash scripts/scaling/submit_hsg_gb200_", readme
        )
        self.assertNotIn(
            "python scripts/scaling/camera_calibration_gate.py accept", readme
        )

    def test_hsg_batch_smoke_opts_into_immediate_isaac_shutdown(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        setting = "export IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP=1"
        self.assertEqual(packed.count(setting), 1)
        self.assertLess(
            packed.index(setting),
            packed.index('srun --nodes="$TRAINER_RANKS"'),
        )
        self.assertIn("export SLURM_EXPORT_ENV=ALL", packed)
        self.assertLess(
            packed.index("export SLURM_EXPORT_ENV=ALL"),
            packed.index(setting),
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
        self.assertIn(
            "else\n"
            "    # Keep the sanitized sbatch export total even when telemetry is disabled.\n"
            "    # The batch entrypoint requires every exported contract field to be\n"
            "    # nonempty before it branches on USE_WANDB; `null` is also Hydra's literal\n"
            "    # disabled entity value.\n"
            "    WANDB_ENTITY=null\n"
            "    WANDB_ONLINE_VALIDATION=NONE\n"
            "    EXPECTED_WANDB_ONLINE_VALIDATION_SHA256=NONE\n"
            "fi",
            submit,
        )
        self.assertIn("--wandb-mode disabled", packed)

    def test_hsg_signal_handlers_cannot_publish_success(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        self.assertIn("trap cleanup EXIT", packed)
        self.assertIn("trap 'terminate_with_status 143' TERM", packed)
        self.assertIn("trap 'terminate_with_status 130' INT", packed)
        self.assertNotIn("trap cleanup EXIT TERM INT", packed)

    def test_hsg_renderer_shutdown_is_cooperative_attested_and_fail_closed(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        array = (ROOT / "scripts/server/image3f_server_array.sbatch").read_text()
        server = (ROOT / "scripts/server/image3f_server.py").read_text()
        for contract in (
            "FLEET_STOP_FILE=$FLEET_CANONICAL/STOP",
            "RENDERER_CLEAN_STOP_EVIDENCE=$RUN_ATTEMPT_CANONICAL/renderer_clean_stop.json",
            "stop_renderer_fleet_cleanly",
            'renderer srun exited before STOP request',
            'if wait "$pid"; then renderer_rc=0; else renderer_rc=$?; fi',
            'endpoint_count=$(find "$FLEET_DIR/endpoints" -mindepth 1 -maxdepth 1',
            'ready_count=$(find "$FLEET_DIR/queues" -name READY',
            'ack_count=$(find "$FLEET_DIR/stop_acks"',
            "renderer-clean-stop",
            "--srun-was-live-before-stop",
            '--renderer-clean-stop-evidence "$RENDERER_CLEAN_STOP_EVIDENCE"',
            'forced_termination=1',
            '[ "$forced_termination" -eq 0 ] || return 74',
        ):
            self.assertIn(contract, packed)
        self.assertLess(
            packed.index('stop_renderer_fleet_cleanly "$RENDER_STEP_PID"'),
            packed.index('hsg_gb200_smoke_gate.py" postrun'),
        )
        self.assertIn('export IMAGE3F_STOP_FILE="$RUN_DIR/STOP"', array)
        self.assertIn(
            'export IMAGE3F_STOP_ACK_FILE="$RUN_DIR/stop_acks/renderer_${INDEX}.json"',
            array,
        )
        self.assertIn("export IMAGE3F_STOP_POLL_S=1", array)
        self.assertIn("trap cleanup EXIT", array)
        self.assertIn("trap 'exit 143' TERM", array)
        self.assertNotIn("trap cleanup EXIT INT TERM", array)
        self.assertIn(
            "Image3F server exited zero without the exact STOP sentinel", array
        )
        self.assertIn(
            "Image3F server exited zero without its exact nonempty STOP ACK", array
        )
        self.assertLess(
            server.index("if stop_controller.requested():"),
            server.index("Image3FSession(memory_mode"),
        )
        for contract in (
            "connections.start_handler(sock, handler)",
            "connections.close_clients()",
            "with session_lock:",
            "connections.drain(timeout_seconds=10.0)",
            "acknowledge_stop(stop_controller)",
        ):
            self.assertIn(contract, server)
        self.assertNotIn(
            'RENDERER_CLEAN_STOP_EVIDENCE=$(dirname "$RESUME_CANDIDATE")',
            packed,
        )
        self.assertNotIn('if [ "$PRODUCTION_ACTION" != validate ]; then', packed)
        self.assertIn(
            "Finalize the already durable step-500 checkpoint without retraining",
            packed,
        )

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
        self.assertIn('for slurm_command in sbatch scontrol squeue sacct srun; do', submit)
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
        preflight = packed.index(
            'FFMPEG_PREFLIGHT="$PREFLIGHT_ARTIFACT_DIR/trainer_ffmpeg_preflight.txt"'
        )
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

    def test_hsg_nested_renderer_and_trainer_shells_have_all_variables(self):
        packed = (ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch").read_text()
        submit = (ROOT / "scripts/scaling/submit_hsg_gb200_smoke.sh").read_text()

        # Variables in sbatch's sanitized --export list enter the batch as
        # environment variables.  Variables derived later must be explicitly
        # exported before a nested srun shell can consume them.
        export_lines = "\n".join(
            line for line in submit.splitlines() if line.startswith("EXPORTS=")
        )
        submitted = set(
            re.findall(
                r"(?:^|,)([A-Z][A-Z0-9_]*)=",
                export_lines.replace('EXPORTS="', "").replace('"', ""),
            )
        )

        def audit_nested_shell(marker: str, terminator: str) -> set[str]:
            launch = packed.index(marker)
            body_start = launch + len(marker)
            body = packed[body_start : packed.index(terminator, body_start)]
            references = set(
                re.findall(r"\$(?:\{)?([A-Za-z_][A-Za-z0-9_]*)", body)
            )
            # Arithmetic expansion permits bare variable names.
            for expression in re.findall(r"\$\(\((.*?)\)\)", body, re.DOTALL):
                references.update(
                    re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression)
                )

            exported_before_launch: set[str] = set()
            for statement in re.findall(
                r"(?m)^\s*export\s+([^\n#]+)$", packed[:launch]
            ):
                for token in statement.split():
                    match = re.match(
                        r"([A-Za-z_][A-Za-z0-9_]*)(?:=|$)", token
                    )
                    if match:
                        exported_before_launch.add(match.group(1))

            locally_defined = set(
                re.findall(
                    r"(?m)^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", body
                )
            )
            slurm_injected = {"SLURM_PROCID", "SLURM_JOB_ID", "SLURMD_NODENAME"}
            available = (
                submitted | exported_before_launch | locally_defined | slurm_injected
            )
            return references - available

        blocks = {
            "renderer": (
                "    bash -lc '\n",
                "\n    ' &",
            ),
            "trainer": (
                "        /usr/local/bin/roborender-trainer-sm100 bash -lc '\n",
                "\n        ' &",
            ),
        }
        for name, (marker, terminator) in blocks.items():
            with self.subTest(shell=name):
                self.assertEqual(set(), audit_nested_shell(marker, terminator))


if __name__ == "__main__":
    unittest.main()
