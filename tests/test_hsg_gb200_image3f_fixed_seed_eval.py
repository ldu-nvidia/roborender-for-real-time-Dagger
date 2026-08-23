from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import py_compile
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixed = load_module("fixed_seed_gate", ROOT / "scripts/eval/fixed_seed_gate.py")
sys.modules["fixed_seed_gate"] = fixed
gate = load_module(
    "hsg_gb200_image3f_fixed_seed_gate",
    ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py",
)
sys.modules["hsg_gb200_image3f_fixed_seed_gate"] = gate
wandb_upload = load_module(
    "hsg_gb200_image3f_fixed_seed_wandb",
    ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_wandb.py",
)


def write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_rgb_png(path: Path, width: int = 1248, height: int = 720) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + b"\x00\x00\x00" * width
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanline * height, level=9))
        + chunk(b"IEND", b"")
    )


class HsgGb200Image3fFixedSeedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run_dir = self.base / "train"
        self.baseline = self.run_dir / "initial_policy.pt"
        self.baseline_hash = write(self.baseline, b"initial")
        for step in (5, 10, 15):
            write(self.run_dir / f"model_step_{step:06d}.pt", f"step-{step}".encode())
        self.config = self.run_dir / "config.yaml"
        self.config_hash = write(
            self.config,
            b"policy_view_count: 3\ninit_at_random_ep_len: true\n",
        )
        self.manifest = self.base / "eval/manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_image3f_production_eval_requires_a_sealed_source_snapshot(self):
        self.assertIn(
            "image3f_fixed_seed_production_eval",
            gate.smoke_gate.SEALED_SOURCE_PROFILES,
        )

    def test_prepare_pins_image3f_three_view_sequential_topology(self):
        payload = gate.prepare(
            argparse.Namespace(
                train_run_dir=self.run_dir,
                baseline_checkpoint=self.baseline,
                reference_config=self.config,
                seeds="11,29",
                num_envs=32,
                expected_baseline_sha256=self.baseline_hash,
                expected_reference_config_sha256=self.config_hash,
                output=self.manifest,
            )
        )
        self.assertEqual(payload["observation_mode"], gate.OBSERVATION_MODE)
        self.assertTrue(payload["image3f_in_the_loop"])
        self.assertEqual(payload["topology"]["nodes"], 2)
        self.assertEqual(payload["topology"]["renderer_services"], 7)
        self.assertEqual(payload["topology"]["evaluator_gpus"], 1)
        self.assertEqual(payload["runtime_contract"]["policy_views"], list(gate.VIEWS))
        self.assertEqual(payload["runtime_contract"]["raytraced_mix_prob"], 0.0)
        self.assertFalse(payload["runtime_contract"]["init_at_random_ep_len"])
        validated = gate.validate_manifest(
            self.manifest,
            train_run_dir=self.run_dir,
            expected_seeds=(11, 29),
            expected_num_envs=32,
        )
        self.assertEqual(validated["effective_contract_sha256"], payload["effective_contract_sha256"])
        payload["runtime_contract"]["raytraced_mix_prob"] = 0.1
        self.manifest.write_text(json.dumps(payload) + "\n")
        with self.assertRaisesRegex(gate.GateError, "contract"):
            gate.validate_manifest(
                self.manifest,
                train_run_dir=self.run_dir,
                expected_seeds=(11, 29),
                expected_num_envs=32,
            )

    def test_camera_acceptance_is_bound_to_exact_source_and_asset_hashes(self):
        fast_revision = "1" * 40
        gr00t_revision = "2" * 40
        base = {
            "fast_roborender_source_tree_sha256": "a" * 64,
            "gr00t_source_tree_sha256": "b" * 64,
            "source_snapshot_attestation_sha256": "c" * 64,
            "source_snapshot_inventory_sha256": "d" * 64,
            "asset_contract_sha256": "e" * 64,
        }
        panels = [
            self.base / "step000_env0.png",
            self.base / "step001_env0.png",
        ]
        for panel in panels:
            write_rgb_png(panel)
        technical = self.base / "camera_calibration_manifest.json"
        technical.write_text(
            json.dumps(
                {
                    "schema": gate.camera_gate.TECHNICAL_SCHEMA,
                    "status": gate.camera_gate.TECHNICAL_STATUS,
                    "run_id": "calibration_3view_fixture",
                    "fast_roborender_revision": fast_revision,
                    "gr00t_revision": gr00t_revision,
                    "fast_roborender_source_tree_sha256": base[
                        "fast_roborender_source_tree_sha256"
                    ],
                    "gr00t_source_tree_sha256": base[
                        "gr00t_source_tree_sha256"
                    ],
                    "source_snapshot_attestation_sha256": base[
                        "source_snapshot_attestation_sha256"
                    ],
                    "source_snapshot_inventory_sha256": base[
                        "source_snapshot_inventory_sha256"
                    ],
                    "immutable_asset_contract_sha256": base[
                        "asset_contract_sha256"
                    ],
                    "policy_views": list(gate.VIEWS),
                    "columns": list(gate.VIEWS),
                    "rows": list(gate.camera_gate.EXPECTED_ROWS),
                    "tile_shape_hwc": list(gate.camera_gate.EXPECTED_TILE_SHAPE_HWC),
                    "panel_shape_hwc": list(
                        gate.camera_gate.EXPECTED_PANEL_SHAPE_HWC
                    ),
                    "panels": [
                        {
                            "path": panel.name,
                            "sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
                        }
                        for panel in panels
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        calibration_run_id = "calibration_3view_fixture"
        acceptance = (
            self.base
            / f"camera_calibration_acceptance_{calibration_run_id}.json"
        )
        gate.camera_gate.accept_manifest(
            technical,
            acceptance,
            "reviewer@example.com",
            gate.camera_gate.CONFIRMATION,
            fast_revision,
            gr00t_revision,
            base["fast_roborender_source_tree_sha256"],
            base["gr00t_source_tree_sha256"],
            base["source_snapshot_attestation_sha256"],
            base["source_snapshot_inventory_sha256"],
            base["asset_contract_sha256"],
        )
        args = argparse.Namespace(
            camera_acceptance=acceptance,
            camera_acceptance_sha256=hashlib.sha256(
                acceptance.read_bytes()
            ).hexdigest(),
            camera_calibration_run_id=calibration_run_id,
            fast_revision=fast_revision,
            gr00t_revision=gr00t_revision,
        )
        verified = gate.verify_camera_acceptance_binding(args, base)
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["run_id"], calibration_run_id)

        args.camera_calibration_run_id = "different_calibration"
        with self.assertRaisesRegex(gate.camera_gate.GateError, "run_id mismatch"):
            gate.verify_camera_acceptance_binding(args, base)
        args.camera_calibration_run_id = calibration_run_id

        mutated = dict(base)
        mutated["fast_roborender_source_tree_sha256"] = "d" * 64
        with self.assertRaisesRegex(
            gate.camera_gate.GateError,
            "fast_roborender_source_tree_sha256 mismatch",
        ):
            gate.verify_camera_acceptance_binding(args, mutated)

    def _probe_report(self, requests, *, errors=None):
        rows = []
        errors = errors or [0] * 7
        identity = {
            "schema_version": 1,
            "release_id": "release",
            "adapter_sha256": "a" * 64,
            "generation": {
                "steps": 5,
                "tea_cache_threshold": 0.3,
                "cfg_scale": 1.0,
                "cuda_graph": False,
            },
            "engine": {"sha256": "e" * 64},
        }
        for index in range(7):
            rows.append(
                {
                    "file": f"/fleet/renderer_{index:03d}.txt",
                    "host": f"host{index}",
                    "port": 10000 + index,
                    "response": {
                        "server_id": f"eval:{index:03d}",
                        "runtime_identity": identity,
                        "metrics": {
                            "requests": requests[index],
                            "errors": errors[index],
                            "replays": 0,
                        },
                    },
                }
            )
        return {"selected": rows, "capable_count": 7, "expected": 7}

    def _metrics(self):
        conditions = {str(i): hashlib.sha256(f"c{i}".encode()).hexdigest() for i in range(4)}
        generated = {str(i): hashlib.sha256(f"g{i}".encode()).hexdigest() for i in range(4)}
        episodes = []
        for env_index in range(4):
            episodes.append(
                {
                    "env_index": env_index,
                    "seed": 11,
                    "success": env_index % 2 == 0,
                    "reward": float(env_index),
                    "episode_length": env_index + 1,
                    "object_class": "cube",
                    "first_condition_sha256": conditions[str(env_index)],
                    "first_generated_sha256": generated[str(env_index)],
                    "scenario_id": hashlib.sha256(f"s{env_index}".encode()).hexdigest(),
                }
            )
        return {
            "collected_episodes": 4,
            "goal_reached_rate": 0.5,
            "reward": 1.5,
            "episode_length": 2.5,
            "_roborender_eval_adapter": {"schema_version": 1},
            "_roborender_image3f_eval": {
                "schema": "fast_roborender.image3f_fixed_eval_runtime.v1",
                "eval_relabel_enabled": True,
                "raytraced_mix_prob": 0.0,
                "reference_lag": 0,
                "generation_seed": 11,
                "policy_views": list(gate.VIEWS),
                "generated_policy_steps": 2,
                "generated_frames": 8,
                "first_condition_sha256_by_env": conditions,
                "first_generated_sha256_by_env": generated,
            },
            "_roborender_fixed_seed_episodes": episodes,
        }

    def _attest_args(self):
        before = self.base / "before.json"
        after = self.base / "after.json"
        metrics = self.base / "metrics.json"
        endpoint_manifest = self.base / "selected_endpoints.txt"
        before.write_text(json.dumps(self._probe_report([0] * 7)) + "\n")
        after.write_text(json.dumps(self._probe_report([2, 1, 1, 1, 1, 1, 1])) + "\n")
        metrics.write_text(json.dumps(self._metrics()) + "\n")
        endpoint_manifest.write_text(
            "".join(f"/fleet/renderer_{index:03d}.txt\n" for index in range(7))
        )
        return argparse.Namespace(
            before=before,
            after=after,
            metrics=metrics,
            endpoint_manifest=endpoint_manifest,
            label="baseline",
            seed=11,
            run_id="eval_baseline_seed11",
            num_envs=4,
            expected_renderers=7,
            output=self.base / "attestation.json",
        )

    def test_runtime_attestation_requires_every_renderer_and_policy_frame(self):
        args = self._attest_args()
        report = gate.attest(args)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["served_requests"], 8)
        self.assertEqual(report["generated_frames"], 8)
        self.assertEqual(len(report["scenario_ids"]), 4)

        bad = self._probe_report([2, 1, 1, 1, 1, 1, 1], errors=[1, 0, 0, 0, 0, 0, 0])
        args.after.write_text(json.dumps(bad) + "\n")
        with self.assertRaisesRegex(gate.GateError, "errors or lag-0 replays"):
            gate.attest(args)

        args = self._attest_args()
        metrics = self._metrics()
        metrics["_roborender_image3f_eval"]["eval_relabel_enabled"] = False
        args.metrics.write_text(json.dumps(metrics) + "\n")
        with self.assertRaisesRegex(gate.GateError, "eval_relabel_enabled"):
            gate.attest(args)

    def test_launchers_are_executable_syntax_valid_and_fail_closed(self):
        paths = (
            ROOT / "scripts/eval/submit_hsg_gb200_image3f_fixed_seed_gate.sh",
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_policy_eval.sbatch",
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_task.sh",
        )
        for path in paths:
            self.assertTrue(os.access(path, os.X_OK), path)
            parsed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, f"{path}: {parsed.stderr}")
        for path in (
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py",
            ROOT / "scripts/eval/hsg_gb200_image3f_fixed_seed_wandb.py",
        ):
            self.assertTrue(os.access(path, os.X_OK), path)
            py_compile.compile(str(path), doraise=True)

        submit, packed, task = (path.read_text() for path in paths)
        self.assertIn("--nodes=2", submit)
        self.assertIn("--gpus-per-node=4", submit)
        self.assertIn("--time=48:00:00", submit)
        self.assertNotIn("camera_calibration_acceptance.json", submit)
        self.assertIn(
            "CAMERA_CALIBRATION_ACCEPTANCE must explicitly select a versioned",
            submit,
        )
        self.assertIn(
            'camera_calibration_acceptance_${CAMERA_CALIBRATION_RUN_ID}.json',
            submit,
        )
        self.assertIn('--expected-run-id "$CAMERA_CALIBRATION_RUN_ID"', submit)
        self.assertIn(
            "CAMERA_CALIBRATION_RUN_ID=$CAMERA_CALIBRATION_RUN_ID,",
            submit,
        )
        self.assertIn("#SBATCH --partition=batch_long", packed)
        self.assertIn("CAMERA_CALIBRATION_RUN_ID", packed)
        self.assertIn('--expected-run-id "$CAMERA_CALIBRATION_RUN_ID"', packed)
        self.assertIn(
            '--camera-calibration-run-id "$CAMERA_CALIBRATION_RUN_ID"',
            submit,
        )
        self.assertIn(
            '--camera-calibration-run-id "$CAMERA_CALIBRATION_RUN_ID"',
            packed,
        )
        self.assertIn("--ntasks=7", packed)
        self.assertIn("--nodelist=\"$EVAL_NODE\" --gpus=1", packed)
        self.assertIn("RENDER_SHUTDOWN_TIMEOUT_S=180", packed)
        self.assertIn("shutdown_renderer()", packed)
        self.assertIn("--expected-host-counts \"$PLACEMENT_PLAN\"", packed)
        self.assertIn("--expected-engine-sha256 \"$EXPECTED_IMAGE3F_ENGINE_SHA256\"", packed)
        self.assertNotIn('rm -f "$CURRENT_PREFLIGHT"', packed)
        self.assertIn("for TASK_INDEX in 0 1 2 3", task)
        self.assertIn("IMAGE_DAGGER_EVAL_RELABEL=1", task)
        self.assertIn("algo.config.videogen.raytraced_mix_prob=0.0", task)
        self.assertIn("algo.config.init_at_random_ep_len=false", task)
        self.assertIn("head_color,hand_left,hand_right", task)
        self.assertIn('IMAGE3F_RUN_ID="${EVAL_RUN_ID}_${LABEL}_seed${SEED}"', task)
        self.assertIn('--endpoint-manifest "$ENDPOINT_MANIFEST"', task)
        self.assertNotIn("WANDB_API_KEY", submit + packed + task)

    def test_fixed_seed_submit_rejects_missing_or_nonversioned_acceptance_early(self):
        launcher = ROOT / "scripts/eval/submit_hsg_gb200_image3f_fixed_seed_gate.sh"
        stage_root = self.base / "stage"
        base_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "STAGE_ROOT": str(stage_root),
            "TRAIN_RUN_DIR": str(self.run_dir),
            "SLURM_BIN_DIR": str(self.base / "missing-slurm"),
        }

        cases = (
            ({}, "must explicitly select a versioned"),
            (
                {
                    "CAMERA_CALIBRATION_ACCEPTANCE": str(
                        stage_root
                        / "runtime/validation/camera_calibration_acceptance.json"
                    )
                },
                "must name camera_calibration_acceptance_<run-id>.json",
            ),
            (
                {
                    "CAMERA_CALIBRATION_ACCEPTANCE": str(
                        stage_root
                        / "runtime/validation/"
                        "camera_calibration_acceptance_bad.run.json"
                    )
                },
                "malformed run ID",
            ),
            (
                {
                    "CAMERA_CALIBRATION_ACCEPTANCE": str(
                        stage_root
                        / "runtime/validation/"
                        "camera_calibration_acceptance_calibration_good.json"
                    ),
                    "CAMERA_CALIBRATION_RUN_ID": "different_calibration",
                },
                "does not match the selected acceptance filename",
            ),
            (
                {
                    "CAMERA_CALIBRATION_ACCEPTANCE": str(
                        self.base
                        / "other/camera_calibration_acceptance_calibration_good.json"
                    )
                },
                "must be the exact versioned validation artifact",
            ),
        )
        for extra_env, message in cases:
            with self.subTest(message=message):
                result = subprocess.run(
                    ["bash", str(launcher)],
                    env={**base_env, **extra_env},
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)

        accepted = subprocess.run(
            ["bash", str(launcher)],
            env={
                **base_env,
                "CAMERA_CALIBRATION_ACCEPTANCE": str(
                    stage_root
                    / "runtime/validation/"
                    "camera_calibration_acceptance_calibration_good.json"
                ),
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(accepted.returncode, 2, accepted.stderr)
        self.assertIn("missing HSG Slurm command", accepted.stderr)
        self.assertNotIn("camera calibration acceptance", accepted.stderr.lower())

    def test_runbook_standalone_workflows_select_versioned_acceptance(self):
        runbook = (ROOT / "docs/HSG_GB200_PRODUCTION_RUNBOOK.md").read_text(
            encoding="utf-8"
        )
        headings = (
            "## 3. Validate submission without consuming the logical run ID",
            "## 4. Submit a fresh production run",
            "## 5. Resume after an interrupted allocation",
            "## 6. Validation-only recovery at step 500",
            "## 7. Run the primary fixed-seed policy-improvement gate",
        )
        for index, heading in enumerate(headings):
            start = runbook.index(heading)
            end = (
                runbook.index(headings[index + 1], start)
                if index + 1 < len(headings)
                else runbook.index("## 8. Expected durable evidence", start)
            )
            section = runbook[start:end]
            with self.subTest(heading=heading):
                self.assertIn(
                    "export CAMERA_CALIBRATION_RUN_ID=<accepted-calibration-run-id>",
                    section,
                )
                self.assertIn(
                    "camera_calibration_acceptance_${CAMERA_CALIBRATION_RUN_ID}.json",
                    section,
                )
        fixed_seed_section = runbook[
            runbook.index(headings[-1]) : runbook.index(
                "## 8. Expected durable evidence"
            )
        ]
        self.assertIn(
            'bash "$ROOT/scripts/eval/submit_hsg_gb200_image3f_fixed_seed_gate.sh"',
            fixed_seed_section,
        )

    def test_wandb_readback_requires_image3f_and_policy_evidence(self):
        rows = [
            {
                "eval/checkpoint_ordinal": ordinal,
                "eval/checkpoint_label": label,
                "eval/goal_reached_rate": 0.1 * ordinal,
                "eval/reward": float(ordinal),
                "eval/episodes": 256,
                "image3f/served_requests": 1024,
                "image3f/generated_frames": 1024,
            }
            for ordinal, label in enumerate(fixed.ROLE_ORDER)
        ]
        summary = {
            "policy_improvement_gate/status": "PASS",
            "policy_improvement_gate/paired_success_delta": 0.2,
            "policy_improvement_gate/paired_success_delta_ci95_low": 0.1,
            "policy_improvement_gate/paired_success_delta_ci95_high": 0.3,
            "image3f_gate/status": "PASS",
            "image3f_gate/total_served_requests": 4096,
        }
        report = wandb_upload.validate_readback(rows, summary)
        self.assertEqual(report["observed_checkpoint_ordinals"], [0, 1, 2, 3])
        rows[-1].pop("image3f/generated_frames")
        with self.assertRaisesRegex(wandb_upload.UploadError, "readback incomplete"):
            wandb_upload.validate_readback(rows, summary)

    def test_nested_trainer_evidence_hash_and_return_contract(self):
        videogen_path = (
            ROOT
            / "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
        )
        homie_path = (
            ROOT
            / "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py"
        )
        tree = ast.parse(videogen_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_image3f_payload_sha256"
        )
        namespace = {"hashlib": hashlib, "np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(videogen_path), "exec"), namespace)
        hasher = namespace["_image3f_payload_sha256"]
        groups = (("depth", {view: np.zeros((2, 3), np.uint8) for view in gate.VIEWS}),)
        first = hasher(views=gate.VIEWS, groups=groups, prompt="pick", seed=11)
        second = hasher(views=gate.VIEWS, groups=groups, prompt="pick", seed=11)
        changed = hasher(views=gate.VIEWS, groups=groups, prompt="pick", seed=12)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

        homie_source = homie_path.read_text(encoding="utf-8")
        videogen_source = videogen_path.read_text(encoding="utf-8")
        self.assertIn('eval_dict["_roborender_fixed_seed_episodes"]', homie_source)
        self.assertIn("return eval_dict", homie_source)
        self.assertIn("_image3f_eval_generated_frames += len(gens)", videogen_source)
        self.assertIn('"first_condition_sha256_by_env"', videogen_source)


if __name__ == "__main__":
    unittest.main()
