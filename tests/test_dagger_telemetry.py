import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from integrations.gr00t import dagger_telemetry as telemetry
from integrations.gr00t.patch_trainer_scaling import patch_source
from scripts.scaling import install_gr00t_integration as installer
from scripts.scaling import wandb_run_watcher
from tests.test_trainer_patch import REPRESENTATIVE


VIEWS = ("ext1", "ext2", "wrist")


def triplet(value, *, height=4, width=6, channels=3):
    shape = (height, width) if channels == 1 else (height, width, channels)
    return {view: np.full(shape, value + index, np.uint8) for index, view in enumerate(VIEWS)}


class DaggerTelemetryTest(unittest.TestCase):
    def test_pure_bc_config_requires_exact_numeric_zero(self):
        self.assertEqual(installer.yaml_numeric_scalar("value_loss_coef: 0\n", "value_loss_coef"), 0.0)
        self.assertEqual(installer.yaml_numeric_scalar("value_loss_coef: -0.0 # pure BC\n", "value_loss_coef"), -0.0)
        self.assertEqual(installer.yaml_numeric_scalar("value_loss_coef: 0.1\n", "value_loss_coef"), 0.1)
        self.assertIsNone(
            installer.yaml_numeric_scalar(
                "value_loss_coef: 0\nvalue_loss_coef: 0\n", "value_loss_coef"
            )
        )
        self.assertIsNone(installer.yaml_numeric_scalar("value_loss_coef: '0'\n", "value_loss_coef"))

    def test_contact_sheet_preserves_per_view_resolution_and_has_four_rows(self):
        sheet = telemetry.build_contact_sheet(
            triplet(10, channels=1),
            triplet(20),
            triplet(30),
            triplet(40),
            view_order=VIEWS,
        )
        self.assertEqual(sheet.shape, (4 * (4 + telemetry.LABEL_HEIGHT) + 6, 3 * 6 + 4, 3))
        # The first generated tile retains its exact source pixels beneath its label.
        generated_y = 3 * (4 + telemetry.LABEL_HEIGHT + telemetry.GRID_GAP)
        np.testing.assert_array_equal(
            sheet[generated_y + telemetry.LABEL_HEIGHT : generated_y + telemetry.LABEL_HEIGHT + 4, :6],
            triplet(40)["ext1"],
        )

    def test_atomic_names_finite_scalars_and_dependency_safe_video_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            producer = telemetry.DaggerTelemetry(
                temporary,
                image_every=1,
                clip_every=2,
                clip_frames=3,
                clip_fps=7,
                ffmpeg_path="",
            )
            common = {
                "depth": triplet(1, channels=1),
                "mask": triplet(2),
                "client_anchor": triplet(3),
                "generated": triplet(4),
            }
            first = producer.record(
                trainer_step=0,
                client_metadata={
                    "client_wall_ms": 12.5,
                    "nested": {"server_ms": np.float32(8.25)},
                    "bad_nan": float("nan"),
                    "bad_inf": float("inf"),
                    "reference_source": "anchor",
                    "reference_provenance_valid": True,
                    "reference_provenance": {
                        "schema": "image3f.reference_provenance.v1",
                        "request_step": 0,
                        "configured_lag": 256,
                        "kind": "anchor",
                        "source_step": 0,
                    },
                },
                **common,
            )
            second = producer.record(trainer_step=1, client_metadata={}, **common)

            self.assertEqual(first["image"].name, "step_000000000_conditions_generated.png")
            self.assertEqual(first["scalar"].name, "step_000000000.json")
            self.assertEqual(second["video"].name, "step_000000001_generated_triplet.npz")
            self.assertFalse(any((Path(temporary) / "telemetry" / "videos").glob("*.mp4")))
            self.assertFalse(any(path.name.startswith(".") for path in Path(temporary).rglob("*")))

            document = json.loads(first["scalar"].read_text(encoding="utf-8"))
            self.assertEqual(document["step"], 0)
            self.assertEqual(document["metrics"]["renderer/client_wall_ms"], 12.5)
            self.assertAlmostEqual(document["metrics"]["renderer/nested/server_ms"], 8.25)
            self.assertNotIn("renderer/bad_nan", document["metrics"])
            self.assertNotIn("renderer/bad_inf", document["metrics"])
            self.assertNotIn("renderer/reference_source", document["metrics"])
            self.assertEqual(document["metrics"]["renderer/reference_source_anchor"], 1)
            self.assertEqual(document["metrics"]["renderer/reference_source_generated_lag"], 0)
            self.assertEqual(document["metrics"]["renderer/reference_provenance_valid"], 1)
            self.assertEqual(
                document["metrics"]["renderer/reference_provenance/request_step"], 0
            )
            self.assertEqual(
                document["metrics"]["renderer/reference_provenance/source_step"], 0
            )
            self.assertEqual(document["metrics"]["rollout_frame_step"], 0)
            self.assertEqual(document["metrics"]["policy_update_index"], 0)
            json.dumps(document, allow_nan=False)
            step, watcher_metrics = wandb_run_watcher.load_scalar_file(first["scalar"])
            self.assertEqual(step, 0)
            self.assertEqual(watcher_metrics["renderer/client_wall_ms"], 12.5)

            png = first["image"].read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack("!II", png[16:24])
            self.assertEqual((width, height), (3 * 6 + 4, 4 * (4 + telemetry.LABEL_HEIGHT) + 6))

            with np.load(second["video"], allow_pickle=False) as bundle:
                self.assertEqual(bundle["frames"].shape, (2, 4 + telemetry.LABEL_HEIGHT, 3 * 6 + 4, 3))
                self.assertEqual(int(bundle["fps"]), 7)
            capability = json.loads(
                (Path(temporary) / "telemetry" / "video_capability.json").read_text()
            )
            self.assertFalse(capability["available"])
            self.assertIn("disabled", capability["reason"])

    def test_scalar_cadence_rollout_step_update_index_and_reference_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            producer = telemetry.DaggerTelemetry(
                temporary,
                scalar_every=2,
                image_every=999,
                clip_every=0,
                rollout_steps=4,
                ffmpeg_path="",
            )
            common = {
                "depth": triplet(1, channels=1),
                "mask": triplet(2),
                "client_anchor": triplet(3),
                "generated": triplet(4),
            }
            outputs = [
                producer.record(
                    trainer_step=step,
                    client_metadata={"reference_source": "generated_t-256"},
                    **common,
                )
                for step in range(5)
            ]
            self.assertEqual(
                [step for step, output in enumerate(outputs) if "scalar" in output],
                [0, 2, 4],
            )
            document = json.loads(outputs[4]["scalar"].read_text(encoding="utf-8"))
            self.assertEqual(document["metrics"]["rollout_frame_step"], 0)
            self.assertEqual(document["metrics"]["policy_update_index"], 1)
            self.assertEqual(document["metrics"]["renderer/reference_source_generated_lag"], 1)
            self.assertEqual(document["metrics"]["renderer/reference_lag_frames"], 256)
            self.assertEqual(document["metrics"]["renderer/reference_source_anchor"], 0)
            self.assertEqual(document["metrics"]["rendered_frame_count"], 4)

    def test_required_video_fails_instead_of_silent_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "required DAgger MP4 telemetry"):
                telemetry.DaggerTelemetry(
                    temporary,
                    ffmpeg_path="",
                    require_video=True,
                )

    def test_environment_factory_is_rank_zero_and_run_dir_gated(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {
                "TRAIN_RUN_DIR": temporary,
                "ROBORENDER_TELEMETRY_DIR": str(Path(temporary) / "telemetry"),
            },
            clear=True,
        ):
            self.assertIsNone(telemetry.DaggerTelemetry.from_environment(rank=1))
            producer = telemetry.DaggerTelemetry.from_environment(rank=0)
            self.assertIsNotNone(producer)
            self.assertEqual(producer.scalar_every, 10)
            self.assertEqual(producer.image_every, 300)
            self.assertEqual(producer.clip_every, 300)
            self.assertEqual(producer.clip_frames, 32)
            os.environ["ROBORENDER_TELEMETRY_DIR"] = str(Path(temporary) / "wrong")
            with self.assertRaisesRegex(ValueError, "TRAIN_RUN_DIR/telemetry"):
                telemetry.DaggerTelemetry.from_environment(rank=0)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(telemetry.DaggerTelemetry.from_environment(rank=0))

    def test_installer_hash_verifies_telemetry_utility(self):
        repo_root = Path(__file__).resolve().parents[1]
        client_source = repo_root / "integrations/gr00t/image3f_frame_client.py"
        telemetry_source = repo_root / "integrations/gr00t/dagger_telemetry.py"
        with tempfile.TemporaryDirectory() as temporary:
            gr00t = Path(temporary)
            target = gr00t / "gr00t/rl/utils/dagger_telemetry.py"
            trainer = gr00t / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
            trainer.parent.mkdir(parents=True)
            trainer.write_text(patch_source(REPRESENTATIVE), encoding="utf-8")
            installer.install_file(telemetry_source, target)
            report = installer.inspect(gr00t, client_source, telemetry_source)
            self.assertTrue(report["trainer_telemetry_integrated"])
            self.assertEqual(report["missing_trainer_telemetry_markers"], [])
            self.assertTrue(report["telemetry_utility_exists"])
            self.assertTrue(report["telemetry_utility_matches"])
            self.assertEqual(
                report["telemetry_source_sha256"], report["telemetry_installed_sha256"]
            )
            target.write_text("modified\n", encoding="utf-8")
            report = installer.inspect(gr00t, client_source, telemetry_source)
            self.assertFalse(report["telemetry_utility_matches"])

    def test_production_launcher_pins_bounded_telemetry_cadence(self):
        repo_root = Path(__file__).resolve().parents[1]
        launcher = (repo_root / "scripts/train/dagger_scale_256x300.sbatch").read_text(
            encoding="utf-8"
        )
        for assignment in (
            "IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY=${IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY:-10}",
            "IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY=${IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY:-300}",
            "IMAGE_DAGGER_TELEMETRY_CLIP_EVERY=${IMAGE_DAGGER_TELEMETRY_CLIP_EVERY:-300}",
            "IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES=${IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES:-32}",
            "IMAGE_DAGGER_TELEMETRY_REQUIRED=1",
            "IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO=1",
        ):
            self.assertIn(assignment, launcher)


if __name__ == "__main__":
    unittest.main()
