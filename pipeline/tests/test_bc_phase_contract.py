"""Focused BC collection/rendering contract tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from bc_collection.materialize_raven_bc_chunks import (
    CHUNK_T,
    VIEWS,
    materialize_environment,
)
from bc_rendering.render_bc_teacher_worker import (
    RAVEN_STUDENT_RESOLUTION,
    VIEW_SLOTS,
    next_wan_frame_count,
    validate_completed_chunk,
    write_raven_rendered_chunk,
)
from bc_rendering import render_bc_teacher_8gpu as planner


class BcPhaseContractTest(unittest.TestCase):
    def test_native_stream_materializes_exact_raven_chunks(self) -> None:
        steps = CHUNK_T
        expert = np.arange(steps * 26, dtype=np.float32).reshape(steps, 1, 26)
        arrays = {
            "expert_actions": expert,
            "behavior_actions": expert.copy(),
            "expert_mask": np.ones((steps, 1), dtype=np.bool_),
            "dones": np.zeros((steps, 1), dtype=np.bool_),
            "global_env_ids": np.full((steps, 1), 17, dtype=np.int64),
            "episode_ids": np.full((steps, 1), 4, dtype=np.int64),
            "episode_steps": np.arange(steps, dtype=np.int64)[:, None],
            "transition_image_indices": np.arange(steps, dtype=np.int64)[:, None],
            "obs__actor_obs": np.zeros((steps, 1, 113), dtype=np.float32),
        }
        arrays["dones"][-1, 0] = True
        for view in VIEWS:
            arrays[f"depth_rawq__{view}"] = np.zeros(
                (steps, 480, 832), dtype=np.uint16
            )
            arrays[f"depth_raw__{view}"] = np.zeros(
                (steps, 240, 416), dtype=np.float16
            )
            arrays[f"depth_ctrl__{view}"] = np.zeros(
                (steps, 240, 416), dtype=np.uint8
            )
            arrays[f"rgb_robot__{view}"] = np.zeros(
                (steps, 240, 416, 3), dtype=np.uint8
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            np.savez_compressed(source, **arrays)
            output = root / "chunks"
            output.mkdir()
            items, next_index, discarded = materialize_environment(
                source_path=source,
                prompt="robot task",
                output_dir=output,
                chunk_index=0,
            )
            self.assertEqual((len(items), next_index, discarded), (1, 1, 0))
            with np.load(output / items[0]["path"], allow_pickle=False) as chunk:
                self.assertEqual(chunk["gt_actions"].shape, (81, 26))
                self.assertEqual(chunk["obs_actor_obs"].shape, (81, 113))
                for view in VIEWS:
                    self.assertEqual(chunk[f"depth_ctrl__{view}"].shape, (81, 240, 416))
                    self.assertEqual(chunk[f"rgb_robot__{view}"].shape, (81, 240, 416, 3))

            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "roborender.raven_bc_chunks.v1",
                        "status": "sealed",
                        "chunk_length": 81,
                        "items": items,
                    }
                )
            )
            rendered = root / "rendered"
            with mock.patch.object(planner, "SOURCE_ROOT", output), mock.patch.object(
                planner, "OUTPUT_ROOT", rendered
            ):
                jobs = planner.discover_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertTrue(all(job["frame_count"] == 81 for job in jobs))
            self.assertTrue(all("chunk_nativeh200_" in job["output_npz"] for job in jobs))
            self.assertTrue(all(job["seed"] == 42 for job in jobs))
            expected_prompt = planner.RAVEN_RENDER_PROMPT.read_text().strip()
            self.assertTrue(all(job["prompt"] == expected_prompt for job in jobs))

    def test_offline_renderer_uses_raven_view_mapping_and_81_frames(self) -> None:
        self.assertEqual(
            VIEW_SLOTS,
            (("ext1", "hand_left"), ("ext2", "hand_right"), ("wrist", "head_color")),
        )
        self.assertEqual(next_wan_frame_count(81), 81)

    def test_rendered_chunk_is_the_exact_raven_learner_archive(self) -> None:
        frame_count = 3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            output = root / "rendered.npz"
            gt = np.arange(frame_count * 26, dtype=np.float32).reshape(frame_count, 26)
            actor_obs = np.arange(frame_count * 113, dtype=np.float32).reshape(
                frame_count, 113
            )
            np.savez_compressed(
                source,
                gt_actions=gt,
                obs_actor_obs=actor_obs,
                dones=np.zeros(frame_count, dtype=np.bool_),
                prompt=np.asarray("prompt"),
                env_id=np.asarray(4),
                seq=np.asarray(2),
            )
            by_camera = {
                view: [
                    Image.new(
                        "RGB", (416, 240), color=(index, 2 * index, 3 * index)
                    )
                    for index in range(frame_count)
                ]
                for _, view in VIEW_SLOTS
            }
            write_raven_rendered_chunk(source, output, by_camera)
            self.assertTrue(validate_completed_chunk(output, frame_count))
            with np.load(output, allow_pickle=False) as rendered:
                self.assertTrue(np.array_equal(rendered["gt_actions"], gt))
                self.assertTrue(np.array_equal(rendered["obs_actor_obs"], actor_obs))
                expected_shape = (frame_count, *RAVEN_STUDENT_RESOLUTION, 3)
                self.assertEqual(rendered["rgb_gen"].shape, expected_shape)
                self.assertEqual(rendered["rgb_gen"].dtype, np.uint8)
                self.assertTrue(
                    np.array_equal(rendered["rgb_gen"], rendered["rgb_gen__head_color"])
                )
                for _, view in VIEW_SLOTS:
                    self.assertEqual(rendered[f"rgb_gen__{view}"].shape, expected_shape)
                    self.assertEqual(rendered[f"rgb_gen__{view}"].dtype, np.uint8)

    def test_bc_training_config_pins_raven_scientific_parameters(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = (
            root
            / "pipeline/GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/"
            "wsdpt_videogen_offline_bc_3view.yaml"
        ).read_text(encoding="utf-8")
        base = (
            root
            / "pipeline/GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/"
            "wsdpt_student_for_teacher_v8q8.002_resnet_rgb_delay.yaml"
        ).read_text(encoding="utf-8")
        algo = (
            root
            / "pipeline/GR00T-VisualSim2Real/gr00t/rl/config/algo/"
            "dagger_vision_distributed.yaml"
        ).read_text(encoding="utf-8")
        for exact in (
            "policy_views: [head_color, hand_left, hand_right]",
            "camera_resolutions: [108, 192]",
            "num_mini_batches: 1",
            "num_steps_per_env: 81",
            "value_loss_coef: 0.0",
            "entropy_coef: 0.0",
            "seed: 42",
            "rollout_action_source: teacher",
            "policy_view_count: 3",
        ):
            self.assertIn(exact, config)
        for inherited in (
            "actor_learning_rate: 2e-4",
            "resnet_type: resnet18",
            "pretrained: true",
            "hidden_dims: [512, 256, 128]",
            "activation: SiLU",
            "obj_pred_loss_coef: 0.0",
            "use_cosine_scheduler: false",
        ):
            self.assertIn(inherited, base)
        for inherited in (
            "num_learning_epochs: 1",
            "dagger_bc_loss_type: l2",
            "dagger_bc_loss_coef: 1.0",
            "max_grad_norm: 1.0",
        ):
            self.assertIn(inherited, algo)


if __name__ == "__main__":
    unittest.main()
