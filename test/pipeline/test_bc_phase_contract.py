"""Focused BC collection/rendering contract tests."""
from test.deployment.source_paths import REPO_ROOT

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from bc_training.create_bc_splits import create_splits
from bc_training.train_bc_h200 import (
    BCChunks,
    EarlyStopping,
    infer_resume_validation_state,
    load_resume_state,
)
from bc_rendering.render_bc_teacher_worker import (
    RAVEN_STUDENT_RESOLUTION,
    VIEW_SLOTS,
    next_wan_frame_count,
    validate_completed_chunk,
    write_raven_rendered_chunk,
)
from bc_rendering.validate_bc_teacher_render import (
    validate_chunk,
)
from bc_rendering import render_bc_teacher_8gpu as planner


class BcPhaseContractTest(unittest.TestCase):

    def test_split_indexes_are_exact_and_stream_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            # Twenty equal collection streams give an exact 80/10/10 split.
            for stream in range(20):
                for index in range(10):
                    chunk_id = f"bc_chunk_{stream * 10 + index:06d}"
                    rows.append({
                        "chunk_id": chunk_id,
                        "isaac_npz": f"isaac/{chunk_id}.npz",
                        "rendered_npz": f"rendered/{chunk_id}.npz",
                        "source_run_tag": "run",
                        "source_batch": 1,
                        "rank": stream % 2,
                        "local_env_id": stream,
                    })
            (root / "manifest.json").write_text(json.dumps({
                "frames_per_chunk": 81,
                "chunks": rows,
            }))
            create_splits(root)
            documents = {
                name: json.loads((root / "splits" / f"{name}.json").read_text())
                for name in ("train", "validation", "test")
            }
            self.assertEqual(
                {name: doc["chunk_count"] for name, doc in documents.items()},
                {"train": 160, "validation": 20, "test": 20},
            )
            chunk_sets = {
                name: {item["chunk_id"] for item in doc["items"]}
                for name, doc in documents.items()
            }
            stream_sets = {
                name: {item["stream_id"] for item in doc["items"]}
                for name, doc in documents.items()
            }
            for left, right in (("train", "validation"), ("train", "test"),
                                ("validation", "test")):
                self.assertFalse(chunk_sets[left] & chunk_sets[right])
                self.assertFalse(stream_sets[left] & stream_sets[right])


    def test_offline_renderer_uses_raven_view_mapping_and_81_frames(self) -> None:
        self.assertEqual(
            VIEW_SLOTS,
            (("ext1", "hand_left"), ("ext2", "hand_right"), ("wrist", "head_color")),
        )
        self.assertEqual(next_wan_frame_count(81), 81)

    def test_990_chunks_are_count_balanced_across_eight_gpus(self) -> None:
        jobs = [
            {
                "frame_count": 81,
                "global_env_id": index % 256,
                "episode_id": index,
            }
            for index in range(990)
        ]
        assignments, totals = planner.assign_lpt(jobs, 8)
        self.assertEqual(
            [len(items) for items in assignments],
            [124, 124, 124, 124, 124, 124, 123, 123],
        )
        self.assertEqual(sum(map(len, assignments)), 990)
        self.assertEqual(totals, [len(items) * 81 for items in assignments])

    def test_progress_aggregates_each_physical_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            lane_plans = []
            for gpu, assigned, done, rendered, skipped in (
                (0, 124, 7, 6, 1),
                (1, 123, 5, 5, 0),
            ):
                progress_path = run_dir / "progress" / f"gpu-{gpu:02d}-lane-00.json"
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(
                    json.dumps(
                        {
                            "status": "rendering",
                            "done": done,
                            "rendered_this_run": rendered,
                            "skipped_preexisting": skipped,
                        }
                    )
                )
                lane_plans.append(
                    {
                        "physical_gpu": gpu,
                        "lane": 0,
                        "jobs": [{}] * assigned,
                        "progress_path": str(progress_path),
                    }
                )

            snapshot = planner.publish_progress(
                run_dir, lane_plans, status="running"
            )
            self.assertEqual(snapshot["assigned"], 247)
            self.assertEqual(snapshot["done"], 12)
            self.assertEqual(snapshot["remaining"], 235)
            self.assertEqual(snapshot["rendered_this_run"], 11)
            self.assertEqual(snapshot["skipped_preexisting"], 1)
            self.assertEqual(snapshot["gpus"][0]["remaining"], 117)
            self.assertEqual(snapshot["gpus"][1]["remaining"], 118)
            text = (run_dir / "progress.txt").read_text()
            self.assertIn("GPU0 7/124 remaining=117", text)
            self.assertIn("GPU1 5/123 remaining=118", text)

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

    def test_final_render_validator_checks_npz_pairing(self) -> None:
        frame_count = 81
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            output = root / "rendered.npz"
            np.savez_compressed(
                source,
                gt_actions=np.zeros((frame_count, 26), dtype=np.float32),
                obs_actor_obs=np.zeros((frame_count, 113), dtype=np.float32),
                obs_teacher_obs=np.zeros((frame_count, 226), dtype=np.float32),
                dones=np.zeros(frame_count, dtype=np.bool_),
                prompt=np.asarray("prompt"),
                env_id=np.asarray(4),
                seq=np.asarray(2),
            )
            by_camera = {
                view: [Image.new("RGB", (1, 1), color=(index, 0, 0)) for index in range(frame_count)]
                for _, view in VIEW_SLOTS
            }
            write_raven_rendered_chunk(source, output, by_camera)
            self.assertEqual(validate_chunk(source, output), [])

    def test_bc_training_config_pins_raven_scientific_parameters(self) -> None:
        root = REPO_ROOT
        materialized_raven = root / "pipeline/GR00T-VisualSim2Real/.git"
        if not materialized_raven.exists():
            self.skipTest("requires the ignored, pinned Raven checkout")
        launcher = (root / "pipeline/bc_training/train_raven_bc_h200.sh").read_text(
            encoding="utf-8"
        )
        trainer_source = (root / "pipeline/bc_training/train_bc_h200.py").read_text(
            encoding="utf-8"
        )
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
            "every_epochs: 1",
            "minimum_epochs: 10",
            "patience_validations: 8",
            "relative_min_delta: 0.005",
            "maximum_epochs: 100",
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
        for required in (
            "pipeline/bc_training/launcher_data.py",
            "REQUIRE_PRODUCTION_DATA_ADMISSION:-1",
            "WANDB_MODE",
            "g1_3view_student_bootstrap_v1.pt",
            "train_bc_h200.py",
        ):
            self.assertIn(required, launcher)
        self.assertNotIn("splits/test.json", launcher)
        for required in (
            '"train/mse": train_mse',
            '"validation/mse": validation_mse',
            '"policy_state_dict": policy.state_dict()',
            'args.validation_split.resolve(), "validation", args.dataset_root.resolve()',
            'backward_raven_bc_mse(policy, vision, actor, target)',
            'RAVEN_TRAIN_CHUNKS_PER_UPDATE = 16',
            'install_index_safe_max_pool2d(policy)',
            'drop_last=True',
            'trainable, args.max_grad_norm, error_if_nonfinite=True',
            'weight_decay=0.0',
            'validation_checkpoints / (',
            'update_checkpoint_alias(validation_checkpoint, output / "best_validation.pt")',
            'wandb.define_metric("validation/*", step_metric="epoch")',
            'parser.add_argument("--wandb-project", default="dagger")',
            'parser.add_argument("--wandb-run-name", default="BC")',
            'budget.add_argument("--max-epochs", type=int, default=100)',
            'budget.add_argument("--no-epoch-cap", dest="max_epochs", action="store_const", const=None',
        ):
            self.assertIn(required, trainer_source)
        self.assertNotIn("isaac", trainer_source.lower().replace("no isaac", ""))

    def test_native_bc_loader_uses_only_three_views_state_and_teacher_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunk.npz"
            arrays = {
                f"rgb_gen__{view}": np.full((3, 108, 192, 3), index, np.uint8)
                for index, view in enumerate(("head_color", "hand_left", "hand_right"))
            }
            arrays.update({
                "obs_actor_obs": np.ones((3, 113), np.float32),
                "gt_actions": np.arange(3 * 26, dtype=np.float32).reshape(3, 26),
                # Must be ignored by the learner.
                "obs_teacher_obs": np.ones((3, 226), np.float32),
                "depth_ctrl__head_color": np.ones((3, 240, 416), np.uint8),
            })
            np.savez_compressed(path, **arrays)
            item = BCChunks([path], actor_dim=113, action_dim=11)[0]
            self.assertEqual(tuple(item["vision"].shape), (3, 324, 192, 3))
            self.assertEqual(tuple(item["actor"].shape), (3, 113))
            self.assertEqual(tuple(item["target"].shape), (3, 11))
            self.assertNotIn("obs_teacher_obs", item)
            self.assertNotIn("depth_ctrl__head_color", item)

    def test_validation_stopping_obeys_delta_patience_and_minimum_epoch(self) -> None:
        stop = EarlyStopping(min_epochs=10, patience=8, relative_improvement=0.005)
        improved, saturated = stop.update(0, 1.0)
        self.assertTrue(improved)
        self.assertFalse(saturated)
        # A 0.4% decrease does not clear the required 0.5% relative delta.
        improved, saturated = stop.update(1, 0.996)
        self.assertFalse(improved)
        self.assertFalse(saturated)
        # Even eight stale checks cannot stop before the minimum epoch.
        for epoch in range(2, 9):
            _, saturated = stop.update(epoch, 0.996)
        self.assertFalse(saturated)
        _, saturated = stop.update(10, 0.996)
        self.assertTrue(saturated)

        # A meaningful improvement resets patience and replaces the best epoch.
        stop = EarlyStopping(min_epochs=1, patience=2, relative_improvement=0.005)
        stop.update(0, 1.0)
        improved, saturated = stop.update(1, 0.99)
        self.assertTrue(improved)
        self.assertFalse(saturated)
        self.assertEqual(stop.best_epoch, 1)
        self.assertEqual(stop.bad_validations, 0)

    def test_bc_resume_restores_model_optimizer_epoch_and_history(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prior"
            source.mkdir()
            checkpoint_path = source / "best_validation.pt"
            state_path = source / "validation_state.json"
            spec = {"policy": "tiny-test"}

            original = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(original.parameters(), lr=2e-4, weight_decay=0.0)
            loss = original(torch.ones(1, 2)).square().mean()
            loss.backward()
            optimizer.step()
            torch.save({
                "policy_state_dict": original.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 2,
                "global_step": 8,
                "validation_mse": 0.5,
                "policy_spec": spec,
                "bootstrap_sha256": "bootstrap",
                "train_split_sha256": "train",
                "validation_split_sha256": "validation",
            }, checkpoint_path)
            state_path.write_text(json.dumps({"history": [
                {"epoch": 0, "global_step": 0, "mse": 1.0},
                {"epoch": 1, "global_step": 4, "mse": 0.8},
                {"epoch": 2, "global_step": 8, "mse": 0.5},
                # Must be discarded when resuming an older selected checkpoint.
                {"epoch": 3, "global_step": 12, "mse": 0.6},
            ]}))

            restored = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(
                restored.parameters(), lr=2e-4, weight_decay=0.0
            )
            generator = torch.Generator().manual_seed(42)
            stopper = EarlyStopping()
            checkpoint, history, rng_mode = load_resume_state(
                checkpoint_path,
                state_path,
                policy=restored,
                optimizer=restored_optimizer,
                train_generator=generator,
                spec=spec,
                bootstrap_sha256="bootstrap",
                train_split_sha256="train",
                validation_split_sha256="validation",
                stopper=stopper,
                seed=42,
                expected_learning_rate=2e-4,
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual([record["epoch"] for record in history], [0, 1, 2])
            self.assertEqual(rng_mode, "legacy_deterministic_epoch_seed")
            self.assertEqual(stopper.best_epoch, 2)
            self.assertTrue(restored_optimizer.state)
            for expected, actual in zip(original.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(infer_resume_validation_state(checkpoint_path), state_path)


if __name__ == "__main__":
    unittest.main()
