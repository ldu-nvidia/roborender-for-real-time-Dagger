import unittest

from scripts.scaling.scale_plan import build_plan


class ScalePlanTest(unittest.TestCase):
    def test_production_defaults(self):
        plan = build_plan()
        self.assertEqual(plan.global_envs, 256)
        self.assertEqual(plan.trainer_nodes, 8)
        self.assertEqual(plan.local_envs, 32)
        self.assertEqual(plan.rollout_steps, 300)
        self.assertEqual(plan.num_rollout_iterations, 1)
        self.assertEqual(plan.transitions_per_rollout_iteration, 76_800)
        self.assertEqual(plan.generated_envs_capacity, 256)
        self.assertEqual(
            plan.generated_requests_per_rollout_iteration_capacity, 76_800
        )
        self.assertEqual(plan.required_renderers, 52)
        self.assertEqual(plan.local_mini_batch, 8)
        self.assertEqual(plan.gradient_accumulation_steps, 8)
        self.assertEqual(plan.num_learning_epochs, 1)
        self.assertEqual(plan.optimizer_steps_per_rollout_iteration, 4)
        self.assertEqual(plan.transitions_per_optimizer_step, 19_200)
        self.assertEqual(plan.total_collected_transitions, 76_800)
        self.assertEqual(plan.total_optimizer_steps, 4)
        self.assertEqual(plan.behavior_source, "student")
        self.assertEqual(plan.rollout_with_teacher_num_steps, 0)
        self.assertFalse(plan.enforce_teacher_rollout)
        self.assertEqual(plan.teacher_rollout_ratio, 0.0)
        self.assertEqual(plan.primary_view, "head_color")

    def test_renderer_extremes(self):
        self.assertEqual(build_plan(raytraced_mix=0.0).required_renderers, 52)
        self.assertEqual(build_plan(raytraced_mix=1.0).required_renderers, 0)

    def test_global_envs_must_divide_ranks(self):
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            build_plan(global_envs=256, trainer_nodes=7)

    def test_local_envs_must_divide_mini_batches(self):
        with self.assertRaisesRegex(ValueError, "divide num_mini_batches"):
            build_plan(global_envs=256, trainer_nodes=8, num_mini_batches=5)

    def test_mix_validation(self):
        with self.assertRaisesRegex(ValueError, "raytraced_mix"):
            build_plan(raytraced_mix=1.01)

    def test_behavior_source_is_explicit_and_consistent(self):
        mixed = build_plan(behavior_source="mixed", teacher_rollout_ratio=0.25)
        self.assertTrue(mixed.enforce_teacher_rollout)
        self.assertEqual(mixed.teacher_rollout_ratio, 0.25)

        teacher = build_plan(behavior_source="teacher", teacher_rollout_ratio=1.0)
        self.assertTrue(teacher.enforce_teacher_rollout)

        invalid = (
            {"behavior_source": "student", "teacher_rollout_ratio": 0.5},
            {"behavior_source": "mixed", "teacher_rollout_ratio": 0.0},
            {"behavior_source": "mixed", "teacher_rollout_ratio": 1.0},
            {"behavior_source": "teacher", "teacher_rollout_ratio": 0.5},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "behavior requires"
            ):
                build_plan(**kwargs)

    def test_learning_epoch_accounting(self):
        plan = build_plan(num_learning_epochs=3, num_rollout_iterations=2)
        self.assertEqual(plan.optimizer_steps_per_rollout_iteration, 12)
        self.assertEqual(plan.transitions_per_optimizer_step, 19_200)
        self.assertEqual(plan.total_collected_transitions, 153_600)
        self.assertEqual(plan.total_optimizer_steps, 24)

    def test_primary_view_must_be_a_generated_view(self):
        with self.assertRaisesRegex(ValueError, "primary_view"):
            build_plan(primary_view="unknown")


if __name__ == "__main__":
    unittest.main()
