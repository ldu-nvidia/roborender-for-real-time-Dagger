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
        self.assertEqual(plan.local_mini_batch, 32)
        self.assertEqual(plan.gradient_accumulation_steps, 32)
        self.assertEqual(plan.num_learning_epochs, 1)
        self.assertEqual(plan.optimizer_steps_per_rollout_iteration, 1)
        self.assertEqual(plan.transitions_per_optimizer_step, 76_800)
        self.assertEqual(plan.total_collected_transitions, 76_800)
        self.assertEqual(plan.total_optimizer_steps, 1)
        self.assertEqual(plan.behavior_source, "annealed")
        self.assertEqual(plan.rollout_action_source, "beta")
        self.assertEqual(plan.beta_schedule, "linear")
        self.assertEqual(plan.beta_start, 1.0)
        self.assertEqual(plan.beta_end, 0.0)
        self.assertEqual(plan.beta_decay_iters, 500)
        self.assertEqual(plan.reference_policy, "fixed_anchor")
        self.assertEqual(plan.reference_lag, 0)
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
        self.assertFalse(mixed.enforce_teacher_rollout)
        self.assertEqual(mixed.teacher_rollout_ratio, 0.25)
        self.assertEqual(mixed.rollout_action_source, "beta")
        self.assertEqual(mixed.beta_schedule, "constant")
        self.assertEqual(mixed.beta_start, 0.25)
        self.assertEqual(mixed.beta_end, 0.25)

        teacher = build_plan(behavior_source="teacher", teacher_rollout_ratio=1.0)
        self.assertFalse(teacher.enforce_teacher_rollout)
        self.assertEqual(teacher.rollout_action_source, "teacher")

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

        annealed = build_plan(
            behavior_source="annealed",
            beta_schedule="linear",
            beta_start=1.0,
            beta_end=0.0,
            beta_decay_iters=250,
        )
        self.assertEqual(annealed.rollout_action_source, "beta")
        self.assertEqual(annealed.beta_decay_iters, 250)

    def test_reference_policy_is_explicit(self):
        self.assertEqual(build_plan(reference_lag=0).reference_policy, "fixed_anchor")
        lagged = build_plan(reference_lag=256)
        self.assertEqual(lagged.reference_policy, "generated_lag")
        with self.assertRaisesRegex(ValueError, "reference_lag"):
            build_plan(reference_lag=-1)

    def test_learning_epoch_accounting(self):
        plan = build_plan(num_learning_epochs=3, num_rollout_iterations=2)
        self.assertEqual(plan.optimizer_steps_per_rollout_iteration, 3)
        self.assertEqual(plan.transitions_per_optimizer_step, 76_800)
        self.assertEqual(plan.total_collected_transitions, 153_600)
        self.assertEqual(plan.total_optimizer_steps, 6)

    def test_primary_view_must_be_a_generated_view(self):
        with self.assertRaisesRegex(ValueError, "primary_view"):
            build_plan(primary_view="unknown")


if __name__ == "__main__":
    unittest.main()
