import unittest

from scripts.scaling.scale_plan import build_plan


class ScalePlanTest(unittest.TestCase):
    def test_production_defaults(self):
        plan = build_plan()
        self.assertEqual(plan.global_envs, 256)
        self.assertEqual(plan.trainer_nodes, 8)
        self.assertEqual(plan.local_envs, 32)
        self.assertEqual(plan.rollout_steps, 300)
        self.assertEqual(plan.transitions_per_update, 76_800)
        self.assertEqual(plan.generated_envs_capacity, 256)
        self.assertEqual(plan.generated_requests_per_update_capacity, 76_800)
        self.assertEqual(plan.required_renderers, 52)
        self.assertEqual(plan.local_mini_batch, 8)
        self.assertEqual(plan.gradient_accumulation_steps, 8)

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


if __name__ == "__main__":
    unittest.main()
