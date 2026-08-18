import ast
import unittest

from integrations.gr00t.patch_trainer_scaling import (
    MARKER,
    RUNTIME_GATES_MARKER,
    TELEMETRY_MARKER,
    PatchError,
    patch_source,
)


REPRESENTATIVE = '''
class Trainer:
    def __init__(self):
        self._id_env_clients = None
        self._id_gen_ms = []

    def _image_dagger_relabel(self, obs_dict, cur_dones):
        if self._id_anchors is None:
            self._id_anchors = [None] * B
            self._id_prompts = [None] * B
            for b in range(B):
                self._id_sample_anchor(b)
        if cur_dones is not None:
            done_idx = torch.nonzero(cur_dones.reshape(-1)).reshape(-1).tolist()
            for b in done_idx:
                if b < B:
                    self._id_sample_anchor(b)

        from concurrent.futures import ThreadPoolExecutor
        reqs = []
        def _render(b):
            depth_d, mask_d, prev_d = reqs[b]
            return self._id_env_clients[b].render_frame(
                depth_d, mask_d, prev_d, prompt=self._id_prompts[b], seed=self.vg_seed)
        with ThreadPoolExecutor(max_workers=B) as pool:
            gens = list(pool.map(_render, range(B)))
'''


class TrainerPatchTest(unittest.TestCase):
    def test_patches_known_parallel_trainer(self):
        patched = patch_source(REPRESENTATIVE)
        ast.parse(patched)
        self.assertIn(MARKER, patched)
        self.assertIn(RUNTIME_GATES_MARKER, patched)
        self.assertIn(TELEMETRY_MARKER, patched)
        self.assertIn("self._id_episode_ids[b] += 1", patched)
        self.assertIn("episode_id=int(self._id_episode_ids[b])", patched)
        self.assertIn("self._id_episode_steps += 1", patched)
        self.assertIn("IMAGE_DAGGER_EXPECT_PRIMARY_VIEW", patched)
        self.assertIn("IMAGE_DAGGER_EXPECT_TEACHER_RATIO", patched)
        self.assertIn("IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF", patched)
        self.assertIn("IMAGE_DAGGER_EXPECT_ENTROPY_COEF", patched)
        self.assertIn("trainer_step=int(self._id_telemetry_step)", patched)
        self.assertIn("depth=reqs[0][0], mask=reqs[0][1]", patched)
        self.assertIn("client_anchor=reqs[0][2]", patched)
        self.assertIn("DaggerTelemetry.from_environment(rank=self._id_rank)", patched)
        self.assertEqual(patch_source(patched), patched)

    def test_upgrades_existing_scaling_patch_with_runtime_gates(self):
        lines = patch_source(REPRESENTATIVE).splitlines()
        gate_start = next(
            i for i, line in enumerate(lines) if RUNTIME_GATES_MARKER in line
        )
        gate_end = next(
            i
            for i, line in enumerate(lines[gate_start:], start=gate_start)
            if "FAST_ROBORENDER_DAGGER_TELEMETRY_V1" in line
        )
        legacy = "\n".join(lines[:gate_start] + lines[gate_end:]) + "\n"
        self.assertIn(MARKER, legacy)
        self.assertNotIn(RUNTIME_GATES_MARKER, legacy)

        upgraded = patch_source(legacy)
        ast.parse(upgraded)
        self.assertIn(RUNTIME_GATES_MARKER, upgraded)

    def test_refuses_unknown_source(self):
        with self.assertRaises(PatchError):
            patch_source("def train(): pass")

    def test_refuses_partial_marker_only_telemetry_patch(self):
        patched = patch_source(REPRESENTATIVE)
        corrupted = patched.replace("self._id_telemetry.record(", "self.telemetry_record(")
        with self.assertRaisesRegex(PatchError, "invalid DAgger telemetry integration"):
            patch_source(corrupted)

    def test_refuses_partial_marker_only_runtime_gate_patch(self):
        patched = patch_source(REPRESENTATIVE)
        corrupted = patched.replace(
            "production gate failed: primary view mismatch",
            "production gate failed: removed assertion",
        )
        with self.assertRaisesRegex(PatchError, "invalid production runtime gates"):
            patch_source(corrupted)

    def test_refuses_partial_marker_only_scaling_patch(self):
        patched = patch_source(REPRESENTATIVE)
        corrupted = patched.replace(
            "self._id_episode_ids[b] += 1", "self._id_episode_ids[b] += 2"
        )
        with self.assertRaisesRegex(PatchError, "invalid scaling integration"):
            patch_source(corrupted)

    def test_upgrades_existing_scaling_and_runtime_patch_with_telemetry(self):
        patched = patch_source(REPRESENTATIVE)
        legacy = patched.replace(
            "        # FAST_ROBORENDER_DAGGER_TELEMETRY_V1\n"
            "        from gr00t.rl.utils.dagger_telemetry import DaggerTelemetry\n"
            "        self._id_telemetry_step = 0\n"
            "        self._id_telemetry = DaggerTelemetry.from_environment(rank=self._id_rank)\n",
            "",
        ).replace(
            "                reference_lag=self.vg_reference_lag,\n"
            "                trainer_step=int(self._id_telemetry_step))",
            "                reference_lag=self.vg_reference_lag)",
        )
        telemetry_start = legacy.index("            if self._id_telemetry is not None:\n")
        advance_start = legacy.index(
            "            # Advance only after every environment response succeeded.\n",
            telemetry_start,
        )
        legacy = legacy[:telemetry_start] + legacy[advance_start:]
        legacy = legacy.replace("            self._id_telemetry_step += 1", "")
        self.assertIn(MARKER, legacy)
        self.assertIn(RUNTIME_GATES_MARKER, legacy)
        self.assertNotIn(TELEMETRY_MARKER, legacy)

        upgraded = patch_source(legacy)
        ast.parse(upgraded)
        self.assertIn(TELEMETRY_MARKER, upgraded)
        self.assertEqual(patch_source(upgraded), upgraded)


if __name__ == "__main__":
    unittest.main()
