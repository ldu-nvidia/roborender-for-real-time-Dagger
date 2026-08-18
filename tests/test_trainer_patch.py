import ast
import unittest

from integrations.gr00t.patch_trainer_scaling import MARKER, PatchError, patch_source


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
        def _render(b):
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
        self.assertIn("self._id_episode_ids[b] += 1", patched)
        self.assertIn("episode_id=int(self._id_episode_ids[b])", patched)
        self.assertIn("self._id_episode_steps += 1", patched)
        self.assertEqual(patch_source(patched), patched)

    def test_refuses_unknown_source(self):
        with self.assertRaises(PatchError):
            patch_source("def train(): pass")


if __name__ == "__main__":
    unittest.main()
