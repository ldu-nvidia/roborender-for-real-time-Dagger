import unittest

from scripts.data.offline_render_dataset import shard_for


class OfflineShardTest(unittest.TestCase):
    def test_all_chunks_from_env_stay_on_one_shard(self):
        first = shard_for("16222830", 17, 8)
        self.assertEqual(first, shard_for("16222830", 17, 8))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 8)

    def test_collector_tag_is_part_of_stream_identity(self):
        assignments = {shard_for(str(tag), 17, 16) for tag in range(32)}
        self.assertGreater(len(assignments), 1)


if __name__ == "__main__":
    unittest.main()
