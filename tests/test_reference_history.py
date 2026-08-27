import unittest

import numpy as np

from scripts.server.reference_history import LaggedReferenceStore, ReferenceHistoryError


def frame(value):
    return {"head_color": np.full((2, 3, 3), value, dtype=np.uint8)}


class LaggedReferenceStoreTest(unittest.TestCase):
    def test_exact_lag256_boundary_selection_and_digest_chain(self):
        store = LaggedReferenceStore()
        anchor = frame(99)
        observed = {}
        for step in range(258):
            reference, source, digest = store.prepare_with_provenance(
                stream_id="env0",
                episode_id="ep0",
                step=step,
                lag=256,
                anchor=anchor,
                request_id=f"request-{step}",
            )
            if step in {0, 1, 255, 256, 257}:
                observed[step] = (
                    source,
                    int(reference["head_color"][0, 0, 0]),
                    digest,
                )
            store.commit(
                stream_id="env0",
                episode_id="ep0",
                step=step,
                generated=frame(step % 256),
                request_id=f"request-{step}",
            )
        self.assertEqual(
            [(step, observed[step][0], observed[step][1]) for step in observed],
            [
                (0, "bootstrap_anchor", 99),
                (1, "generated_episode_frame0", 0),
                (255, "generated_episode_frame0", 0),
                (256, "generated_t-256", 0),
                (257, "generated_t-256", 1),
            ],
        )
        self.assertEqual(observed[1][2], store.image_mapping_sha256(frame(0)))
        self.assertEqual(observed[257][2], store.image_mapping_sha256(frame(1)))

    def test_lag_contract(self):
        store = LaggedReferenceStore(max_streams=2)
        anchor = frame(99)
        sources = []
        values = []
        for step in range(6):
            reference, source = store.prepare(
                stream_id="env0", episode_id="ep0", step=step, lag=3, anchor=anchor,
                request_id=f"request-{step}",
            )
            sources.append(source)
            values.append(int(reference["head_color"][0, 0, 0]))
            store.commit(
                stream_id="env0", episode_id="ep0", step=step, generated=frame(step),
                request_id=f"request-{step}",
            )
        self.assertEqual(
            sources,
            [
                "bootstrap_anchor",
                "generated_episode_frame0",
                "generated_episode_frame0",
                "generated_t-3",
                "generated_t-3",
                "generated_t-3",
            ],
        )
        self.assertEqual(values, [99, 0, 0, 0, 1, 2])
        self.assertEqual(store.stats()["frames"], 3)

    def test_episode_reset_requires_step_zero(self):
        store = LaggedReferenceStore()
        with self.assertRaisesRegex(ReferenceHistoryError, "begin at step 0"):
            store.prepare(
                stream_id="env0", episode_id=1, step=4, lag=3, anchor=frame(1),
                request_id="bad",
            )

    def test_abort_allows_same_step_retry(self):
        store = LaggedReferenceStore()
        store.prepare(
            stream_id="env0", episode_id=0, step=0, lag=3, anchor=frame(1),
            request_id="request-0",
        )
        store.abort(stream_id="env0", episode_id=0, step=0)
        reference, source = store.prepare(
            stream_id="env0", episode_id=0, step=0, lag=3, anchor=frame(1),
            request_id="request-0-retry",
        )
        self.assertEqual(source, "bootstrap_anchor")
        self.assertEqual(int(reference["head_color"][0, 0, 0]), 1)

    def test_rejects_out_of_order_and_parallel_same_stream(self):
        store = LaggedReferenceStore()
        store.prepare(
            stream_id="env0", episode_id=0, step=0, lag=3, anchor=frame(1),
            request_id="request-0",
        )
        with self.assertRaisesRegex(ReferenceHistoryError, "in flight"):
            store.prepare(
                stream_id="env0", episode_id=0, step=0, lag=3, anchor=frame(1),
                request_id="parallel",
            )
        store.abort(stream_id="env0", episode_id=0, step=0)
        with self.assertRaisesRegex(ReferenceHistoryError, "expected step 0"):
            store.prepare(
                stream_id="env0", episode_id=0, step=1, lag=3, anchor=frame(1),
                request_id="request-1",
            )

    def test_anchor_is_fixed_for_episode(self):
        store = LaggedReferenceStore()
        store.prepare(
            stream_id="env0", episode_id=0, step=0, lag=3, anchor=frame(1),
            request_id="request-0",
        )
        store.commit(
            stream_id="env0", episode_id=0, step=0, generated=frame(10),
            request_id="request-0",
        )
        with self.assertRaisesRegex(ReferenceHistoryError, "anchor changed"):
            store.prepare(
                stream_id="env0", episode_id=0, step=1, lag=3, anchor=frame(2),
                request_id="request-1",
            )

    def test_committed_request_can_be_replayed(self):
        store = LaggedReferenceStore()
        store.prepare(
            stream_id="env0", episode_id=0, step=0, lag=3, anchor=frame(1),
            request_id="request-0",
        )
        store.commit(
            stream_id="env0", episode_id=0, step=0, generated=frame(10),
            request_id="request-0",
        )
        replay = store.replay(
            stream_id="env0", episode_id=0, step=0, request_id="request-0"
        )
        self.assertIsNotNone(replay)
        self.assertEqual(int(replay["head_color"][0, 0, 0]), 10)
        replay_with_source = store.replay_with_source(
            stream_id="env0", episode_id=0, step=0, request_id="request-0"
        )
        self.assertIsNotNone(replay_with_source)
        self.assertEqual(replay_with_source[1], "bootstrap_anchor")
        replay_with_provenance = store.replay_with_provenance(
            stream_id="env0", episode_id=0, step=0, request_id="request-0"
        )
        self.assertEqual(
            replay_with_provenance[2], store.image_mapping_sha256(frame(1))
        )
        self.assertIsNone(
            store.replay(
                stream_id="env0", episode_id=0, step=0, request_id="different"
            )
        )


if __name__ == "__main__":
    unittest.main()
