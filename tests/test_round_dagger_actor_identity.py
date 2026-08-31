"""Unit gates for independent round-actor process identities."""

from __future__ import annotations

import unittest

from round_dagger.actor_identity import (
    ActorIdentityError,
    ActorProcessIdentity,
    actor_process_identity,
    renderer_slot,
)


class ActorProcessIdentityTest(unittest.TestCase):
    def test_absent_identity_is_not_inferred_from_torch_rank(self) -> None:
        self.assertIsNone(
            actor_process_identity({"RANK": "7", "WORLD_SIZE": "8", "LOCAL_RANK": "3"})
        )

    def test_complete_identity_is_exact(self) -> None:
        self.assertEqual(
            actor_process_identity(
                {
                    "ROUND_DAGGER_ACTOR_RANK": "1",
                    "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
                }
            ),
            ActorProcessIdentity(rank=1, world_size=2),
        )

    def test_partial_noncanonical_and_out_of_range_values_fail(self) -> None:
        cases = (
            {"ROUND_DAGGER_ACTOR_RANK": "0"},
            {"ROUND_DAGGER_ACTOR_WORLD_SIZE": "2"},
            {
                "ROUND_DAGGER_ACTOR_RANK": "01",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            },
            {
                "ROUND_DAGGER_ACTOR_RANK": "2",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            },
            {
                "ROUND_DAGGER_ACTOR_RANK": "0",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "0",
            },
        )
        for environment in cases:
            with self.subTest(environment=environment), self.assertRaises(
                ActorIdentityError
            ):
                actor_process_identity(environment)

    def test_renderer_slots_cover_ordered_global_actor_environments(self) -> None:
        self.assertEqual(
            [
                renderer_slot(
                    rank=rank,
                    local_env_count=3,
                    local_env_index=local,
                    renderer_count=4,
                )
                for rank in range(2)
                for local in range(3)
            ],
            [0, 1, 2, 3, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()
