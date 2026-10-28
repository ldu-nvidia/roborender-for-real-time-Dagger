from __future__ import annotations

from collections import Counter

import pytest

from dagger_rtx.contracts import ContractError
from dagger_rtx.lean_client_v4 import ShardedLeanH200BatchClientV4


class _Gateway:
    def __init__(self, index: int) -> None:
        self.index = index
        self.prepare_groups: list[list[str]] = []
        self.exchange_commit_groups: list[list[str]] = []
        self.exchange_prepare_groups: list[list[str]] = []
        self.finalize_groups: list[list[str]] = []
        self.closed = False

    @staticmethod
    def _actors(rows):
        return [row["step_key"]["actor_id"] for row in rows]

    def prepare_batch(self, local_prepares, idempotency_keys):
        assert len(local_prepares) == len(idempotency_keys)
        actors = self._actors(local_prepares)
        self.prepare_groups.append(actors)
        return [
            {"step_key": row["step_key"], "gateway": self.index}
            for row in local_prepares
        ]

    def exchange_batch(
        self,
        previous_selections,
        previous_local_commits,
        previous_commit_idempotency_keys,
        current_local_prepares,
        current_prepare_idempotency_keys,
    ):
        assert len(previous_selections) == len(previous_local_commits)
        assert len(previous_selections) == len(previous_commit_idempotency_keys)
        assert len(current_local_prepares) == len(current_prepare_idempotency_keys)
        self.exchange_commit_groups.append(self._actors(previous_local_commits))
        self.exchange_prepare_groups.append(self._actors(current_local_prepares))
        acknowledgements = [
            {"step_key": row["step_key"], "gateway": self.index}
            for row in previous_local_commits
        ]
        selections = [
            {"step_key": row["step_key"], "gateway": self.index}
            for row in current_local_prepares
        ]
        return acknowledgements, selections

    def finalize_batch(self, selections, local_commits, idempotency_keys):
        assert len(selections) == len(local_commits) == len(idempotency_keys)
        self.finalize_groups.append(self._actors(local_commits))
        return [
            {"step_key": row["step_key"], "gateway": self.index}
            for row in local_commits
        ]

    def close(self):
        self.closed = True


def _rows(
    sources: list[str], *, step: int = 0, window: str = "window-000000"
):
    return [
        {
            "step_key": {
                "actor_id": f"cost-aware-lane-{lane:03d}",
                "rollout_window_id": window,
                "rollout_step_index": step,
            },
            "metadata": {"observation_source": source},
        }
        for lane, source in enumerate(sources)
    ]


def _idempotency(prefix: str, count: int):
    return [f"{prefix}-{lane:03d}" for lane in range(count)]


def test_cost_aware_mapping_spreads_generated_lanes_and_stays_pinned():
    gateways = [_Gateway(index) for index in range(8)]
    client = ShardedLeanH200BatchClientV4(
        gateways, cost_aware_sharding=True
    )
    # Deliberately cluster every generated lane on odd modulo shards.  Legacy
    # i%8 placement has generated counts [0,4,0,4,0,3,0,3].
    sources0 = [
        "raytraced" if lane % 2 == 0 else "generated_controls"
        for lane in range(28)
    ]
    step0 = _rows(sources0)
    selections0 = client.prepare_batch(step0, _idempotency("prepare0", 28))

    assignment = client.lane_to_gateway
    assert assignment is not None
    assert client.shard_mapping_mode == "cost_aware"
    generated_counts = Counter(
        assignment[lane]
        for lane, source in enumerate(sources0)
        if source == "generated_controls"
    )
    assert sorted(generated_counts.values()) == [1, 1, 2, 2, 2, 2, 2, 2]
    assert max(generated_counts.values()) == 2
    assert [row["step_key"]["actor_id"] for row in selections0] == [
        f"cost-aware-lane-{lane:03d}" for lane in range(28)
    ]

    # Source assignments can change on episode reset, but actors must remain on
    # their original gateway because H200's v4 window ledger is process-local.
    step1 = _rows(list(reversed(sources0)), step=1)
    commits0 = [dict(row) for row in step0]
    acknowledgements, selections1 = client.exchange_batch(
        selections0,
        commits0,
        _idempotency("commit0", 28),
        step1,
        _idempotency("prepare1", 28),
    )
    assert client.lane_to_gateway == assignment
    assert [row["step_key"]["actor_id"] for row in acknowledgements] == [
        f"cost-aware-lane-{lane:03d}" for lane in range(28)
    ]
    assert [row["step_key"]["actor_id"] for row in selections1] == [
        f"cost-aware-lane-{lane:03d}" for lane in range(28)
    ]
    assert [gateway.exchange_prepare_groups[0] for gateway in gateways] == [
        [
            f"cost-aware-lane-{lane:03d}"
            for lane, target in enumerate(assignment)
            if target == gateway
        ]
        for gateway in range(8)
    ]
    client.close()


def test_cost_aware_retry_is_identical_and_actor_reordering_fails_closed():
    gateways = [_Gateway(index) for index in range(4)]
    client = ShardedLeanH200BatchClientV4(
        gateways, cost_aware_sharding=True
    )
    sources = ["generated_controls"] * 6 + ["raytraced"] * 4
    step0 = _rows(sources)
    keys = _idempotency("prepare", len(step0))

    first = client.prepare_batch(step0, keys)
    pinned = client.lane_to_gateway
    second = client.prepare_batch(step0, keys)
    assert first == second
    assert client.lane_to_gateway == pinned
    assert all(
        gateway.prepare_groups[0] == gateway.prepare_groups[1]
        for gateway in gateways
    )

    reordered = list(step0)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    before = [len(gateway.prepare_groups) for gateway in gateways]
    with pytest.raises(ContractError, match="actor order changed"):
        client.prepare_batch(reordered, keys)
    assert [len(gateway.prepare_groups) for gateway in gateways] == before
    client.close()


def test_cost_aware_unknown_source_uses_deterministic_modulo_fallback():
    gateways = [_Gateway(index) for index in range(3)]
    client = ShardedLeanH200BatchClientV4(
        gateways, cost_aware_sharding=True
    )
    rows = _rows(
        ["generated_controls", "legacy-source", "raytraced", "generated_controls"]
    )
    client.prepare_batch(rows, _idempotency("prepare", len(rows)))
    assert client.shard_mapping_mode == "modulo_fallback"
    assert client.lane_to_gateway == (0, 1, 2, 0)
    assert [gateway.prepare_groups[0] for gateway in gateways] == [
        ["cost-aware-lane-000", "cost-aware-lane-003"],
        ["cost-aware-lane-001"],
        ["cost-aware-lane-002"],
    ]
    client.close()


def test_cost_aware_mapping_stays_process_pinned_after_window_finalize():
    gateways = [_Gateway(index) for index in range(4)]
    client = ShardedLeanH200BatchClientV4(
        gateways, cost_aware_sharding=True
    )
    first_sources = [
        "generated_controls" if lane < 5 else "raytraced" for lane in range(8)
    ]
    first = _rows(first_sources, window="window-000000")
    selections = client.prepare_batch(first, _idempotency("prepare0", 8))
    first_mapping = client.lane_to_gateway

    second_sources = [
        "raytraced",
        "generated_controls",
        "raytraced",
        "generated_controls",
        "raytraced",
        "generated_controls",
        "generated_controls",
        "generated_controls",
    ]
    second = _rows(second_sources, window="window-000001")
    with pytest.raises(ContractError, match="before the active window is finalized"):
        client.prepare_batch(second, _idempotency("prepare1", 8))

    client.finalize_batch(
        selections,
        [dict(row) for row in first],
        _idempotency("commit0", 8),
    )
    reordered = list(second)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ContractError, match="process-pinned gateway assignment"):
        client.prepare_batch(reordered, _idempotency("prepare1", 8))
    client.prepare_batch(second, _idempotency("prepare1", 8))
    assert client.lane_to_gateway == first_mapping
    assert client.shard_mapping_mode == "cost_aware"
    assert [gateway.prepare_groups[1] for gateway in gateways] == [
        [
            f"cost-aware-lane-{lane:03d}"
            for lane, target in enumerate(first_mapping)
            if target == gateway
        ]
        for gateway in range(4)
    ]
    client.close()


def test_identical_finalize_retry_reuses_pinned_gateways():
    gateways = [_Gateway(index) for index in range(3)]
    client = ShardedLeanH200BatchClientV4(
        gateways, cost_aware_sharding=True
    )
    rows = _rows(["generated_controls", "raytraced", "generated_controls"])
    selections = client.prepare_batch(rows, _idempotency("prepare", 3))
    commits = [dict(row) for row in rows]
    keys = _idempotency("commit", 3)

    first = client.finalize_batch(selections, commits, keys)
    second = client.finalize_batch(selections, commits, keys)
    assert second == first
    assert all(len(gateway.finalize_groups) == 2 for gateway in gateways)

    with pytest.raises(ContractError, match="differs from the pinned request"):
        client.finalize_batch(selections, commits, _idempotency("changed", 3))
    client.close()


def test_new_window_repin_requires_step_zero():
    gateways = [_Gateway(index) for index in range(2)]
    client = ShardedLeanH200BatchClientV4(
        gateways, cost_aware_sharding=True
    )
    first = _rows(["generated_controls", "raytraced"])
    selections = client.prepare_batch(first, _idempotency("prepare0", 2))
    client.finalize_batch(
        selections,
        [dict(row) for row in first],
        _idempotency("commit0", 2),
    )
    with pytest.raises(ContractError, match="not rollout step zero"):
        client.prepare_batch(
            _rows(
                ["raytraced", "generated_controls"],
                step=1,
                window="window-000001",
            ),
            _idempotency("prepare1", 2),
        )
    client.close()
