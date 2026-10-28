"""A failed second peer cannot hide behind an indefinitely waiting first peer."""
from threading import Event
import time

import pytest

from pipeline.deployment.rtx_actor.production_resident_pool import observe_commands
from pipeline.deployment.rtx_actor.resident_isaac_client import ResidentClient


def test_second_peer_failure_detaches_first_observer_without_native_mutation():
    entered = Event()
    clients = {i: object.__new__(ResidentClient) for i in (0, 1)}
    def action(row):
        if row['gpu'] == 0:
            entered.set()
            while True:
                clients[0]._observer_sleep(10)
        assert entered.wait(2)
        raise RuntimeError('native peer failed')
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='native peer failed'):
        observe_commands([dict(gpu=0), dict(gpu=1)], clients, action)
    assert time.monotonic() - start < 2
    assert all(c.observer_cancel is None for c in clients.values())


def test_completed_results_preserve_lane_order():
    clients = {i: object.__new__(ResidentClient) for i in (0, 1)}
    second = Event()
    def action(row):
        if row['gpu'] == 0:
            assert second.wait(2)
        else:
            second.set()
        return row['gpu']
    assert observe_commands([dict(gpu=0), dict(gpu=1)], clients, action) == [0, 1]
