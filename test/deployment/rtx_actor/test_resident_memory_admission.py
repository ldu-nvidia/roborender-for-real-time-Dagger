import pytest
from pipeline.deployment.rtx_actor.resident_memory_admission import GIB, admit_future_caps


def owner(name='one', anon=10, shmem=1, file=10, kernel=1):
    return {'container': name, 'memory_current': (anon+file+kernel)*GIB,
            'memory_swap.current': 0, 'events': {'max': 0, 'oom': 0, 'oom_kill': 0},
            'stat': {'anon': anon*GIB, 'shmem': shmem*GIB, 'file': file*GIB, 'kernel': kernel*GIB}}


def test_file_cache_is_not_credited_twice():
    owners = [owner('one'), owner('two')]
    # The old formula credits all42GiB of owned current on top of available25,
    # incorrectly accepting future49GiB+8GiB. Actual anon/shmem credit is22.
    assert 25*GIB + sum(o['memory_current'] for o in owners) >= (49+8)*GIB
    result = admit_future_caps(25*GIB, owners, [28*GIB, 21*GIB])
    assert not result['admitted']
    assert result['projected_headroom_bytes'] == -2*GIB
    assert result['file_cache_credit_bytes'] == 0


def test_potential_smaller_idle_cap_requires_actual_margin():
    result = admit_future_caps(29*GIB, [owner('one'), owner('two')], [28*GIB, 14*GIB])
    assert result['admitted'] and result['projected_headroom_bytes'] == 9*GIB
    with pytest.raises(ValueError, match='margin'):
        admit_future_caps(29*GIB, [owner('one'), owner('two', anon=14)], [28*GIB,14*GIB])


def test_cache_growth_cannot_inflate_admission():
    low = admit_future_caps(25*GIB, [owner(file=5)], [28*GIB])
    high = admit_future_caps(25*GIB, [owner(file=20)], [28*GIB])
    assert low['admitted'] == high['admitted']
    assert low['projected_headroom_bytes'] == high['projected_headroom_bytes']


@pytest.mark.parametrize('mutation', [
    {'memory_swap.current': GIB}, {'oom_killed': True},
    {'sampling_error': 'unavailable'}, {'events': {'oom_kill': 1}},
    {'events': {}}, {'events': {'max': 1, 'oom': 0, 'oom_kill': 0}},
    {'events': {'max': 0, 'oom': 1, 'oom_kill': 0}},
    {'events': {'max': False, 'oom': 0, 'oom_kill': 0}},
    {'memory_current': 1},
])
def test_incomplete_or_unhealthy_attribution_fails_closed(mutation):
    with pytest.raises(ValueError):
        admit_future_caps(50*GIB, [{**owner(), **mutation}], [28*GIB])


def test_duplicate_owner_and_lower_floor_fail():
    with pytest.raises(ValueError):
        admit_future_caps(50*GIB, [owner(), owner()], [28*GIB,21*GIB])
    with pytest.raises(ValueError):
        admit_future_caps(50*GIB, [owner()], [28*GIB], floor_bytes=7*GIB)
