"""No Docker/GPU calls: regression of the actual successor cap-change caller."""
import json

import pytest

from pipeline.deployment.rtx_actor import resident_pool as pool_module
from pipeline.deployment.rtx_actor.resident_memory_admission import GIB


def setup(monkeypatch, *, available=25, anon=10, cache=10):
    samples = [{'container': f'owned-{i}', 'memory_current': (anon + cache + 1) * GIB,
                'memory_swap.current': 0, 'events': {'max': 0, 'oom': 0, 'oom_kill': 0},
                'stat': {'anon': anon * GIB, 'shmem': GIB, 'file': cache * GIB, 'kernel': GIB}}
               for i in range(2)]
    pool = {'workers': [{'container_name': s['container']} for s in samples]}
    calls = []
    monkeypatch.setattr(pool_module, 'quiescent', lambda p: True)
    monkeypatch.setattr(pool_module, 'container_memory_sample', lambda name: next(s for s in samples if s['container'] == name))
    monkeypatch.setattr(pool_module, 'memory_kib', lambda: {'MemAvailable': available * GIB // 1024})
    monkeypatch.setattr(pool_module.subprocess, 'check_output', lambda argv: json.dumps([
        {'Config': {'Labels': {pool_module.LABEL: argv[-1]}}}]).encode())
    monkeypatch.setattr(pool_module.subprocess, 'run', lambda argv, **kw: calls.append(argv))
    return pool, samples, calls


def test_cache_rich_host_does_not_authorize_cap_raise(monkeypatch):
    pool, _, calls = setup(monkeypatch)
    with pytest.raises(RuntimeError, match='admission'):
        pool_module.evaluation_memory(pool, True)
    assert not calls


def test_admitted_cap_raise_is_owned_and_reports_exact_budget(monkeypatch):
    pool, _, calls = setup(monkeypatch, available=40)
    report = pool_module.evaluation_memory(pool, True)
    assert report['admitted'] and report['projected_headroom_bytes'] == 13 * GIB
    assert report['file_cache_credit_bytes'] == 0
    assert calls == [['docker', 'update', '--memory', '28g', '--memory-swap', '28g', 'owned-0']]


def test_idle_barrier_required_before_sampling_or_update(monkeypatch):
    pool, _, calls = setup(monkeypatch, available=40)
    monkeypatch.setattr(pool_module, 'quiescent', lambda p: False)
    with pytest.raises(RuntimeError, match='idle'):
        pool_module.evaluation_memory(pool, True)
    assert not calls


@pytest.mark.parametrize('change', [
    {'stat': {}}, {'memory_swap.current': GIB}, {'events': {}}, {'oom_killed': True},
])
def test_unhealthy_or_incomplete_sample_never_changes_cap(monkeypatch, change):
    pool, samples, calls = setup(monkeypatch, available=40)
    samples[0].update(change)
    with pytest.raises(ValueError):
        pool_module.evaluation_memory(pool, True)
    assert not calls


def test_lower_cap_also_respects_committed_private_margin(monkeypatch):
    pool, _, calls = setup(monkeypatch, available=60, anon=20)
    with pytest.raises(ValueError, match='margin'):
        pool_module.evaluation_memory(pool, False)
    assert not calls


def test_owner_label_mismatch_cannot_change_cap(monkeypatch):
    pool, _, calls = setup(monkeypatch, available=40)
    monkeypatch.setattr(pool_module.subprocess, 'check_output', lambda argv: b'[{"Config":{"Labels":{}}}]')
    with pytest.raises(RuntimeError, match='owner mismatch'):
        pool_module.evaluation_memory(pool, True)
    assert not calls
