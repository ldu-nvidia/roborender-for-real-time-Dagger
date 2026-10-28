"""CPU safety/boundary checks; never a native capacity certificate."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from pipeline.deployment.rtx_actor import paired_bc_guard as pair
from pipeline.deployment.rtx_actor.bounded_bc_guard import memory_gate
from test.pipeline.run_paired_bc_smoke import host_gate, actor_gate, paired_command, record_actor_sample

G = 2**30


def config():
    return dict(schema=pair.SCHEMA, num_envs_per_actor=32, windows_per_actor=4,
        host_reserve_bytes=4*G, actor_cap_bytes=43*G, seeds=[0, 1],
        request_sha256=['a'*64, 'b'*64], source_sha256='c'*64,
        guard_sha256=hashlib.sha256(Path(pair.__file__).read_bytes()).hexdigest(),
        production_admitted=False, retain_compute=True, replay_allowed=False)


def sample():
    return dict(pid=1, memory_current=20*G, memory_peak=22*G, memory_max=43*G,
        **{'memory_swap.current': 0}, events=dict(max=0, oom=0, oom_kill=0), oom_killed=False)


@pytest.mark.parametrize('key,value', [('num_envs_per_actor',40),('windows_per_actor',8),
    ('host_reserve_bytes',2*G),('production_admitted',True),('retain_compute',False),
    ('seeds',[0,0]),('request_sha256',['a'*64]),('guard_sha256','bad')])
def test_bound_scope_cannot_widen(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError):
        pair.validate_config(c)


def test_legacy_eight_gib_floor_unchanged_and_pair_accounts_peer():
    with pytest.raises(RuntimeError):
        memory_gate(sample(), 9*G, num_envs=32, cap_bytes=43*G)
    memory_gate(sample(), 8*G, num_envs=32, cap_bytes=43*G,
                host_reserve_bytes=4*G, peer_capture_bytes=2*G)
    with pytest.raises(RuntimeError):
        memory_gate(sample(), 8*G-1, num_envs=32, cap_bytes=43*G,
                    host_reserve_bytes=4*G, peer_capture_bytes=2*G)
    memory_gate(sample(), 12*G, num_envs=32, cap_bytes=43*G, saving=True,
                tail_growth_bytes=0, host_reserve_bytes=4*G)
    with pytest.raises(RuntimeError):
        memory_gate(sample(), 12*G-1, num_envs=32, cap_bytes=43*G, saving=True,
                    tail_growth_bytes=0, host_reserve_bytes=4*G)


@pytest.mark.parametrize('key', ['max', 'oom', 'oom_kill'])
def test_no_pressure_events_waived(key):
    s = sample()
    s['events'][key] = 1
    with pytest.raises(RuntimeError):
        actor_gate(s)


def test_host_floor_and_actor_swap_remain_enforced():
    host_gate(dict(MemAvailable=4*G//1024))
    with pytest.raises(RuntimeError):
        host_gate(dict(MemAvailable=4*G//1024-1))
    s = sample()
    s['memory_swap.current'] = 1
    with pytest.raises(RuntimeError):
        actor_gate(s)


@pytest.mark.parametrize('bad', [
    {'sampling_error': 'docker inspect timed out'},
    {**sample(), 'memory_swap.current': 4096},
    {**sample(), 'events': {'max': 1, 'oom': 0, 'oom_kill': 0}},
    {'pid': 1, 'memory_current': 123},
])
def test_rejected_observation_is_preserved_before_abort(bad):
    row = {'actors': []}
    with pytest.raises(RuntimeError) as exc:
        record_actor_sample(row, 'exact-owner', bad, {'used_bytes': 123}, enforce_gate=True)
    assert row['actors'][0]['container'] == bad
    assert json.dumps(bad, sort_keys=True) in str(exc.value)
    assert json.loads(json.dumps(row)) == row


def test_startup_observation_recorded_without_false_admission():
    row = {'actors': []}
    sample_before_docker_acceptance = {'sampling_error': 'not yet created'}
    record_actor_sample(row, 'exact-owner', sample_before_docker_acceptance, {}, enforce_gate=False)
    assert row['actors'][0]['container'] == sample_before_docker_acceptance


def test_shared_steps_exclusive_saves_and_failure_release(tmp_path):
    (tmp_path/'save.lock').touch()
    a, b = [pair.PairLease(tmp_path, i, config()) for i in (0,1)]
    try:
        a.acquire()
        b.acquire()
        with pytest.raises(BlockingIOError):
            with (tmp_path/'save.lock').open('a') as fd:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.release()
        b.release()
        with pytest.raises(ValueError, match='payload failed'):
            with a.saving():
                with pytest.raises(BlockingIOError):
                    fcntl.flock(b.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                raise ValueError('payload failed')
        assert not (tmp_path/'save-waiting0').exists()
        b.acquire()
        b.release()
        events = [json.loads(r)['phase'] for r in (tmp_path/'actor0-events.jsonl').read_text().splitlines()]
        assert events == ['save_begin','save_end']
        (tmp_path/'abort.json').write_text('{}')
        with pytest.raises(RuntimeError, match='aborted'):
            b.acquire()
    finally:
        a.close()
        b.close()


def test_pair_command_changes_only_runtime_guard_and_scope():
    original = ['docker','run','image','trainer._target_=bounded_bc_guard.BoundedContinuousBCTrainer',
                'num_envs=32','algo.config.num_steps_per_env=81']
    changed = paired_command(original, actor=0, stage='/stage', config_sha='a'*64, image='image')
    assert original[3].endswith('BoundedContinuousBCTrainer')
    assert changed[-2:] == original[-2:]
    assert 'trainer._target_=paired_bc_guard.PairedBCTrainer' in changed
    assert 'ROBORENDER_PAIR_ACTOR=0' in changed


def test_observation_hold_waits_without_reset_and_abort_takes_priority(tmp_path, monkeypatch):
    (tmp_path/'save.lock').touch()
    hold = tmp_path/'observation-hold.json'
    hold.write_text('{}')
    lease = pair.PairLease(tmp_path, 0, config())
    sleeps = []
    def sleep(_):
        sleeps.append(1)
        hold.unlink()
    monkeypatch.setattr(pair.time, 'sleep', sleep)
    try:
        lease.check()
        assert sleeps == [1] and not lease.held
        hold.write_text('{}')
        (tmp_path/'abort.json').write_text('{}')
        with pytest.raises(RuntimeError, match='aborted'):
            lease.check()
        assert sleeps == [1]
    finally:
        lease.close()


def test_lost_controller_cannot_leave_native_held_forever(tmp_path, monkeypatch):
    (tmp_path/'save.lock').touch()
    (tmp_path/'observation-hold.json').write_text('{}')
    lease = pair.PairLease(tmp_path, 0, config())
    ticks = iter([0, 31])
    monkeypatch.setattr(pair.time, 'monotonic', lambda: next(ticks))
    try:
        with pytest.raises(TimeoutError, match='unresolved'):
            lease.check()
    finally:
        lease.close()


@pytest.mark.parametrize('fail_step', [False, True])
def test_actual_trainer_wrapper_preserves_calls_and_releases_on_failure(tmp_path, monkeypatch, fail_step):
    c = config()
    (tmp_path/'config.json').write_text(json.dumps(c))
    (tmp_path/'save.lock').touch()
    (tmp_path/'ready1.json').write_text(json.dumps(dict(actor=1,request_sha256='b'*64)))
    meminfo = tmp_path/'meminfo'
    meminfo.write_text('MemAvailable: 50000000 kB\n')
    real_path = Path
    def paths(value):
        value = str(value)
        if value == '/run/host-meminfo':
            return meminfo
        if value == '/bc-pair' or value.startswith('/bc-pair/'):
            return tmp_path / value.removeprefix('/bc-pair').lstrip('/')
        return real_path(value)
    monkeypatch.setattr(pair, 'Path', paths)
    monkeypatch.setenv('ROBORENDER_PAIR_ACTOR','0')
    monkeypatch.setenv('ROBORENDER_PAIR_CONFIG_SHA256',hashlib.sha256((tmp_path/'config.json').read_bytes()).hexdigest())
    monkeypatch.setenv('ROBORENDER_BC_COLLECTION_REQUEST_SHA256','a'*64)
    calls = []
    def check_lock(operation):
        with (tmp_path/'save.lock').open('a') as fd:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, operation | fcntl.LOCK_NB)
    def step(action):
        check_lock(fcntl.LOCK_EX)
        calls.append(('step',action))
        if fail_step:
            raise RuntimeError('native failure')
        return action
    class Base:
        def train(self):
            for window in range(4):
                for index in range(81):
                    action = self.policy_step(window*81+index)
                    assert self.env.step(action) == action
                self._save_collected_chunks()
        def policy_step(self,index):
            check_lock(fcntl.LOCK_EX)
            self._check_host_memory()
            calls.append(('policy',index))
            return index
        def _save_collected_chunks(self):
            check_lock(fcntl.LOCK_SH)
            self._check_host_memory(saving=True,tail_growth_bytes=0)
            calls.append(('save',None))
    bounded = ModuleType('bounded_bc_guard')
    bounded.BoundedContinuousBCTrainer = Base
    bounded.memory_gate = memory_gate
    native = ModuleType('isaac6_bc_production')
    native.load_request = lambda: dict(motions=[{}]*32,max_windows=4,process_seed=0)
    session = ModuleType('isaac6_bc_session')
    session.memory_snapshot = sample
    for module in (bounded,native,session):
        monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.delattr(pair,'PairedBCTrainer',raising=False)
    cls = pair.__getattr__('PairedBCTrainer')
    owner = cls()
    owner.env = SimpleNamespace(step=step)
    if fail_step:
        with pytest.raises(RuntimeError,match='native failure'):
            owner.train()
        assert calls == [('policy',0),('step',0)]
    else:
        owner.train()
        assert [v for k,v in calls if k=='step'] == list(range(324))
        assert len([k for k,v in calls if k=='save']) == 4
    assert owner.env.step is step
    with (tmp_path/'save.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.delattr(pair,'PairedBCTrainer',raising=False)
