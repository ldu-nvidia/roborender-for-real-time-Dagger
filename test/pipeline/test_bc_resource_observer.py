"""Observer fault injection: no simulator/GPU work and no capacity claim."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.deployment.rtx_actor.bc_resource_observer import (
    ObservationUnavailable, OwnerChanged, PinnedCgroupObserver, recover_observation, attest_scope_limits,
)
from test.pipeline.run_paired_bc_smoke import actor_gate

G = 2**30


@pytest.mark.parametrize('swap', ['0', 'infinity', '1'])
def test_scope_policy_must_persist_zero_not_only_kernel_file(swap):
    group = '/system.slice/docker-' + 'a' * 64 + '.scope'
    def run(command, **kwargs):
        assert command[:3] == ['systemctl', 'show', Path(group).name]
        return SimpleNamespace(returncode=0,
            stdout=f'ControlGroup={group}\nMemoryMax={43*G}\nMemorySwapMax={swap}\n')
    if swap == '0':
        assert attest_scope_limits(group, 43*G, run=run)['MemorySwapMax'] == '0'
    else:
        with pytest.raises(OwnerChanged, match='persistent'):
            attest_scope_limits(group, 43*G, run=run)


@pytest.fixture
def bound(tmp_path):
    proc, cg = tmp_path/'proc', tmp_path/'cgroup'
    (proc/'sys/kernel/random').mkdir(parents=True)
    (proc/'sys/kernel/random/boot_id').write_text('boot-id')
    (proc/'123').mkdir()
    # comm may contain spaces and closing parentheses; starttime is field22.
    (proc/'123/stat').write_text('123 (test ) process) S ' + '0 '*18 + '777 0')
    (proc/'123/cgroup').write_text('0::/docker/abc\n')
    group = cg/'docker/abc'
    group.mkdir(parents=True)
    values = {'current':20*G,'peak':22*G,'max':43*G,'swap.current':0,'swap.max':0}
    for key, value in values.items():
        (group/('memory.'+key)).write_text(str(value))
    (group/'memory.events').write_text('max 0\noom 0\noom_kill 0\n')
    (group/'memory.stat').write_text('anon 100\nfile 200\nshmem 0\nkernel 20\n')
    cid = tmp_path/'container.cid'
    cid.write_text('a'*64)
    info = dict(Id='a'*64,Name='/owner',Image='sha256:image',
        Config={'Labels':{'org.roborender.qualification':'owner'}},
        HostConfig={'Memory':43*G,'MemorySwap':43*G},State={'Running':True,'Pid':123})
    calls = []
    def inspect(value):
        calls.append(value)
        return info
    def bind():
        return PinnedCgroupObserver(cid,'owner','sha256:image',43*G,
            proc_root=proc,cgroup_root=cg,inspect=inspect,
            scope_check=lambda group, cap: dict(ControlGroup=group, MemoryMax=str(cap), MemorySwapMax='0'))
    return bind, proc, group, calls, info, cid


def test_pins_once_reads_cgroup_not_docker(bound):
    bind, _, _, calls, _, _ = bound
    observer = bind()
    for _ in range(3):
        sample = observer.sample()
        actor_gate(sample)
        assert sample['process_start_ticks'] == 777
        assert sample['stat']['file'] == 200
    assert calls == ['a'*64]


@pytest.mark.parametrize('part', ['boot','pid','cgroup','inode'])
def test_owner_change_never_rebinds(bound, part):
    bind, proc, group, calls, _, _ = bound
    observer = bind()
    if part == 'boot':
        (proc/'sys/kernel/random/boot_id').write_text('reboot')
    elif part == 'pid':
        p = proc/'123/stat'
        p.write_text(p.read_text().replace('777','888'))
    elif part == 'cgroup':
        (proc/'123/cgroup').write_text('0::/different\n')
    else:
        group.rename(group.with_name('old'))
        group.mkdir()
    with pytest.raises(OwnerChanged):
        observer.sample()
    assert len(calls) == 1


@pytest.mark.parametrize('key,value', [('max',40*G),('swap.max',G)])
def test_actual_kernel_limits_not_only_docker_flags(bound, key, value):
    bind, _, group, _, _, _ = bound
    observer = bind()
    (group/('memory.'+key)).write_text(str(value))
    with pytest.raises(OwnerChanged, match='limits changed') as error:
        observer.sample()
    assert 'memory_'+key in str(error.value)


@pytest.mark.parametrize('key', ['max', 'swap.max'])
def test_unlimited_limit_is_durable_owner_drift_not_transient_unknown(bound, key):
    bind, _, group, _, _, _ = bound
    observer = bind()
    (group / ('memory.' + key)).write_text('max\n')
    with pytest.raises(OwnerChanged, match='unlimited') as error:
        observer.sample()
    assert '"memory_' + key + '": "max"' in str(error.value)


@pytest.mark.parametrize('failure', ['missing','malformed'])
def test_counter_read_failure_is_unknown_not_oom(bound, failure):
    bind, _, group, _, _, _ = bound
    observer = bind()
    p = group/'memory.current'
    if failure == 'missing':
        p.unlink()
    else:
        p.write_text('not a number')
    with pytest.raises(ObservationUnavailable):
        observer.sample()


def test_binding_requires_exact_image_and_cid(bound):
    bind, _, _, _, info, cid = bound
    info['Image'] = 'wrong-image'
    with pytest.raises(OwnerChanged):
        bind()
    cid.write_text('partial-id')
    with pytest.raises(OwnerChanged):
        bind()


def test_empty_startup_cidfile_waits_for_acceptance(bound):
    bind, _, _, calls, _, cid = bound
    cid.write_text('')
    with pytest.raises(ObservationUnavailable, match='not written yet'):
        bind()
    assert calls == []
    cid.write_text('a'*64)
    observer = bind()
    actor_gate(observer.sample())
    assert calls == ['a'*64]


def drive(read, validate=lambda _:None, **kwargs):
    events, actions = [], []
    def run():
        return recover_observation(read,validate,journal=events.append,
            hold=lambda:actions.append('hold'),release=lambda:actions.append('release'),
            host_check=lambda:actions.append('host_check'),delay=lambda _:None,**kwargs)
    return run, events, actions


def test_transient_read_holds_then_recovers_without_resubmission():
    results = iter([ObservationUnavailable('injected read error'), {'fresh':True}])
    def read():
        v = next(results)
        if isinstance(v, Exception):
            raise v
        return v
    run, events, actions = drive(read)
    assert run() == {'fresh':True}
    assert [e['status'] for e in events] == ['UNKNOWN','OBSERVED','RECOVERED']
    assert actions.count('hold') == actions.count('release') == 1


def test_persistent_unknown_never_uses_stale_sample_or_releases_hold():
    def read():
        raise ObservationUnavailable('missing cgroup')
    run, events, actions = drive(read)
    with pytest.raises(ObservationUnavailable, match='3 failed'):
        run()
    assert len(events) == 3
    assert actions.count('hold') == 1 and 'release' not in actions


@pytest.mark.parametrize('field', ['swap','oom','max','headroom'])
def test_unsafe_sample_is_durable_and_never_retried(bound, field):
    bind, _, group, _, _, _ = bound
    observer = bind()
    if field == 'swap':
        (group/'memory.swap.current').write_text('4096')
    elif field == 'headroom':
        (group/'memory.current').write_text(str(42*G))
    else:
        p = group/'memory.events'
        p.write_text(p.read_text().replace(field+' 0',field+' 1'))
    run, events, actions = drive(observer.sample,actor_gate)
    with pytest.raises(RuntimeError):
        run()
    assert [e['status'] for e in events] == ['OBSERVED','REJECTED']
    assert events[0]['sample']['pid'] == 123
    assert 'hold' not in actions
    assert json.loads(json.dumps(events)) == events


def test_changed_owner_is_not_transient():
    def read():
        raise OwnerChanged('PID recycled')
    run, events, actions = drive(read)
    with pytest.raises(OwnerChanged):
        run()
    assert len(events) == 1 and 'hold' not in actions


def test_startup_grace_still_requires_fresh_observation():
    attempts = []
    def read():
        attempts.append(1)
        if len(attempts) < 6:
            raise ObservationUnavailable('Docker not accepted yet')
        return {'accepted':True}
    run, events, actions = drive(read,attempts=20)
    assert run() == {'accepted':True}
    assert len(attempts) == 6
    assert actions.count('hold') == actions.count('release') == 1


def test_host_pressure_still_stops_retry():
    def fail():
        raise RuntimeError('host reserve breached')
    with pytest.raises(RuntimeError, match='host reserve'):
        recover_observation(lambda:pytest.fail('must not read'),lambda _:None,
            journal=lambda _:None,hold=lambda:None,release=lambda:None,host_check=fail)


def test_native_observer_hold_waits_without_work_and_resumes_when_removed(tmp_path):
    from pipeline.deployment.rtx_actor.bc_resource_observer import wait_for_observer
    marker = tmp_path / 'observer-hold.json'
    marker.write_text('{}')
    waits = []
    def delay(seconds):
        waits.append(seconds)
        marker.unlink()
    wait_for_observer(tmp_path, delay=delay, clock=lambda: 0)
    assert waits == [.1]
    wait_for_observer(tmp_path, delay=lambda _: pytest.fail('no hold, no wait'))


def test_native_observer_hold_timeout_does_not_clear_marker(tmp_path):
    from pipeline.deployment.rtx_actor.bc_resource_observer import wait_for_observer
    marker = tmp_path / 'observer-hold.json'
    marker.write_text('{}')
    clock = iter([0, 30])
    with pytest.raises(ObservationUnavailable, match='30seconds'):
        wait_for_observer(tmp_path, clock=lambda: next(clock), delay=lambda _:None)
    assert marker.exists()


def test_native_exit_before_docker_client_is_reconciled_only_with_receipt_and_zero_exit():
    from pipeline.deployment.rtx_actor.bc_resource_observer import observe_command
    calls = []
    def wait(*, timeout):
        calls.append(timeout)
        return 0
    process = SimpleNamespace(poll=lambda: None, wait=wait)
    def dead(): raise ObservationUnavailable('native PID is zombie')
    terminal = dict(completion_receipt_sha256='a'*64, native_pass_receipt=True)
    result = observe_command(process, dead, lambda: dict(terminal), lambda: True)
    assert calls == [2] and result == dict(lifecycle='completed_attested', exit_code=0, **terminal)


def test_no_terminal_evidence_means_no_wait_and_no_invented_completion():
    from pipeline.deployment.rtx_actor.bc_resource_observer import observe_command
    process = SimpleNamespace(poll=lambda: None, wait=lambda **_: pytest.fail('no wait without receipt'))
    def unavailable(): raise ObservationUnavailable('unknown cgroup')
    with pytest.raises(ObservationUnavailable):
        observe_command(process, unavailable, lambda: pytest.fail('no terminal read'), lambda: False)


def test_invalid_receipt_or_nonzero_exit_cannot_complete():
    from pipeline.deployment.rtx_actor.bc_resource_observer import observe_command
    def unavailable(): raise ObservationUnavailable('gone')
    def invalid(): raise ValueError('request/source mismatch')
    process = SimpleNamespace(poll=lambda: None, wait=lambda **_: pytest.fail('invalid receipt cannot wait'))
    with pytest.raises(ValueError):
        observe_command(process, unavailable, invalid, lambda: True)
    process.wait = lambda **_: 137
    with pytest.raises(RuntimeError, match='nonzero'):
        observe_command(process, unavailable, lambda: {}, lambda: True)


def test_pending_exit_stays_unknown_and_changed_owner_never_reconciles():
    import subprocess
    from pipeline.deployment.rtx_actor.bc_resource_observer import observe_command
    def pending(**_): raise subprocess.TimeoutExpired('exact', 2)
    def unavailable(): raise ObservationUnavailable('gone')
    process = SimpleNamespace(poll=lambda: None, wait=pending)
    with pytest.raises(ObservationUnavailable, match='exit still pending'):
        observe_command(process, unavailable, lambda: {}, lambda: True)
    def changed(): raise OwnerChanged('scope limit changed')
    with pytest.raises(OwnerChanged):
        observe_command(process, changed, lambda: pytest.fail('no terminal override'), lambda: True)


def test_terminal_evidence_cannot_change_during_exit_wait():
    from pipeline.deployment.rtx_actor.bc_resource_observer import observe_command
    values = iter([dict(sha='a'), dict(sha='b')])
    process = SimpleNamespace(poll=lambda: None, wait=lambda **_: 0)
    def unavailable(): raise ObservationUnavailable('gone')
    with pytest.raises(OwnerChanged, match='terminal evidence changed'):
        observe_command(process, unavailable, lambda: next(values), lambda: True)
