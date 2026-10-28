"""CPU-only ownership/resource guard checks; no CUDA or compute control."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0,str(PIPELINE_ROOT/'dagger/rtx/code/src'))
from dagger_rtx.phase_resource_guard import (PhaseResourceGuard, own_cgroup_root,
    sample_resources, classify_sample, validate_summary, validate_transition, GIB, SCHEMA, POLICY_VERSION)


def healthy(**changes):
    row=dict(host_available_bytes=12*GIB,memory_current_bytes=3*GIB,swap_bytes=0,
        memory_limit_bytes=21*GIB,swap_limit_bytes=0,inactive_file_bytes=GIB,
        some_avg10=0.0,full_avg10=0.0,process_id=1,cgroup_path='/owned',
        cgroup_device=1,cgroup_inode=2,events={'max':0,'oom':0,'oom_kill':0})
    row.update(changes)
    row.setdefault('working_set_bytes',row['memory_current_bytes']-row['inactive_file_bytes'])
    return row


def cgroup_fixture(tmp_path,*,membership='0::/owned\n'):
    mount=tmp_path/'cgroup'; group=mount/'owned'; group.mkdir(parents=True)
    for folder in (mount,group):
        (folder/'memory.events').write_text('low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n')
        (folder/'memory.current').write_text(str(3*GIB))
        (folder/'memory.swap.current').write_text('0')
        (folder/'memory.max').write_text(str(21*GIB))
        (folder/'memory.swap.max').write_text('0')
        (folder/'memory.stat').write_text('inactive_file '+str(GIB)+'\n')
        (folder/'memory.pressure').write_text('some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n')
    member=tmp_path/'self-cgroup'; member.write_text(membership)
    meminfo=tmp_path/'meminfo'; meminfo.write_text('MemTotal: 64000000 kB\nMemAvailable: 12582912 kB\n')
    return mount,group,member,meminfo


def test_owned_unified_cgroup_root_and_nested_mapping(tmp_path):
    mount,group,member,meminfo=cgroup_fixture(tmp_path)
    assert own_cgroup_root(mount=mount,membership=member)==group
    member.write_text('0::/\n')
    assert own_cgroup_root(mount=mount,membership=member)==mount


@pytest.mark.parametrize('membership',['0::/missing\n','0::/../elsewhere\n',
    '2:memory:/owned\n','0::/owned\n0::/\n','0::relative\n'])
def test_missing_or_ambiguous_owned_cgroup_never_falls_back_to_root(tmp_path,membership):
    mount,group,member,meminfo=cgroup_fixture(tmp_path,membership=membership)
    with pytest.raises((RuntimeError,FileNotFoundError)):
        own_cgroup_root(mount=mount,membership=member)


def test_cgroup_alias_cannot_escape_owned_mount(tmp_path):
    mount,group,member,meminfo=cgroup_fixture(tmp_path)
    elsewhere=tmp_path/'elsewhere'; elsewhere.mkdir(); (elsewhere/'memory.events').write_text('max 0\n')
    (mount/'alias').symlink_to(elsewhere,target_is_directory=True); member.write_text('0::/alias\n')
    with pytest.raises(RuntimeError): own_cgroup_root(mount=mount,membership=member)


def test_actual_file_counters_include_process_and_cgroup_identity(tmp_path):
    mount,group,member,meminfo=cgroup_fixture(tmp_path)
    value=sample_resources(cgroup_root=own_cgroup_root(mount,membership=member),meminfo=meminfo)
    assert value['host_available_bytes']==12*GIB
    assert value['memory_current_bytes']==3*GIB and value['swap_bytes']==0
    assert value['events']=={'max':0,'oom':0,'oom_kill':0}
    assert value['inactive_file_bytes']==GIB and value['working_set_bytes']==2*GIB
    assert value['some_avg10']==value['full_avg10']==0
    assert value['memory_limit_bytes']==21*GIB and value['swap_limit_bytes']==0
    assert value['process_id']==os.getpid() and value['cgroup_path']==str(group)
    assert value['cgroup_device']==group.stat().st_dev and value['cgroup_inode']==group.stat().st_ino


@pytest.mark.parametrize('fault',['missing_swap','missing_available','missing_oom','negative_current'])
def test_missing_or_invalid_actual_counters_are_errors(tmp_path,fault):
    mount,group,member,meminfo=cgroup_fixture(tmp_path)
    if fault=='missing_swap': (group/'memory.swap.current').unlink()
    elif fault=='missing_available': meminfo.write_text('MemTotal: 64000000 kB\n')
    elif fault=='missing_oom': (group/'memory.events').write_text('max 0\noom_kill 0\n')
    else: (group/'memory.current').write_text('-1')
    with pytest.raises((RuntimeError,FileNotFoundError,KeyError)):
        sample_resources(cgroup_root=group,meminfo=meminfo)


@pytest.mark.parametrize('interval',[0,-.1,.50001,1,float('nan'),float('inf')])
def test_monitor_interval_is_at_most_half_a_second(tmp_path,interval):
    with pytest.raises(ValueError): PhaseResourceGuard(tmp_path/'resources.json',interval_s=interval)


def test_normal_lifecycle_receipt_is_bounded_actual_and_hash_bound(tmp_path):
    path=tmp_path/'resources.json'; calls=[]
    def sample():
        row=healthy(memory_current_bytes=(3+len(calls))*GIB); calls.append(deepcopy(row)); return row
    with PhaseResourceGuard(path,sample=sample,interval_s=.5) as guard:
        guard.check()
    raw=path.read_bytes(); value=json.loads(raw)
    assert path.stat().st_mode & 0o777 == 0o644
    assert len(raw)<16384 and value['passed'] is True and value['samples']==2
    assert value['first_sample']==calls[0] and value['last_sample']==calls[-1]
    assert value['memory_current_peak_bytes']==4*GIB and value['minimum_host_available_bytes']==12*GIB
    assert value['swap_peak_bytes']==value['oom_events']==value['limit_pressure_events']==0
    assert guard.summary['path']==str(path) and guard.summary['raw_sha256']==hashlib.sha256(raw).hexdigest()
    assert guard.summary['passed'] is True
    with pytest.raises(RuntimeError,match='replayed'): PhaseResourceGuard(path,sample=sample)


@pytest.mark.parametrize('unsafe',[healthy(host_available_bytes=8*GIB-1),healthy(swap_bytes=1),
    healthy(some_avg10=10.01),healthy(events={'max':0,'oom':1,'oom_kill':0}),
    healthy(events={'max':0,'oom':0,'oom_kill':1})])
def test_unsafe_initial_sample_rejects_entry_and_persists_failed_receipt(tmp_path,unsafe):
    path=tmp_path/'resources.json'; entered=False
    with pytest.raises(RuntimeError):
        with PhaseResourceGuard(path,sample=lambda:deepcopy(unsafe)):
            entered=True
    assert not entered
    value=json.loads(path.read_bytes())
    assert value['passed'] is False and value['samples']==1
    assert value['first_sample']==value['last_sample']==unsafe
    assert value['sampling_error'] and value['memory_current_peak_bytes']==unsafe['memory_current_bytes']


def test_initial_unknown_sample_never_becomes_a_zero_resource_claim(tmp_path):
    def unavailable(): raise PermissionError('synthetic procfs denial')
    path=tmp_path/'resources.json'
    with pytest.raises(RuntimeError):
        with PhaseResourceGuard(path,sample=unavailable): pass
    row=json.loads(path.read_bytes())
    assert row['passed'] is False and row['samples']==0
    assert row['first_sample'] is None and row['last_sample'] is None
    for key in ('minimum_host_available_bytes','memory_current_peak_bytes','swap_peak_bytes','oom_events','limit_pressure_events'):
        assert row[key] is None


@pytest.mark.parametrize('failure',['unsafe','error'])
def test_failure_stays_latched_after_later_good_sample(tmp_path,failure):
    calls=0
    def sample():
        nonlocal calls
        calls+=1
        if calls==2:
            if failure=='error': raise OSError('synthetic counter failure')
            return healthy(full_avg10=5.01)
        return healthy()
    path=tmp_path/'resources.json'; guard=PhaseResourceGuard(path,sample=sample,interval_s=.5)
    guard.__enter__(); guard._sample(); guard._sample()
    with pytest.raises(RuntimeError): guard.check()
    with pytest.raises(RuntimeError): guard.__exit__(None,None,None)
    row=json.loads(path.read_bytes()); assert row['passed'] is False and row['sampling_error']


def test_application_error_is_preserved_and_receipt_fails(tmp_path):
    path=tmp_path/'resources.json'
    with pytest.raises(ValueError,match='native phase failed'):
        with PhaseResourceGuard(path,sample=healthy): raise ValueError('native phase failed')
    assert json.loads(path.read_bytes())['passed'] is False


def test_stuck_background_sampler_cannot_hang_finalization_or_late_update_receipt(tmp_path):
    started=threading.Event(); release=threading.Event(); calls=0
    def sample():
        nonlocal calls
        calls+=1
        if calls==1: return healthy()
        started.set(); release.wait(10)
        return healthy(swap_bytes=GIB)
    path=tmp_path/'resources.json'; guard=PhaseResourceGuard(path,sample=sample,interval_s=.01)
    try:
        guard.__enter__(); assert started.wait(1)
        begin=time.monotonic()
        with pytest.raises(RuntimeError): guard.__exit__(None,None,None)
        assert time.monotonic()-begin<2.8
        raw=path.read_bytes(); row=json.loads(raw)
        assert row['passed'] is False and 'failed to stop' in row['sampling_error']
        release.set(); guard.thread.join(1)
        assert path.read_bytes()==raw and guard.samples==1
    finally: release.set()


def test_v2_reclaim_retains_absolute_counter_and_separate_phase_delta(tmp_path):
    calls=iter([healthy(memory_current_bytes=21*GIB,inactive_file_bytes=8*GIB,
                        events={'max':7,'oom':0,'oom_kill':0}),
                healthy(memory_current_bytes=21*GIB,inactive_file_bytes=8*GIB,
                        events={'max':31,'oom':0,'oom_kill':0})])
    with PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:next(calls)) as guard: pass
    value=json.loads(guard.path.read_bytes())
    assert value['schema']==SCHEMA and value['policy_version']==POLICY_VERSION
    assert value['limit_pressure_events']==value['cache_reclaim_events']==31
    assert value['phase_limit_pressure_events']==value['phase_cache_reclaim_events']==24
    assert value['initial_limit_pressure_events']==7
    assert value['unsafe_pressure_events']==value['phase_unsafe_pressure_events']==0
    assert validate_summary(guard.summary) is guard.summary


@pytest.mark.parametrize('changes',[
    {'memory_limit_bytes':22*GIB},{'swap_limit_bytes':GIB},{'swap_bytes':1},
    {'memory_current_bytes':21*GIB,'inactive_file_bytes':GIB-1},
    {'memory_current_bytes':21*GIB+1},{'working_set_bytes':1},
    {'some_avg10':float('nan')},{'full_avg10':float('inf')},{'some_avg10':10.01},
    {'full_avg10':5.01},{'process_id':0},{'cgroup_path':'/owned/../elsewhere'},
    {'events':{'max':31,'oom':1,'oom_kill':0}},
    {'events':{'max':31,'oom':0,'oom_kill':1}},
])
def test_cache_reclaim_never_waives_actual_unsafe_conditions(changes):
    with pytest.raises(RuntimeError): classify_sample(healthy(**changes))


def test_counter_regression_is_against_previous_sample_not_only_first():
    first=healthy(events={'max':7,'oom':0,'oom_kill':0})
    current=healthy(events={'max':31,'oom':0,'oom_kill':0})
    validate_transition(first,current)
    with pytest.raises(RuntimeError,match='regressed'):
        validate_transition(current,healthy(events={'max':8,'oom':0,'oom_kill':0}))
    with pytest.raises(RuntimeError,match='identity'):
        validate_transition(first,healthy(cgroup_inode=3))


def test_v2_summary_rejects_relabelled_legacy_or_bad_classification(tmp_path):
    with PhaseResourceGuard(tmp_path/'guard.json',sample=healthy) as guard: pass
    for changes in ({'cache_reclaim_events':1},{'unsafe_pressure_events':1},
            {'phase_limit_pressure_events':1},{'phase_cache_reclaim_events':1},
            {'maximum_some_avg10':float('nan')},{'maximum_full_avg10':5.1},
            {'memory_limit_bytes':22*GIB},{'policy_version':'unknown'},{'samples':True}):
        with pytest.raises(RuntimeError): validate_summary(guard.summary|changes)
    legacy={k:guard.summary[k] for k in ('passed','samples','minimum_host_available_bytes',
        'swap_peak_bytes','oom_events','limit_pressure_events')}
    assert validate_summary(legacy) is legacy
    with pytest.raises(RuntimeError,match='legacy'):
        validate_summary(legacy|{'limit_pressure_events':31})


@pytest.mark.parametrize('fault',['missing_stat','missing_psi','nonfinite_psi','unbounded_cap'])
def test_v2_unknown_proof_cannot_be_inferred_safe(tmp_path,fault):
    mount,group,member,meminfo=cgroup_fixture(tmp_path)
    if fault=='missing_stat':(group/'memory.stat').write_text('anon 1\n')
    elif fault=='missing_psi':(group/'memory.pressure').write_text('some avg10=0.00\n')
    elif fault=='nonfinite_psi':(group/'memory.pressure').write_text('some avg10=nan\nfull avg10=0.00\n')
    else:(group/'memory.max').write_text('max')
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        sample_resources(cgroup_root=group,meminfo=meminfo)
