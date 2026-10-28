"""Versioned bounded PSI stalls; hard memory limits never get a grace period."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(PIPELINE_ROOT/'dagger/rtx/code/src'))
from dagger_rtx.phase_resource_guard import (
    PhaseResourceGuard, PressureWindow, SUSTAINED_VERSION, SUSTAINED_SCHEMA,
    POLICY_VERSION, SCHEMA, classify_sample, validate_summary, GIB,
)
from test.pipeline.test_dagger_phase_resource_guard import healthy, cgroup_fixture
from dagger_rtx.phase_resource_guard import sample_resources


def test_transient_pressure_recovers_without_erasing_peak(tmp_path):
    rows=iter([healthy(full_avg10=5.07),healthy()])
    times=iter([100.,101.])
    with PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:next(rows),
            clock=lambda:next(times),policy_version=SUSTAINED_VERSION) as guard:
        guard.check()
    assert guard.summary['schema']==SUSTAINED_SCHEMA
    assert guard.summary['maximum_full_avg10']==5.07
    assert guard.summary['pressure_window']==dict(warning_samples=1,
        maximum_warning_seconds=1.,failed=False)
    validate_summary(guard.summary)
    value=json.loads((tmp_path/'guard.json').read_text())
    assert value['pressure_window']==guard.summary['pressure_window']


@pytest.mark.parametrize('recovered',[False,True])
def test_warning_deadline_latches_even_on_recovery(recovered):
    monitor=PressureWindow()
    monitor.observe(healthy(full_avg10=5.07),100.)
    monitor.observe(healthy(full_avg10=5.07),129.9)
    with pytest.raises(RuntimeError,match='sustained'):
        monitor.observe(healthy(full_avg10=0 if recovered else 5.07),130.)
    with pytest.raises(RuntimeError): monitor.observe(healthy(),131.)
    assert monitor.summary()['failed'] is True


def test_short_separated_warnings_do_not_accumulate_continuously():
    monitor=PressureWindow()
    for now in (0.,20.,40.):
        monitor.observe(healthy(full_avg10=5.07),now)
        monitor.observe(healthy(),now+10)
    assert monitor.summary()==dict(warning_samples=3,maximum_warning_seconds=10.,failed=False)


@pytest.mark.parametrize('now',[float('nan'),float('inf'),-1.])
def test_unknown_or_regressing_clock_is_rejected(now):
    monitor=PressureWindow();monitor.observe(healthy(),0.)
    with pytest.raises(RuntimeError):monitor.observe(healthy(),now)


@pytest.mark.parametrize('changes',[
    {'full_avg10':20.01},{'some_avg10':40.01},{'swap_bytes':1},
    {'events':{'max':0,'oom':1,'oom_kill':0}},
    {'events':{'max':0,'oom':0,'oom_kill':1}},
    {'host_available_bytes':8*GIB-1},
    {'memory_current_bytes':21*GIB+1},
    {'memory_current_bytes':21*GIB,'inactive_file_bytes':0},
    {'memory_limit_bytes':22*GIB},{'swap_limit_bytes':1},
])
def test_hard_failures_remain_immediate(changes):
    with pytest.raises(RuntimeError):
        classify_sample(healthy(**changes),policy_version=SUSTAINED_VERSION)


def test_v2_is_not_relaxed_and_unknown_versions_rejected():
    with pytest.raises(RuntimeError):classify_sample(healthy(full_avg10=5.07))
    with pytest.raises(RuntimeError):classify_sample(healthy(),policy_version='unknown')
    classify_sample(healthy(full_avg10=5.07),policy_version=SUSTAINED_VERSION)


def test_real_counter_reader_uses_explicit_policy_and_never_masks_oom(tmp_path):
    _,group,_,meminfo=cgroup_fixture(tmp_path)
    (group/'memory.pressure').write_text('some avg10=6.00 avg60=0 avg300=0 total=1\nfull avg10=5.07 avg60=0 avg300=0 total=1\n')
    with pytest.raises(RuntimeError):sample_resources(cgroup_root=group,meminfo=meminfo)
    value=sample_resources(cgroup_root=group,meminfo=meminfo,policy_version=SUSTAINED_VERSION)
    assert value['full_avg10']==5.07
    (group/'memory.events').write_text('max 0\noom 1\noom_kill 0\n')
    with pytest.raises(RuntimeError):
        sample_resources(cgroup_root=group,meminfo=meminfo,policy_version=SUSTAINED_VERSION)


@pytest.mark.parametrize('fault',['missing','duration','failed','samples','relabel'])
def test_summary_requires_bounded_pressure_history(tmp_path,fault):
    with PhaseResourceGuard(tmp_path/'guard.json',sample=healthy,
            policy_version=SUSTAINED_VERSION) as guard:pass
    row=deepcopy(guard.summary)
    if fault=='missing':del row['pressure_window']
    elif fault=='duration':row['pressure_window']['maximum_warning_seconds']=30.
    elif fault=='failed':row['pressure_window']['failed']=True
    elif fault=='samples':row['pressure_window']['warning_samples']=row['samples']+1
    else:row['schema']=SCHEMA
    with pytest.raises(RuntimeError):validate_summary(row)


def test_failed_v2_receipt_stays_failed(tmp_path):
    with pytest.raises(RuntimeError):
        with PhaseResourceGuard(tmp_path/'old.json',sample=lambda:healthy(full_avg10=5.07)):pass
    old=json.loads((tmp_path/'old.json').read_text())
    assert old['policy_version']==POLICY_VERSION and old['passed'] is False
    with pytest.raises(RuntimeError):validate_summary(old)


def test_host_reader_binds_v3_history_to_actual_receipt(tmp_path):
    from test.pipeline.test_native_resource_guard_v2 import republish
    from production_resident_pool import validate_guard
    with PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:healthy(full_avg10=5.07),
            policy_version=SUSTAINED_VERSION) as guard:pass
    receipt=validate_guard(guard.summary,dict(runtime_root=str(tmp_path)))
    assert receipt['schema']==SUSTAINED_SCHEMA
    receipt['pressure_window']['maximum_warning_seconds']+=1
    with pytest.raises(RuntimeError,match='pressure history'):
        validate_guard(republish(guard.summary,receipt),dict(runtime_root=str(tmp_path)))


def test_outer_launcher_requires_lifetime_pressure_window():
    from test.pipeline.test_native_resource_guard_v2 import actual
    from run_production_resident_isaac import resource_gate
    row=dict(pid=12,oom_killed=False,**{'memory_swap.current':0},
        resource_sample_v2=actual(full_avg10=5.07))
    host={'MemAvailable':12*GIB//1024}
    with pytest.raises(RuntimeError,match='stateful'):
        resource_gate(host,row,policy_version=SUSTAINED_VERSION)
    tracker=PressureWindow()
    resource_gate(host,row,policy_version=SUSTAINED_VERSION,pressure=tracker,now=0.)
    with pytest.raises(RuntimeError,match='sustained'):
        resource_gate(host,row,policy_version=SUSTAINED_VERSION,pressure=tracker,now=30.)


def test_peak_without_warning_evidence_is_rejected(tmp_path):
    with PhaseResourceGuard(tmp_path/'guard.json',sample=healthy,
            policy_version=SUSTAINED_VERSION) as guard:pass
    row=deepcopy(guard.summary);row['maximum_full_avg10']=5.07
    with pytest.raises(RuntimeError,match='disagree'):validate_summary(row)
