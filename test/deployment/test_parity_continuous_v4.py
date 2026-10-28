"""Runtime-only bounded diagnostic guards; no GPU or network actions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'continuous_v4.py'
if not PATH.is_file():pytest.skip('runtime diagnostic adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('continuous_v4_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_current_source_fresh_scope_and_no_release_operations():
    assert m.base.RUN==m.RUN and m.RUN.endswith('-20261026-v4')
    assert m.OUTPUT==ROOT/'continuous-v4-output'
    source=PATH.read_text()
    assert "api.collect(site,sleep=drive)" in source
    assert "training_curriculum=[bootstrap_request(),bootstrap_request()]" in source
    assert "diagnostic_learner_chain=plan['chain']['path']" in source
    assert 'shutdown(' not in source and 'allocation(' not in source

@pytest.mark.parametrize('iid,status,barriers,valid',[
    (1,'collecting',[],True),(4,'learning',[],True),(5,'collecting',[],True),
    (5,'learning',[],False),(5,'collecting',[dict(iteration_id=5)],False),
    (6,'collecting',[],False),(4,'failed',[],False),
])
def test_controller_cannot_cross_four_update_scope(iid,status,barriers,valid):
    value=dict(run=dict(current_round=1,status=status),iterations=[dict(round_id=1,iteration_id=iid)],barriers=barriers)
    controller=SimpleNamespace(coordinator=SimpleNamespace(get=lambda _:value),tick=lambda:dict(ticked=True))
    if valid:
        assert m.base.pump_controller(controller)==(dict(action='four_update_boundary_wait') if iid==5 else dict(ticked=True))
    else:
        with pytest.raises(RuntimeError):m.base.pump_controller(controller)

def test_prior_dispatch_only_observes(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'OUTPUT',tmp_path)
    m.immutable(tmp_path/'launch-intent.json',dict(plan=dict(existing=True)),new=True)
    monkeypatch.setattr(m.u,'authority',lambda **_:None)
    monkeypatch.setattr(m,'load',lambda:({},None,None,None))
    monkeypatch.setattr(m,'binding',lambda _:dict(existing=True))
    monkeypatch.setattr(m,'observe',lambda:dict(read_only=True))
    monkeypatch.setattr(m,'retained',lambda *a:pytest.fail('duplicate service start'))
    assert m.execute()==dict(read_only=True)

def test_missing_actual_scale_proof_fails_before_native_or_service_actions(tmp_path):
    context=SimpleNamespace(paired=SimpleNamespace(validate_dispatch_evidence=lambda _:pytest.fail('missing proof')))
    with pytest.raises(ValueError,match='bounded absolute'):m.qualify(context,dict(output=str(tmp_path)))

def test_plan_admits_current_source_unified_execution():
    # An actual isolated process is required: the whole-suite pytest process
    # legitimately imports the working tree. Never weaken the source jail or
    # delete its modules merely to make this test pass.
    import json
    import subprocess
    import sys
    code="""import importlib.util,json,sys
s=importlib.util.spec_from_file_location('isolated_continuous_v4',sys.argv[1])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.s.dependencies(dict(m.read(m.u.ROOT/'upgrade-plan.json')['config'],cpu_source_release=str(m.u.RELEASE)))
from pipeline.dagger.services.diagnostic_reentry import validate_execution
value=m.execution();assert validate_execution(value,m.u.SOURCE)==value
print(json.dumps(value))
"""
    done=subprocess.run([sys.executable,'-I','-c',code,str(PATH)],capture_output=True,text=True,check=True,timeout=90)
    value=json.loads(done.stdout)
    assert value['native_source_sha256']==value['learner_source_sha256']
    assert len(value['worker_sha256'])==64

def test_fault_is_not_erased_by_progress_heartbeat(monkeypatch,tmp_path):
    import json
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'OUTPUT',tmp_path/'continuous')
    m.OUTPUT.mkdir()
    (m.OUTPUT/'controller-status.json').write_text('{"phase":"FAILED"}')
    def previous(*_):
        (tmp_path/'progress.json').write_text('{"faults":[],"accepted_operations":[]}')
    monkeypatch.setattr(m.v,'heartbeat',previous)
    m.heartbeat('ATTENTION','test',fault='actual failure')
    value=json.loads((tmp_path/'progress.json').read_text())
    assert value['faults']==['actual failure']
    assert value['accepted_operations']==[dict(status_file=str(m.OUTPUT/'controller-status.json'))]
