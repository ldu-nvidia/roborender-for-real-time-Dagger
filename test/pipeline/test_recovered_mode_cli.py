import json
import sys
import pytest
from pipeline.bc_training.handoff_common import binding, canonical
from pipeline.deployment import run_mode, mode_execution


@pytest.mark.parametrize('changed_settings',[False,True])
def test_completed_bc_is_not_regenerated(tmp_path, monkeypatch, changed_settings):
    settings=tmp_path/'settings.json'; settings.write_bytes(b'{}')
    execution=tmp_path/'execution.json'; execution.write_bytes(b'{}')
    plan=tmp_path/'pipeline-mode.json'; plan.write_bytes(b'{}')
    receipt=tmp_path/'bc-receipt.json'; receipt.write_bytes(b'{}')
    root=tmp_path/'execution'; root.mkdir()
    state=dict(plan=binding(plan),settings=binding(execution),status='READY',completed=[
        dict(phase='prepare_bc',receipt=binding(receipt)),
        dict(phase='train_bc',receipt=binding(receipt)),
        dict(phase='test_bc',receipt=binding(receipt))])
    (root/'state.json').write_bytes(canonical(state))
    if changed_settings:execution.write_bytes(b'{"changed":true}')
    monkeypatch.setattr(run_mode,'stage',lambda *args:{})
    def forbidden(*args):raise AssertionError('must not generate or repeat BC')
    monkeypatch.setattr(run_mode,'bc_job',forbidden)
    calls=[]
    monkeypatch.setattr(mode_execution,'execute',lambda *args,**kwargs: calls.append((args,kwargs)) or {'status':'COMPLETE'})
    monkeypatch.setattr(sys,'argv',['run_mode','--mode','production','--settings',str(settings),
        '--output',str(tmp_path),'--execution-settings',str(execution),'--execute'])
    before=(root/'state.json').read_bytes()
    with pytest.raises(ValueError,match='MSE validation is disabled'):run_mode.main()
    assert not calls and (root/'state.json').read_bytes()==before
    assert not (tmp_path/'bc-job.json').exists()
    monkeypatch.setattr(sys,'argv',sys.argv+['--allow-legacy-validation'])
    if changed_settings:
        with pytest.raises(ValueError,match='bindings changed'):run_mode.main()
        assert not calls
    else:
        run_mode.main()
        assert calls==[((plan,execution),{'allow_legacy_validation':True})]
    assert not (tmp_path/'bc-job.json').exists()
