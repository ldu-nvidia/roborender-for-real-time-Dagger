"""Persistent orchestration is not permission for arbitrary GPU jobs/release."""
import json
import pytest
from pipeline.deployment.retained_service_process import read_spec


def spec(tmp_path):
    root=tmp_path/'runtime'/'production';root.mkdir(parents=True)
    settings=root/'settings.json';settings.write_text(json.dumps({'run_id':'10-17-prod'}))
    execution=root/'execution.json';execution.write_text(json.dumps({'compute_release_authorized':False}))
    value=dict(schema='roborender.retained-service-process.v1',role='mode-controller',
        owner='dagger-leo-10-17-prod',release_compute=False,cwd=str(tmp_path),env={},
        argv=['/usr/bin/python3','-m','pipeline.deployment.run_mode','--mode','production',
              '--settings',str(settings),'--output',str(root/'run'),'--execution-settings',str(execution),'--execute'])
    return root,value


def test_only_exact_retained_production_controller_is_persistent(tmp_path):
    root,value=spec(tmp_path);path=root/'spec.json';path.write_text(json.dumps(value))
    assert read_spec(path)==value
    assert 'ttl' not in read_spec(path)


@pytest.mark.parametrize('fault',['module','smoke','extra','owner','release','path','env','execution'])
def test_controller_boundary_cannot_be_used_as_arbitrary_job_runner(tmp_path,fault):
    root,value=spec(tmp_path)
    if fault=='module':value['argv'][2]='arbitrary.optimizer'
    if fault=='smoke':value['argv'][4]='smoke'
    if fault=='extra':value['argv'].append('--unreviewed')
    if fault=='owner':value['owner']='dagger-leo-other'
    if fault=='release':value['release_compute']=True
    if fault=='path':value['argv'][8]='relative/output'
    if fault=='env':value['env']['WANDB_API_KEY']='forbidden-inline-secret'
    if fault=='execution':(root/'execution.json').write_text(json.dumps({'compute_release_authorized':True}))
    path=root/'spec.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):read_spec(path)


@pytest.mark.parametrize('fault',[None,'module','owner','release','missing_admission','changed_admission','extra'])
def test_reconciled_campaign_controller_is_pinned_and_retained(tmp_path,fault):
    import hashlib
    root=tmp_path/'runtime'/'campaign';root.mkdir(parents=True)
    admission=root/'admission.json';admission.write_text('{}')
    config=root/'config.json';config.write_text(json.dumps({
        'schema':'roborender.cross_gpu_dagger_mse_full295.leo.v2','run_id':'dagger-leo-prod'}))
    operations=root/'operations.json';operations.write_text(json.dumps({'compute_release_permitted':False}))
    state={'run_id':'dagger-leo-prod','control_only_release':{'path':str(admission),
        'sha256':hashlib.sha256(admission.read_bytes()).hexdigest()}}
    value=dict(schema='roborender.retained-service-process.v1',role='campaign-controller',
        owner='dagger-leo-prod',release_compute=False,cwd=str(tmp_path),env={},
        argv=['/usr/bin/python3','-m','pipeline.dagger.mse_campaign','--first-round-config',str(config),
              '--mse-plan',str(root/'plan.json'),'--root',str(root),'--operations',str(operations),'--execute'])
    if fault=='module':value['argv'][2]='arbitrary.optimizer'
    if fault=='owner':value['owner']='dagger-leo-wrong'
    if fault=='release':operations.write_text(json.dumps({'compute_release_permitted':True}))
    if fault=='missing_admission':state.pop('control_only_release')
    if fault=='changed_admission':admission.write_text('changed')
    if fault=='extra':value['argv'].append('--unreviewed')
    (root/'state.json').write_text(json.dumps(state))
    path=root/'spec.json';path.write_text(json.dumps(value))
    if fault is None:assert read_spec(path)==value
    else:
        with pytest.raises(ValueError):read_spec(path)
