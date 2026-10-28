"""CPU-only scope/receipt tests; synthetic evidence is not runtime certification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_training.handoff_common import binding,bound,publish
from pipeline.dagger import coverage_queue as queue
from pipeline.dagger.coverage_production import make_plan,stage,digest
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_GPU_UUIDS
from test.pipeline.test_dagger_coverage_production import production,rendering_dependencies


def queue_stub(root,**changes):
    return dict(schema=queue.SCALED_FRESH_SCHEMA,run_id='dagger-leo-reliability-fixture',
        root=str(root),source_sha256='a'*64,**changes)


def test_legacy_execution_is_unchanged_and_unscoped(tmp_path):
    value=queue_stub(tmp_path)
    value['schema']=queue.SCHEMA
    assert queue.admit_execution_scope(value,'b'*64,qualification_only=False) is None
    assert not (tmp_path/'execution-scope.json').exists()


def test_scope_is_durable_and_flag_omission_cannot_broaden_resume(tmp_path):
    value=queue_stub(tmp_path)
    scope=queue.admit_execution_scope(value,'b'*64,qualification_only=True)
    assert queue.admit_execution_scope(value,'b'*64,qualification_only=True)==scope
    assert bound(scope)['optimizer_updates']==4
    assert bound(scope)['compute_release_permitted'] is False
    with pytest.raises(ValueError,match='every execute/resume'):
        queue.admit_execution_scope(value,'b'*64,qualification_only=False)
    with pytest.raises(ValueError,match='immutable'):
        queue.admit_execution_scope(value,'c'*64,qualification_only=True)


@pytest.mark.parametrize('schema',[queue.SCHEMA,queue.FRESH_SCHEMA])
def test_new_flag_does_not_reinterpret_legacy_campaigns(tmp_path,schema):
    value=queue_stub(tmp_path);value['schema']=schema
    with pytest.raises(ValueError,match='fresh scaled'):
        queue.admit_execution_scope(value,'b'*64,qualification_only=True)


def test_cannot_retrofit_scope_on_existing_campaign(tmp_path):
    path=tmp_path/'campaign/state.json';path.parent.mkdir();path.write_text('{}')
    with pytest.raises(ValueError,match='retrofit'):
        queue.admit_execution_scope(queue_stub(tmp_path),'b'*64,qualification_only=True)


@pytest.fixture
def bounded(tmp_path,production):
    _,original=production
    args={key:original[key] for key in ('inventory','panel','excluded_process_seeds','seed',
        'motion_root','source_sha256','teacher_sha256','policy_spec_sha256')}
    plan=make_plan(run_id='dagger-leo-reliability-fixture',environments_per_actor=16,**args)
    staged=stage(plan,output_root=tmp_path/'plans',checkpoint=tmp_path/'parent.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path,plan))
    root=tmp_path/'queue';scope=queue.admit_execution_scope(queue_stub(root),'b'*64,qualification_only=True)
    cfg,mse=bound(staged['first_round']),bound(staged['mse_plan'])
    campaign=queue.QualificationOnlyCampaign(cfg,root/'campaign',object(),plan=mse,execution_scope=scope)
    assignments=[]
    for ci,cohort in enumerate(plan['rounds'][0]['epochs'][0]['cohorts'][:2]):
        for gpu,assignment in enumerate(cohort['assignments']):
            for lane in assignment['lanes']:
                episode=f'fixture-{ci}-{gpu}-{lane["lane"]}'
                native=publish(tmp_path/f'{episode}.json',dict(schema='roborender.native-episode-assignment.v1',
                    purpose='training',simulator_seed=assignment['simulator_seed'],motion_id=lane['motion_id'],
                    motion_sha256=lane['sha256'],episode_id=episode))
                reset=publish(tmp_path/f'{episode}-reset.json',dict(schema='roborender.native-episode-reset-evidence.v1',
                    episode_id=episode,assignment_sha256=digest(assignment)))
                assignments.append(dict(**native,reset_evidence=reset))
    jobs=[];parent='d'*64
    for i in range(4):
        candidate=hashlib.sha256(f'fixture-{i}'.encode()).hexdigest()
        jobs.append(publish(tmp_path/f'job-{i}.json',dict(schema='roborender.dagger_learner_candidate_receipt.v3',
            parent_checkpoint_sha256=parent,output_checkpoint_sha256=candidate,train_chunks=32,train_frames=2592,
            replay_manifest_sha256=None,batching_semantics='raven_exact_flattened_bxt',
            optimizer_state_preserved=True,global_step=101+i,parent_global_step=100+i)))
        parent=candidate
    proof=dict(schema='roborender.dagger-full295-two-cohort-qualification.v1',purpose='qualification',
        plan_sha256=digest(plan),parent_checkpoint_sha256='d'*64,source_sha256=plan['source_sha256'],
        native_cohorts=2,optimizer_updates=4,native_initializations_per_cohort=[1,1],vector_steps_per_cohort=[162,162],
        transitions=4*2592,full_batch_environments=32,no_replay=True,history_reset_at_window_boundary=False,
        minimum_host_available_bytes=9<<30,oom_events=0,swap_peak_bytes=0,limit_pressure_events=0,
        worker_uuid_pairs=[['fixture0','fixture1']]*2,h200_uuids=['fixtureH'+str(i) for i in range(4)],
        native_assignment_receipts=assignments,learner_receipts=jobs,completed=True,
        allocation_retained=True,checkpoint_optimizer_lineage_verified=True)
    receipt=dict(schema='roborender.dagger-production-qualification-result.v1',
        run_id=cfg['run_id'],round_id=1,completed_cohorts=2,completed_updates=4,
        qualification=publish(tmp_path/'qualification.json',proof),
        native_phase_barrier=dict(schema='roborender.isaac-phase-barrier.v1',workers=[
            dict(physical_gpu=gpu,gpu_uuid=LEO_GPU_UUIDS[gpu],worker_instance_id=str(gpu)*32,
                source_sha256=plan['source_sha256'],phase='IDLE') for gpu in (0,1)]))
    return campaign,receipt,cfg,mse,scope


def finish_qualification(campaign,receipt,monkeypatch):
    campaign.save('COLLECT_TRAIN')
    monkeypatch.setattr(queue.CoverageMSECampaign,'operation',lambda self,phase:deepcopy(receipt))
    calls=[]
    def tick(self):
        calls.append('qualification')
        actual=self.operation('collect')
        self.save(operation_id=None,native_qualification=bound(actual['qualification']),
            native_qualification_binding=actual['qualification'],qualification_completed=True)
        return dict(self.state)
    monkeypatch.setattr(queue.CoverageMSECampaign,'tick',tick)
    campaign.tick()
    return calls


def test_actual_four_update_receipt_stops_before_next_tick_and_survives_resume(bounded,monkeypatch):
    campaign,receipt,cfg,mse,scope=bounded
    calls=finish_qualification(campaign,receipt,monkeypatch)
    result=campaign.qualification_completion()
    assert result['completed_optimizer_updates']==4
    assert result['full_production_round_completed'] is False
    assert result['native_owners_kept_warm'] is True and result['compute_released'] is False
    campaign.tick()
    resumed=queue.QualificationOnlyCampaign(cfg,campaign.root,object(),plan=mse,execution_scope=scope)
    resumed.tick()
    with pytest.raises(ValueError,match='cannot dispatch'):
        resumed.operation('collect')
    assert calls==['qualification']  # neither resume nor another poll dispatches a fifth update


@pytest.mark.parametrize('fault',['not_quiescent','cold_stopped','wrong_run','extra_update','fake_proof'])
def test_boolean_or_unsafe_native_completion_never_admits(bounded,monkeypatch,fault):
    campaign,receipt,*_=bounded
    if fault=='not_quiescent':receipt['native_phase_barrier']['workers'][0]['phase']='BUSY'
    if fault=='cold_stopped':receipt['owned_actors_stopped']=True
    if fault=='wrong_run':receipt['run_id']='other'
    if fault=='extra_update':receipt['completed_updates']=5
    if fault=='fake_proof':receipt['qualification']=publish(campaign.root/'fake.json',{'completed':True})
    with pytest.raises(ValueError):finish_qualification(campaign,receipt,monkeypatch)
    assert not (campaign.root/'qualification-only-completion.json').exists()


def test_native_receipt_tamper_on_resume_is_not_a_new_launch(bounded,monkeypatch):
    campaign,receipt,*_=bounded
    calls=finish_qualification(campaign,receipt,monkeypatch)
    path=campaign.root/'qualification-only-native-receipt.json'
    value=json.loads(path.read_text());value['completed_updates']=5;path.write_text(json.dumps(value))
    with pytest.raises(ValueError):campaign.tick()
    assert calls==['qualification']


def test_cpu_exit_after_state_write_recovers_completion_without_a_new_tick(bounded,monkeypatch):
    campaign,receipt,cfg,mse,scope=bounded
    calls=finish_qualification(campaign,receipt,monkeypatch)
    # Simulate losing the CPU before publishing the final summary, not a lost
    # native lifecycle: the qualified state and native operation proof persist.
    (campaign.root/'qualification-only-completion.json').unlink()
    resumed=queue.QualificationOnlyCampaign(cfg,campaign.root,object(),plan=mse,execution_scope=scope)
    resumed.tick()
    assert (campaign.root/'qualification-only-completion.json').is_file()
    assert calls==['qualification']


@pytest.mark.parametrize('phase',['collect','evaluate','release','abort','control_only'])
def test_out_of_phase_operation_never_reaches_native_provider(bounded,monkeypatch,phase):
    campaign,*_=bounded
    def forbidden(*args):pytest.fail('out-of-phase native dispatch occurred')
    monkeypatch.setattr(queue.CoverageMSECampaign,'operation',forbidden)
    with pytest.raises(ValueError,match='cannot dispatch'):campaign.operation(phase)


@pytest.mark.parametrize('fault',['round2','later_phase','scope_tamper','external_qualification'])
def test_no_tick_can_escape_pinned_round1_scope(bounded,monkeypatch,fault):
    campaign,*_=bounded
    def forbidden(self):pytest.fail('unsafe state reached the underlying campaign tick')
    monkeypatch.setattr(queue.CoverageMSECampaign,'tick',forbidden)
    if fault=='round2':campaign.state['round_id']=2
    if fault=='later_phase':campaign.state['phase']='ADVANCE'
    if fault=='scope_tamper':Path(campaign.execution_scope['path']).write_text('{}')
    if fault=='external_qualification':campaign.state['native_qualification']={}
    with pytest.raises(ValueError):campaign.tick()
