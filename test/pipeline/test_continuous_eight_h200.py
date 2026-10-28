"""CPU-only eight-worker routing/admission tests; never live qualification."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from pipeline.bc_training.handoff_common import bound,publish
from pipeline.dagger.coverage_production import (
    make_plan,stage,digest,validate_plan,validate_continuous_qualification,make_continuous_segment)
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    validate_config,gateway_slots,gateway_pids_for_actor,gateway_identities_for_actor,verify_actor_gateways)
from pipeline.dagger.rtx.code.src.dagger_rtx.production_segment import validate_segment
from pipeline.dagger.rtx.code.src.dagger_rtx.episode_continuity import validate_window
from test.pipeline.test_dagger_coverage_production import production,rendering_dependencies
from test.pipeline.test_dagger_continuous_plan import continuous_evidence
from test.pipeline.test_dagger_continuous_actor import fake_driver,sha
from test.pipeline.test_dagger_scaled_queue import scaled_queue
from pipeline.dagger import coverage_queue


@pytest.fixture
def continuous_plan(production):
    _,base=production
    return make_plan(**{k:base[k] for k in ('run_id','inventory','panel','excluded_process_seeds',
        'seed','motion_root','source_sha256','teacher_sha256','policy_spec_sha256')},
        environments_per_actor=16,fixed100=True,continuous_round=True,h200_workers=8)


@pytest.fixture
def config8(continuous_plan,tmp_path):
    identities=[dict(slot=i,pid=7000+i,gateway_instance_id=hashlib.sha256(f'gateway{i}'.encode()).hexdigest(),
        gpu_uuid=f'GPU-CPU-FIXTURE-{i}',source_sha256=continuous_plan['source_sha256']) for i in range(8)]
    staged=stage(continuous_plan,output_root=tmp_path/'stage',checkpoint=tmp_path/'parent.pt',
        checkpoint_sha256='d'*64,gateway_pids=[r['pid'] for r in identities],gateway_identities=identities,
        rendered_observation_config=rendering_dependencies(tmp_path,continuous_plan))
    return bound(staged['first_round'])


def continuity(index,batch=16):
    return dict(schema='roborender.native-window-cursor-continuity.v1',window_index=index,
        start=[dict(lane=i,episode_id=f'episode-{i:03d}-00000000',episode_step_index=81*index) for i in range(batch)],
        end=[dict(lane=i,episode_id=f'episode-{i:03d}-00000000',episode_step_index=81*(index+1)) for i in range(batch)],
        native_resets=[])


@pytest.fixture
def evidence8(continuous_evidence,config8,tmp_path):
    e=continuous_evidence;proof=e['proof']
    proof.update(runtime_config=publish(tmp_path/'runtime8.json',config8),
        h200_gateway_identities=config8['h200']['gateway_identities'],
        h200_uuids=[r['gpu_uuid'] for r in config8['h200']['gateway_identities']])
    proof['renderer_attachment']=publish(tmp_path/'attachment8.json',dict(
        schema='roborender.retained-eight-renderer-attachment.v1',allocation_name='CPU-only',
        allocation_owner='CPU-only',hostname='CPU-only',node_id='CPU-only',boot_id='CPU-only',
        allocation_retained=True,compute_release_authorized=False,render_requests=0,renderer_restarts=0,
        renderers=[dict(slot=i,pid=10000+i,gpu_uuid=row['gpu_uuid'],identity=dict(boot_id='CPU-only'),
            source_sha256='a'*64,generate_source_sha256='b'*64)
            for i,row in enumerate(config8['h200']['gateway_identities'])]))
    for gpu,ref in enumerate(proof['native_prefix_receipts']):
        row=bound(ref);state=row['actor_state']
        # Explicit CPU synthetic admission fields, NOT live certification.
        population=dict(schema='roborender.raven-paired-population-curriculum.v1',
            population='shared_paired_native_population_v1',vector_reset_barriers=325,
            global_reset_events=10,physical_resets_added=0,final_state_sha256='f'*64)
        row['population_curriculum']=population
        state['population_curriculum']=population
        from pipeline.dagger.rtx.code.src.dagger_rtx.reset_inventory import binding, descriptors
        reset=binding(descriptors(bound(e['plan']['inventory']),e['plan']['motion_root']),gpu*16)
        row['reset_inventory']=reset
        state['reset_inventory']=reset
        state.update(config_sha256=digest(config8),h200_gateway_identities=gateway_identities_for_actor(config8,gpu))
        for index,window in enumerate(state['windows']):window['episode_continuity']=continuity(index)
        proof['native_prefix_receipts'][gpu]=publish(tmp_path/f'prefix8-{gpu}.json',row)
    from pipeline.dagger.services.learner_order import PROTOCOL,_ids_sha
    for index,ref in enumerate(proof['learner_receipts']):
        row=bound(ref);ids=[f'cpu-fixture-update{index}-lane{i:03d}' for i in range(32)]
        row.update(train_chunk_ids=ids,learner_order_receipt=dict(schema=PROTOCOL,
            environment_indices=list(range(32)),input_chunk_ids_sha256=_ids_sha(ids),
            permuted_chunk_ids_sha256=_ids_sha(ids),temporal_order='unchanged',rollout_steps=81),
            learner_rng_receipt=dict(schema='roborender.learner-rng-continuation.v1',
                global_step=row['global_step'],parent_checkpoint_sha256=row['parent_checkpoint_sha256']))
        proof['learner_receipts'][index]=publish(tmp_path/f'learner8-{index}.json',row)
    return e


def test_versioned_eight_plan_and_runtime_map_are_exact(config8,continuous_plan):
    assert validate_plan(continuous_plan)==continuous_plan
    assert validate_config(config8,runtime=True)==config8
    assert config8['schema'].endswith('.v2') and continuous_plan['schema'].endswith('.v2')
    assert continuous_plan['topology']['h200_workers']==8
    for gpu in (0,1):
        assert gateway_slots(config8,gpu)==list(range(gpu*4,gpu*4+4))
        rows=gateway_identities_for_actor(config8,gpu)
        assert gateway_pids_for_actor(config8,gpu)==[r['pid'] for r in rows]
        assert verify_actor_gateways(config8,gpu,{'gateways':rows})==rows
    assert not set(gateway_pids_for_actor(config8,0))&set(gateway_pids_for_actor(config8,1))
    assert config8['learner_protocol']=='roborender.raven-environment-randperm.v1'
    assert config8['behavior_selection_protocol']=='roborender.raven-native-vector-selection.v1'
    assert config8['population_curriculum_protocol']=='roborender.raven-paired-population-curriculum.v1'
    assert config8['reset_inventory_protocol']=='roborender.raven-full-reset-inventory.v1'


@pytest.mark.parametrize('fault',['missing_population','peer_population','missing_order','bad_order','missing_rng',
                                  'missing_reset_inventory','wrong_reset_offset','truncated_reset_inventory'])
def test_new_protocol_qualification_requires_actual_bound_evidence(evidence8,tmp_path,fault):
    proof=deepcopy(evidence8['proof'])
    if fault in ('missing_population','peer_population','missing_reset_inventory','wrong_reset_offset','truncated_reset_inventory'):
        row=bound(proof['native_prefix_receipts'][0])
        if fault=='missing_population':row.pop('population_curriculum')
        elif fault=='peer_population':row['population_curriculum']['final_state_sha256']='e'*64
        elif fault=='missing_reset_inventory':row.pop('reset_inventory')
        elif fault=='wrong_reset_offset':row['reset_inventory']['reset_lane_offset']=16
        else:row['reset_inventory']['reset_motion_count']=16
        proof['native_prefix_receipts'][0]=publish(tmp_path/'corrupt-native.json',row)
    else:
        row=bound(proof['learner_receipts'][0])
        if fault=='missing_order':row.pop('learner_order_receipt')
        elif fault=='bad_order':row['learner_order_receipt']['environment_indices']=[0]*32
        else:row.pop('learner_rng_receipt')
        proof['learner_receipts'][0]=publish(tmp_path/'corrupt-learner.json',row)
    with pytest.raises((ValueError,RuntimeError)):
        validate_continuous_qualification(evidence8['plan'],proof,proof['parent_checkpoint_sha256'])


@pytest.mark.parametrize('fault',['legacy','four','shape','overlap','same_uuid','same_server','source','pid','truncated_id'])
def test_reject_mislabeled_or_overlapping_fleet(config8,fault):
    c=deepcopy(config8);h=c['h200']
    if fault=='legacy':c['schema']='roborender.cross_gpu_dagger_continuous_fixed100.leo.v1'
    elif fault=='four':h['render_student_worker_count']=4
    elif fault=='shape':h['resource_shape']='custom.gpu.4xh200'
    elif fault=='overlap':h['actor_gateway_slots']['1']=[0,1,2,3]
    elif fault=='same_uuid':h['gateway_identities'][4]['gpu_uuid']=h['gateway_identities'][0]['gpu_uuid']
    elif fault=='same_server':h['gateway_identities'][4]['gateway_instance_id']=h['gateway_identities'][0]['gateway_instance_id']
    elif fault=='source':h['gateway_identities'][4]['source_sha256']='f'*64
    elif fault=='pid':h['gateway_identities'][4]['pid']=10
    else:h['gateway_identities'][4]['gateway_instance_id']='a'*32
    with pytest.raises(ValueError):validate_config(c,runtime=True)


def test_live_route_must_match_own_four_servers(config8):
    with pytest.raises(ValueError):verify_actor_gateways(config8,1,{'gateways':gateway_identities_for_actor(config8,0)})
    bad=deepcopy(gateway_identities_for_actor(config8,1));bad[-1]['gateway_instance_id']='e'*64
    with pytest.raises(ValueError):verify_actor_gateways(config8,1,{'gateways':bad})


def test_segment_rejects_cross_version_mix(config8,continuous_plan):
    refs=[dict(physical_gpu_index=g,path=f'/native/g{g}.json',sha256=digest(a))
        for g,a in enumerate(continuous_plan['rounds'][0]['native_round']['assignments'])]
    segment=make_continuous_segment(continuous_plan,round_id=1,input_checkpoint_sha256='d'*64,
        mse_protocol_file_sha256='e'*64,motion_assignments=refs,
        coverage_plan_sha256=config8['coverage_plan']['raw_sha256'])
    assert validate_segment(segment,config8,continuous_plan)==segment
    segment['schema']='roborender.dagger-continuous-round.v1'
    with pytest.raises(ValueError):validate_segment(segment,config8,continuous_plan)


def test_eight_prefix_admits_actual_disjoint_fleet_and_cursor_chain(evidence8):
    e=evidence8
    assert validate_continuous_qualification(e['plan'],e['proof'],'d'*64)==e['proof']


@pytest.mark.parametrize('fault',['missing_runtime','four_uuid','wrong_gpu_group','missing_cursor','reset_at162','lost_native_reason','wrong_config','renderer_restart'])
def test_eight_prefix_rejects_unproven_route_or_cursor(evidence8,tmp_path,fault):
    e=evidence8;proof=deepcopy(e['proof'])
    if fault=='missing_runtime':proof.pop('runtime_config')
    elif fault=='four_uuid':proof['h200_uuids']=proof['h200_uuids'][:4]
    elif fault=='renderer_restart':
        attachment=bound(proof['renderer_attachment']);attachment['renderer_restarts']=1
        proof['renderer_attachment']=publish(tmp_path/'bad-attachment.json',attachment)
    else:
        row=bound(proof['native_prefix_receipts'][1]);state=row['actor_state']
        if fault=='wrong_gpu_group':state['h200_gateway_identities']=proof['h200_gateway_identities'][:4]
        elif fault=='wrong_config':state['config_sha256']='f'*64
        elif fault=='missing_cursor':state['windows'][2].pop('episode_continuity')
        elif fault=='reset_at162':state['windows'][2]['episode_continuity']['start'][0]['episode_step_index']=0
        else:state['windows'][1]['episode_continuity']['end'][0]['episode_id']='episode-000-00000001'
        proof['native_prefix_receipts'][1]=publish(tmp_path/'bad8.json',row)
    with pytest.raises(ValueError):validate_continuous_qualification(e['plan'],proof,'d'*64)


def test_four_update_driver_v2_requires_cursor_evidence_before_barrier(tmp_path):
    driver,calls,offsets,barriers,gates=fake_driver(tmp_path)
    driver.segment['schema']='roborender.dagger-continuous-round.v2'
    with pytest.raises(ValueError,match='continuity'):driver.run()
    assert not barriers and len(calls)==3


def test_v2_real_driver_four_windows_preserves_162_boundary(tmp_path):
    driver,calls,offsets,barriers,gates=fake_driver(tmp_path)
    driver.segment['schema']='roborender.dagger-continuous-round.v2'
    original=driver.runner.run_window
    def window(index,manage_lifecycle):
        result=original(index,manage_lifecycle);result['episode_continuity']=continuity(index);return result
    driver.runner.run_window=window
    driver.diagnostic_scope={'CPU_fixture':True};driver.diagnostic_scope_sha256='f'*64
    driver._wait_continuous_qualification=lambda checkpoint:driver.state.update(qualification_proof={'CPU_fixture':True})
    result=driver.run()
    assert result['phase']=='DIAGNOSTIC_COMPLETE' and len(barriers)==64
    assert result['windows'][2]['episode_continuity']['start'][0]['episode_step_index']==162
    assert result['windows'][3]['episode_continuity']['end'][0]['episode_step_index']==324
    assert len(calls)==12 and result['fifth_window_started'] is False


def test_v2_queue_is_explicit(scaled_queue):
    queue=deepcopy(scaled_queue);queue['schema']=coverage_queue.CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA
    assert coverage_queue.validate_queue(queue)==queue


def test_actual_window_runner_records_natural_resets_across_all_four_boundaries():
    from dagger_rtx.window_runner import BatchWindowRunner,RolloutLifecycle
    class Owner:
        num_envs=16
        def begin_windows(self,rows):return [{'ok':True} for row in rows]
    class NativeCommitFixture:
        num_envs=16
        def __init__(self):self.steps=0
        def run_batch(self,keys):
            step=self.steps;self.steps+=1
            return dict(phase='COMPLETE',local_commits=[dict(sealed=keys[i].rollout_step_index==80,
                result=dict(done=step==80 and i==0,timeout=step==161 and i==1)) for i in range(16)],
                remote_acknowledgements=[{} for _ in range(16)])
    native=NativeCommitFixture();lifecycles=[]
    runner=BatchWindowRunner(Owner(),native,run_id='CPU-continuity',round_id=1,actor_prefix='gpu0',
        lifecycle=RolloutLifecycle(begin=lambda:lifecycles.append('begin'),finish=lambda:lifecycles.append('finish')))
    evidence=[];previous=None
    for index in range(4):
        row=runner.run_window(index)['episode_continuity']
        validate_window(row,batch=16,index=index,previous_end=previous)
        evidence.append(row);previous=row['end']
    assert native.steps==324 and len(lifecycles)==8
    assert evidence[0]['native_resets']==[dict(lane=0,step=80,done=True,timeout=False)]
    assert evidence[1]['native_resets']==[dict(lane=1,step=80,done=False,timeout=True)]
    assert evidence[2]['start'][1]['episode_step_index']==0
    assert evidence[2]['start'][2]['episode_step_index']==162
    assert evidence[3]['end'][2]['episode_step_index']==324
    assert evidence[3]['end'][1]['episode_id'].endswith('00000001')
    bad=deepcopy(evidence[1]);bad['native_resets'][0]['timeout']=False
    with pytest.raises(ValueError,match='done/timeout'):validate_window(bad,batch=16,index=1)


def test_provider_builds_v2_proof_from_real_validation_path(evidence8,config8,tmp_path,monkeypatch):
    from pipeline.deployment import continuous_production_collection as module
    e=evidence8;proof=e['proof'];commands=[];owners=[];prefixes=[]
    for gpu,ref in enumerate(proof['native_prefix_receipts']):
        row=bound(ref);row['actor_state']['source_sha256']=config8['source_sha256']
        owners.append(dict(worker_instance_id=row['worker_instance_id'],native_pid=row['native_pid'],
            source_sha256=config8['source_sha256'],gpu_uuid=f'GPU-RTX-{gpu}'))
        commands.append(dict(gpu=gpu,command_id=row['command_id'],
            payload=dict(motion_assignment_sha256=row['motion_assignment_sha256'])))
        prefixes.append(dict(actor_state=row['actor_state'],resource_guard=row['resource_guard']))
    site=SimpleNamespace(config=config8,run=config8['run_id'],rid=1,
        request=dict(round_config_path=proof['runtime_config']['path']),
        renderer_attachment=lambda **kwargs:bound(proof['renderer_attachment']),
        verify_retained=lambda:dict(gpu_uuids=proof['h200_uuids']))
    site.pool_action=lambda action:dict(workers=[dict(owner,phase='RUNNING',active_command_id=c['command_id'])
        for owner,c in zip(owners,commands)])
    intent=dict(commands=commands,owners=owners,segment_sha256=row['segment_sha256'])
    monkeypatch.setattr(module,'copy_native_assignments',lambda *args:proof['native_assignment_receipts'])
    monkeypatch.setattr(module,'online_prefix',lambda *args:bound(proof['online_metrics_receipt']))
    monkeypatch.setattr(module,'learner_prefix',lambda *args:proof['learner_receipts'])
    gates=[];monkeypatch.setattr(module,'publish_gate',lambda s,i,r:gates.append(bound(r)))
    out=module.admit_prefix(site,e['plan'],intent,prefixes,None,tmp_path/'provider8')
    assert bound(out)['h200_gateway_identities']==config8['h200']['gateway_identities']
    assert gates==[bound(out)]
