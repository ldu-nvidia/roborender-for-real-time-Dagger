"""CPU protocol/real ledger tests; no native qualification is claimed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import Mock, patch

import pytest

from pipeline.dagger.coverage_production import make_plan, stage, validate_plan, FIXED100_SCHEMA
from pipeline.dagger.coverage_campaign import CoverageMSECampaign
from pipeline.dagger.beta_campaign import next_round_config
from pipeline.dagger.services import mse_protocol
from pipeline.dagger.services.fixed100_protocol import SCHEMA, validate_protocol, accept_final_candidate
from pipeline.dagger.services.contracts import canonical_json, file_sha256
from pipeline.dagger.services.store import StateStore
from pipeline.dagger.services.server import Handler
from pipeline.dagger.services.client import ServiceError
from pipeline.dagger.rtx.code.src.dagger_rtx.observation_contract import GENERATED_ONLY_CONTRACT
from test.pipeline.test_dagger_coverage_production import production, rendering_dependencies
from test.pipeline.test_dagger_h200_services import running_service


@pytest.fixture
def fixed_plan(production):
    _, old = production
    return make_plan(**{key: old[key] for key in (
        'run_id', 'inventory', 'panel', 'excluded_process_seeds', 'seed', 'motion_root',
        'source_sha256', 'teacher_sha256', 'policy_spec_sha256')}, environments_per_actor=16, fixed100=True)


def staged(fixed_plan, tmp_path):
    result = stage(fixed_plan, output_root=tmp_path/'plan', checkpoint=tmp_path/'parent.pt',
                   checkpoint_sha256='d'*64, rendered_observation_config=rendering_dependencies(tmp_path, fixed_plan))
    return (json.loads(Path(result['first_round']['path']).read_bytes()),
            json.loads(Path(result['mse_plan']['path']).read_bytes()))


def test_fixed100_plan_has_four_exact_budgets_and_no_validation(fixed_plan, tmp_path):
    assert validate_plan(fixed_plan) == fixed_plan
    assert fixed_plan['schema'] == FIXED100_SCHEMA
    assert 'stopping' not in fixed_plan and fixed_plan['max_updates_per_round'] == 100
    assert fixed_plan['qualification_updates_count_toward_round_budget'] is True
    assert [r['teacher_beta'] for r in fixed_plan['rounds']] == [.75, .5, .25, 0]
    for row in fixed_plan['rounds']:
        assert 'heldout' not in row and len(row['epochs']) == 5
        cohorts = [c for e in row['epochs'] for c in e['cohorts']]
        assert len(cohorts) == 50
        assert [c['start_iteration'] for c in cohorts] == list(range(0, 100, 2))
        assert all(len(a['lanes']) == 16 for c in cohorts for a in c['assignments'])
        assert row['success_test']['checkpoint_binding'] == 'final_update_100'
    config, campaign = staged(fixed_plan, tmp_path)
    assert config['execution']['window_count'] == 100
    assert config['student_observation'] == GENERATED_ONLY_CONTRACT
    assert all(row['max_iterations'] == 100 for row in campaign['rounds'])
    assert not list((tmp_path/'plan').rglob('*heldout*'))
    assert not list((tmp_path/'plan').rglob('collection-plan.json'))
    bad = deepcopy(fixed_plan); bad['max_updates_per_round'] = 101
    with pytest.raises(ValueError):
        validate_plan(bad)


def test_campaign_prepare_skips_population_and_validation_worker(fixed_plan, tmp_path):
    config, plan = staged(fixed_plan, tmp_path)
    ops = Mock()
    ops.start.return_value = 'prepare-operation'
    ops.poll.return_value = dict(source_sha256=config['source_sha256'],
        resource_shape='custom.gpu.4xh200', reservation='cosmos-solution-architect-xcwtffiq',
        allow_burst=False,genuine_backends=True,owned_control_plane=True,
        service_state_identity='f'*64,gateway_pids=[1,2,3,4],token_file='/fixture-token')
    controller = Mock(); factory = Mock(return_value=controller)
    campaign = CoverageMSECampaign(config, tmp_path/'campaign', ops, plan=plan, controller_factory=factory)
    with patch.object(campaign, 'admit_population', side_effect=AssertionError('validation forbidden')):
        assert campaign.tick()['phase'] == 'REGISTER'
        assert campaign.tick()['phase'] == 'COLLECT_TRAIN'
        context = campaign.operation_context()
    assert 'heldout_collection' not in context
    assert factory.call_args.kwargs['mse_worker_command'] == []
    protocol = json.loads(Path(context['mse_protocol_path']).read_bytes())
    assert protocol['schema'] == SCHEMA
    assert not any('validation' in key or 'patience' in key for key in protocol)
    next_config = next_round_config(config,'a'*64,'e'*64,'/fixture/final100.pt',[])
    assert next_config['student']['initial_checkpoint_sha256'] == 'e'*64
    assert next_config['execution']['window_count'] == 100
    assert next_config['teacher_beta'] == .5


def test_terminal_completed_prefix_rebuilds_receipt_without_native_replay(fixed_plan,tmp_path):
    from pipeline.deployment import resident_production_collection as collection
    config,plan=staged(fixed_plan,tmp_path)
    campaign=CoverageMSECampaign(config,tmp_path/'campaign',Mock(),plan=plan)
    protocol=campaign.protocol_path()
    token=tmp_path/'token';token.write_text('CPU-fixture-token')
    final=hashlib.sha256(b'accepted-100').hexdigest()
    terminal=dict(status='selected_pending_test',stop_reason='fixed_100_updates',
                  selection=dict(best_checkpoint_sha256=final))
    client=Mock()
    client.get.side_effect=lambda path: ({'mse_round':terminal} if '/mse-rounds/' in path else
                                       {'run':{'status':'selected_pending_test'},'iterations':[]})
    site=SimpleNamespace(config=config,rid=1,run=config['run_id'],round=tmp_path/'round-runtime',
        token=token,remote=tmp_path/'remote',publish=Mock(),read=Mock(return_value={}),
        barrier=Mock(return_value={'owned_actors_stopped':True}),pool_action=Mock(),
        request=dict(round_config_path=str(tmp_path/'plan/first-round.json'),
                     mse_protocol_path=str(protocol),mse_protocol_file_sha256=file_sha256(protocol)))
    def verified_prefix(directory,coverage,rid,epoch,index,parent,rows):
        # The existing completed_prefix validates immutable receipts/ledger
        # before this orchestration branch; this fixture isolates the restart.
        start=(epoch-1)*20+index*2
        sha=hashlib.sha256(f'accepted-{start+2}'.encode()).hexdigest()
        return ({'output_checkpoint_sha256':sha},
                {'coverage_epoch':epoch,'cohort_index':index,'windows':[{'iteration_id':start+1},{'iteration_id':start+2}]})
    with patch.object(collection,'ServiceClient',return_value=client), patch.object(collection,'completed_prefix',side_effect=verified_prefix) as prefix:
        result=collection.collect(site)
        assert prefix.call_count==50
        assert collection.collect(site)==result  # same bytes, no duplicate work
    assert result['completed_updates']==100 and result['completed_cohorts']==50
    assert result['selected_checkpoint_sha256']==final
    site.pool_action.assert_not_called()


def protocol_fixture(root):
    store = StateStore(root)
    parent = root/'parent.pt'; parent.write_bytes(b'CPU fixture parent')
    sha = file_sha256(parent)
    protocol = dict(schema=SCHEMA,run_id='fixed100-fixture',round_id=1,
        initial_checkpoint_sha256=sha,interval_iterations=100,max_iterations=100,
        full_batch_environments=32,rollout_steps=81,checkpoint_selection='final_update_100',
        final_test=dict(scenario_manifest_sha256='d'*64,
                        scenario_ids=[f'test-{i:03d}' for i in range(295)],observation_mode='generated'))
    with store.transaction() as db:
        db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?)',(sha,parent.stat().st_size,'{}',str(parent),0))
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('fixed100-fixture',canonical_json([f'actor-{i}' for i in range(32)]),81,'a'*64,
                    sha,sha,0,None,'created',100,0,0))
        db.execute('INSERT INTO run_observation_contracts VALUES (?,?)',
                   ('fixed100-fixture',canonical_json(GENERATED_ONLY_CONTRACT)))
    return store, protocol


def registration(protocol):
    return dict(protocol=protocol,protocol_sha256=hashlib.sha256(canonical_json(protocol).encode()).hexdigest())


@pytest.mark.parametrize('final_failure',[None,'checkpoint_collision','partial_batch','optimizer_discontinuity'])
def test_real_worker_commit_stops_at100_no_validation_or_update101(tmp_path,final_failure,monkeypatch):
    forbidden=Mock(side_effect=AssertionError('fixed100 must never evaluate held-out validation'))
    monkeypatch.setattr('pipeline.dagger.services.heldout_mse.load_validation_manifest',forbidden)
    monkeypatch.setattr('pipeline.dagger.services.heldout_mse.RoundMSESelection.record',forbidden)
    store, protocol = protocol_fixture(tmp_path)
    handler = Handler.__new__(Handler)
    handler.server = SimpleNamespace(store=store,root=tmp_path,jobs_lock=threading.Lock(),worker=Path('/fixture-worker'))
    with patch('pipeline.dagger.services.heldout_mse.load_validation_manifest', side_effect=AssertionError('no validation')):
        assert mse_protocol.register(store, registration(protocol))['status'] == 'active'
        assert mse_protocol.register(store, registration(protocol))['idempotent'] is True
        with running_service('coordinator',tmp_path) as coordinator:
            coordinator.post('/v1/rounds/start',dict(run_id=protocol['run_id'],round_id=1,beta=.75,
                checkpoint_sha256=protocol['initial_checkpoint_sha256']))
    current = protocol['initial_checkpoint_sha256']
    for iid in range(1,101):
        directory = tmp_path/f'job-{iid}'; directory.mkdir()
        output = directory/'checkpoint.pt'; output.write_bytes(f'CPU fixture update {iid}'.encode())
        candidate = file_sha256(output)
        result = dict(output_checkpoint_sha256=candidate,output_checkpoint_path=str(output),
            parent_checkpoint_sha256=current,train_chunks=32,train_frames=2592,
            batching_semantics='raven_exact_flattened_bxt',optimizer_state_preserved=True,
            parent_global_step=500+iid-1,global_step=500+iid,replay_manifest_sha256=None)
        if iid==100 and final_failure=='partial_batch':result['train_chunks']=31
        if iid==100 and final_failure=='optimizer_discontinuity':result['global_step']+=1
        result_path = directory/'result.json'; result_path.write_text(json.dumps(result))
        request = dict(job_id=f'job-{iid}',run_id=protocol['run_id'],round_id=1,
            input_checkpoint_sha256=current,train_chunk_ids=['fixture-chunk'],
            parameters=dict(iteration_id=iid,final_validation_iteration=100,validation_interval_iterations=100))
        job_path=directory/'request.json';job_path.write_text(json.dumps(request))
        with store.transaction() as db:
            if iid==100 and final_failure=='checkpoint_collision':
                db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?)',
                    (candidate,output.stat().st_size,'{"different_registration":true}',str(output),0))
            db.execute('INSERT INTO learner_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (f'job-{iid}',protocol['run_id'],1,current,'["fixture-chunk"]','queued',canonical_json(request),
                 None,str(directory/'worker.log'),None,0,0))
            db.execute('INSERT INTO learner_job_iterations VALUES (?,?,?,?)',(f'job-{iid}',protocol['run_id'],1,iid))
            db.execute("UPDATE round_iterations SET status='learning' WHERE iteration_id=?",(iid,))
        with patch('pipeline.dagger.services.server.subprocess.Popen', return_value=SimpleNamespace(pid=123,wait=lambda:0)):
            handler._run_worker(f'job-{iid}',job_path,result_path,directory/'worker.log')
        context=mse_protocol.context(store,protocol['run_id'],1)['mse_round']
        if iid==100 and final_failure:
            assert context['status']=='active' and context['selection']['history']==[]
            assert context['selection']['best_checkpoint_sha256']==protocol['initial_checkpoint_sha256']
            with store.connection() as db:
                assert db.execute('SELECT status FROM runs').fetchone()[0]=='learner_failed'
                assert db.execute('SELECT COUNT(*) FROM round_iterations').fetchone()[0]==100
                failed=db.execute('SELECT status,result_json FROM learner_jobs WHERE job_id="job-100"').fetchone()
                assert failed['status']=='failed'
                assert 'admission_error' in json.loads(failed['result_json'])
                if final_failure!='checkpoint_collision':
                    assert db.execute('SELECT COUNT(*) FROM checkpoints WHERE sha256=?',(candidate,)).fetchone()[0]==0
            return
        assert context['status'] == ('active' if iid<100 else 'selected_pending_test')
        assert context['selection']['history'] == ([] if iid<100 else
            [dict(iteration=100,result=dict(checkpoint_sha256=candidate))])
        current=candidate
    assert context['selection']['best_checkpoint_sha256'] == current
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM round_iterations').fetchone()[0] == 100
        assert db.execute('SELECT COUNT(*) FROM learner_jobs WHERE status="succeeded"').fetchone()[0] == 100
    with running_service('learner',tmp_path,worker=Path(__file__)) as learner:
        with patch('pipeline.dagger.services.server.subprocess.Popen',side_effect=AssertionError('no duplicate update')):
            assert learner.post('/v1/jobs',request)['idempotent'] is True
            with pytest.raises(ServiceError):
                learner.post('/v1/jobs',{**request,'job_id':'job-101'})
    forbidden.assert_not_called()
    with pytest.raises(ValueError,match='forbids validation'):
        mse_protocol.record(store,dict(run_id=protocol['run_id'],round_id=1,iteration_id=100,
            result=dict(checkpoint_sha256=current)))
    with running_service('evaluation',tmp_path) as evaluation, running_service('coordinator',tmp_path) as coordinator:
        evaluation.post('/v2/evaluations',dict(evaluation_id='fixed-final-test',run_id=protocol['run_id'],round_id=1,
            purpose='test',checkpoint_sha256=current,metric='raven_goal_reached_rate',**protocol['final_test']))
        with store.transaction() as db:
            for sid in protocol['final_test']['scenario_ids']:
                db.execute('INSERT INTO success_evaluation_outcomes VALUES (?,?,?,?,?,?,?,?,?,?)',
                    ('fixed-final-test',sid,'outcome-'+sid,'{}',0,0.,10,'done',0,0))
            db.execute("UPDATE success_evaluations SET status='complete'")
        payload=dict(run_id=protocol['run_id'],round_id=1,evaluation_id='fixed-final-test')
        coordinator.post('/v1/mse-rounds/finish-test',payload)
        assert coordinator.post('/v1/mse-rounds/finish-test',payload)['idempotent'] is True
        next_protocol={**protocol,'round_id':2,'initial_checkpoint_sha256':current}
        coordinator.post('/v1/mse-rounds/register',registration(next_protocol))
        coordinator.post('/v1/rounds/start',dict(run_id=protocol['run_id'],round_id=2,beta=.5,checkpoint_sha256=current))
        state=coordinator.get('/v1/status/'+protocol['run_id'])
        assert state['run']['active_checkpoint_sha256']==current
        assert state['iterations'][-1]['iteration_id']==1 and state['iterations'][-1]['round_id']==2


@pytest.mark.parametrize('key,value',[('max_iterations',99),('max_iterations',101),
    ('full_batch_environments',16),('rollout_steps',80),('checkpoint_selection','best_mse')])
def test_fixed_budget_cannot_be_overridden(tmp_path,key,value):
    _,protocol=protocol_fixture(tmp_path)
    protocol[key]=value
    with pytest.raises(ValueError):validate_protocol(protocol)
