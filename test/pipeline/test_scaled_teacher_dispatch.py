"""Synthetic CPU transport tests, not native capacity/success evidence."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import fcntl
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(PIPELINE_ROOT/'deployment/rtx_actor'))
from test.deployment.rtx_actor.test_scaled_success_contract import accepted
from pipeline.deployment import scaled_teacher_dispatch as dispatch
from pipeline.deployment.rtx_actor import scaled_success_contract as scaled


@pytest.fixture
def plan(accepted, monkeypatch):
    import production_success_contract as local
    from pipeline.deployment.rtx_actor import production_success_contract as packaged
    for key in ('PANEL_SHA256','PANEL_RAW_SHA256','LEGACY_PANEL_SHA256','LEGACY_PANEL_RAW_SHA256'):
        monkeypatch.setattr(packaged, key, getattr(local,key))
    panel = scaled.build_panel(accepted, batch_size=16)
    owners = [dict(worker_instance_id=str(i+1)*32, source_sha256='a'*64, native_pid=1,
        physical_gpu=i, gpu_uuid=f'GPU-synthetic-{i}') for i in range(2)]
    return dispatch.build_plan(panel,owners,command_prefix='cpu-teacher295',shadow_checkpoint_sha256='b'*64)


def guard(owner, receipt):
    assert receipt['resource_guard'] == {'synthetic_cpu_only':True}


class Transport:
    def __init__(self, plan, root):
        self.plan,self.root=plan,root
        self.rows,self.active,self.calls={},{},[]
        self.lost=False
        self.accept=True
        self.stage_hook=None

    def get_ready(self, owner):
        key=owner['worker_instance_id']
        self.calls.append(('ready',key))
        return dict(schema='roborender.isaac-resident.v1',role='isaac_resident',ok=True,
            **{k:owner[k] for k in dispatch.common.OWNER_KEYS},native_initializations=1,
            phase='RUNNING' if key in self.active else 'IDLE',active_command_id=self.active.get(key))

    def get_command(self, owner, cid):
        self.calls.append(('get',owner['worker_instance_id'],cid))
        return deepcopy(self.rows.get(cid))

    def stage_artifacts(self, owner, artifacts):
        self.calls.append(('stage',owner['worker_instance_id']))
        assert len(artifacts)==2
        for item in artifacts:
            assert dispatch.common.digest(item['value'])==item['raw_sha256']
        if self.stage_hook:self.stage_hook()

    def submit(self, owner, body):
        cid,key=body['command_id'],owner['worker_instance_id']
        entry=next(e for e in self.plan['commands'] if e['body']==body)
        root=self.root/f"cohort-{entry['cohort_index']:03d}"
        assert dispatch.common.read(root/'intent.json')['entry']['body']==body
        assert dispatch.common.read(root/'submit-attempt.json')['body']==body
        assert key not in self.active
        self.calls.append(('post',key,cid))
        if self.accept:
            self.rows[cid]=dict(status='RUNNING',payload_sha256=dispatch.common.digest(body))
            self.active[key]=cid
        if self.lost:raise TimeoutError('synthetic unknown response')
        return dict(schema='roborender.isaac-resident.v1',status='accepted',command_id=cid,
            worker_instance_id=key,payload_sha256=dispatch.common.digest(body))

    def finish(self, index):
        entry=self.plan['commands'][index]; owner=self.plan['owners'][entry['owner_index']]
        cid=entry['body']['command_id']
        self.rows[cid]=dict(status='COMPLETE',payload_sha256=entry['payload_sha256'],receipt=dict(
            complete=True,source_sha256=owner['source_sha256'],worker_instance_id=owner['worker_instance_id'],
            native_pid=1,command_id=cid,native_initializations=1,kit_identity=123,same_kit=True,
            physics_pause_requested=True,scene_initializations=1,phase_resets=1,optimizer_steps=0,
            purpose='test',profile='explicit_evaluation',resource_guard={'synthetic_cpu_only':True},
            result=dict(benchmark=entry['artifacts'][0]['value'],cohort_result={'exported_successes':0})))
        self.active.pop(owner['worker_instance_id'],None)


def setup(plan,tmp_path):
    t=Transport(plan,tmp_path)
    return dispatch.Coordinator(plan,tmp_path,t,resource_validator=guard),t


def posts(t):return [c for c in t.calls if c[0]=='post']


def test_plan_keeps295physical_cases_and_separates_privileged_input(plan):
    assert len(plan['commands'])==19
    assert [e['owner_index'] for e in plan['commands']]==[i%2 for i in range(19)]
    assert sum(len(c['case_ids']) for c in plan['panel']['cohorts'])==295
    for e in plan['commands']:
        assert e['body']['operation']=='evaluate_scaled_teacher_panel'
        p=e['body']['payload']
        assert p['arm']=='teacher' and p['teacher_action_mode']=='raven_ppo'
        assert 'rendered_observation_config_file' not in p
    assert dispatch.validate_plan(plan)==plan


def test_paired_intents_precede_posts_and_inflight_never_reposts(plan,tmp_path):
    c,t=setup(plan,tmp_path)
    assert [x['status'] for x in c.tick()['workers']]==['SUBMITTED','SUBMITTED']
    c.tick()
    assert len(posts(t))==2
    for post in posts(t):
        before=t.calls[:t.calls.index(post)]
        assert sum(x==('get',post[1],post[2]) for x in before)==2


@pytest.mark.parametrize('did_accept',[False,True])
def test_ambiguous_submit_stops_peer_and_reconciles_without_replay(plan,tmp_path,did_accept):
    c,t=setup(plan,tmp_path);t.lost=True;t.accept=did_accept
    assert c.tick()['workers'][0]['status']=='TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER'
    assert len(posts(t))==1
    t.lost=False
    resumed=dispatch.Coordinator(plan,tmp_path,t,resource_validator=guard)
    value=resumed.tick()
    assert value['workers'][0]['status']==('IN_FLIGHT' if did_accept else 'UNKNOWN_REQUIRES_RECONCILIATION')
    assert sum(p[2]==plan['commands'][0]['body']['command_id'] for p in posts(t))==1
    if not did_accept:assert len(posts(t))==1


def test_stop_before_or_during_stage_never_posts(plan,tmp_path):
    c,t=setup(plan,tmp_path)
    t.stage_hook=lambda:dispatch.common.immutable(tmp_path/'STOP',{'user_requested':True})
    assert c.tick()['workers'][0]['status']=='NOT_STARTED_STOPPED'
    assert not posts(t)


def test_lost_accepted_command_and_changed_plan_fail_closed(plan,tmp_path):
    c,t=setup(plan,tmp_path);c.tick()
    t.rows.clear();t.active.clear()
    with pytest.raises(ValueError,match='disappeared'):c.tick()
    changed=deepcopy(plan);changed['commands'][0]['body']['payload']['arm']='bc'
    with pytest.raises(ValueError,match='changed'):dispatch.validate_plan(changed)


def test_first_pair_must_both_complete_before_cohort_two(plan,tmp_path,monkeypatch):
    monkeypatch.setattr(dispatch.teacher,'validate_result',lambda panel,value:deepcopy(value))
    c,t=setup(plan,tmp_path);c.tick();t.finish(0);c.tick()
    value=c.tick()
    assert any(r['status']=='WAITING_INITIAL_PAIR_INTEGRITY' for r in value['workers'])
    assert len(posts(t))==2
    t.finish(1);c.tick();c.tick()
    assert any(p[2]==plan['commands'][2]['body']['command_id'] for p in posts(t))


def test_lock_prevents_duplicate_controller(plan,tmp_path):
    c,t=setup(plan,tmp_path)
    with (tmp_path/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):c.tick()
    assert not posts(t)


@pytest.mark.parametrize('field,bad', [('source_sha256','c'*64),('worker_instance_id','d'*32),
    ('native_pid',2),('command_id','wrong'),('native_initializations',2),('same_kit',False),
    ('optimizer_steps',1),('physics_pause_requested',False),('purpose','train')])
def test_receipt_identity_and_nontraining_guards(plan,tmp_path,field,bad,monkeypatch):
    c,t=setup(plan,tmp_path);c.tick();t.finish(0)
    entry=plan['commands'][0];row=t.rows[entry['body']['command_id']]
    monkeypatch.setattr(dispatch.teacher,'validate_result',lambda panel,value:deepcopy(value))
    row['receipt'][field]=bad
    with pytest.raises(ValueError,match='receipt'):dispatch.validate_complete(plan,entry,row,guard)


def test_partial_then_exact295_counts_preserve_failures(plan,tmp_path,monkeypatch):
    # Scientific validation is tested separately; these synthetic results test
    # only durable controller coverage/counting and must never qualify hardware.
    monkeypatch.setattr(dispatch.teacher,'validate_result',lambda panel,value:deepcopy(value))
    seen=[]
    def aggregate(panel,values):
        assert [v['benchmark']['cohort_index'] for v in values]==list(range(19))
        seen.append(True)
        return dict(synthetic_cpu_only=True,completed=295,successes=0)
    monkeypatch.setattr(dispatch.teacher,'aggregate_results',aggregate)
    c,t=setup(plan,tmp_path);c.tick();t.finish(0);t.finish(1)
    partial=c.tick()
    assert partial['completed']==32 and partial['successes']==0 and partial['partial']
    while not partial['complete']:
        c.tick()
        for entry in plan['commands']:
            row=t.rows.get(entry['body']['command_id'])
            if row and row['status']=='RUNNING':t.finish(entry['cohort_index'])
        partial=c.tick()
    assert partial['completed']==295 and partial['success_rate']==0 and not partial['partial']
    assert len(posts(t))==19 and seen
    assert len(list(tmp_path.glob('cohort-*/commit.json')))==19
