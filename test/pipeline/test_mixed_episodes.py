"""Mixed diagnostic contracts; CPU checks do not certify live rendering."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from pipeline.success_rate.mixed_episodes import make_spec, mix_actions, MixedModeBackend, digest
from test.pipeline.test_continuous_bc_launcher import request


def test_mix_changes_only_manipulation_and_never_mutates_inputs():
    teacher, student = torch.randn(32,26), torch.randn(32,11)
    before = teacher.clone()
    mask = torch.arange(32)%2 == 0
    output = mix_actions(teacher,student,mask)
    assert set(output)=={'actions'}
    assert torch.equal(output['actions'][mask,:11],teacher[mask,:11])
    assert torch.equal(output['actions'][~mask,:11],student[~mask])
    assert torch.equal(output['actions'][:,11:],teacher[:,11:])
    assert torch.equal(teacher,before)
    student[0,0]=float('nan')
    with pytest.raises(RuntimeError):mix_actions(teacher,student,mask)


def test_distinct_protocol_same_motion_support_two_full_repeats():
    s=make_spec(request(),'a'*64,seed=20261102,inputs_sha256='b'*64)
    assert s['expected_episodes']==64 and s['episodes_per_lane']==2
    assert s['motions']==request()['motions'] and s['teacher_probability']==.5
    assert s['renderer']=='Image3F5_generated_only' and s['optimizer_steps']==0


def test_policy_only_does_not_load_curriculum_optimizer_or_rng():
    from pipeline.success_rate.mixed_episodes import policy_only
    weights={'weight':torch.ones(2)}
    checkpoint=dict(policy_state_dict=weights,env_state_dict={'termination_level':1},
        optimizer_state_dict={'must_not_load':True},rng_state={'must_not_load':True})
    assert policy_only(checkpoint) is weights
    assert set(policy_only(checkpoint))=={'weight'}


def test_mixed_does_not_relax_pure_student_route():
    from dagger_rtx.phase_modes import resolve
    body=dict(schema='roborender.isaac-phase.v1',mode='STUDENT_TEST',
              operation='evaluate_collection_mixed',parameters=dict(arm='mixed'))
    assert resolve(body)[0]=='STUDENT_TEST'
    for route in ('evaluate_scaled_rendered_panel','evaluate_raven_pooled_rendered'):
        with pytest.raises(ValueError):resolve({**body,'operation':route})
    with pytest.raises(ValueError):resolve({**body,'parameters':dict(arm='bc')})


def test_runtime_failure_is_consumed_and_cannot_replay():
    spec=make_spec(request(),'a'*64,seed=20261102,inputs_sha256='b'*64)
    def fail():raise RuntimeError('render failed before step')
    b=MixedModeBackend(spec,evaluate=fail,pause=lambda:None,paused=lambda:True,
        kit_identity=lambda:1,revalidate=lambda:spec)
    p=dict(schema='roborender.isaac-phase.v1',mode='STUDENT_TEST',
        operation='evaluate_collection_mixed',parameters=dict(arm='mixed',spec_sha256=digest(spec)))
    with pytest.raises(RuntimeError):b.run('run_phase',p,lambda *a,**k:None)
    with pytest.raises(ValueError):b.validate('run_phase',p)


@pytest.mark.parametrize('change', ['optimizer_steps','frozen_student_unchanged','renderer_calls','checkpoint_sha256'])
def test_missing_science_evidence_never_certified(change):
    from pipeline.success_rate.mixed_episodes import CHECKPOINT_SHA256
    r=dict(vector_steps=4,optimizer_steps=0,student_forward_calls=4,renderer_calls=128,
        frozen_student_unchanged=True,checkpoint_sha256=CHECKPOINT_SHA256,
        rendered_observation_receipt=dict(vector_steps_rendered=4,render_requests_completed=128,
            raytraced_fallbacks=0,failed=False))
    assert MixedModeBackend.valid_policy_result(None,r)
    r[change]='corrupted'
    assert not MixedModeBackend.valid_policy_result(None,r)


def test_actual_launcher_has_mixed_targets_and_readonly_inputs(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'pipeline/deployment/rtx_actor'))
    from qualify_native_isaac6 import command
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import ACTOR_SHARDS, SHARDED_ROUTING
    import pipeline.success_rate.mixed_episodes as m
    # Fake weights are permitted ONLY in this CPU launcher contract test.
    cp=tmp_path/'cp';cp.write_bytes(b'cpu fixture')
    monkeypatch.setattr(m,'CHECKPOINT_SHA256',hashlib.sha256(cp.read_bytes()).hexdigest())
    pool=dict(identity={},routing=SHARDED_ROUTING,actor_shards=deepcopy(ACTOR_SHARDS),
        max_inflight_per_endpoint=1,endpoints=[dict(slot=i,host='127.0.0.1',port=18560+i,
            gpu_uuid=f'GPU-{i:08x}-aaaa-bbbb-cccc-dddddddddddd',server_instance=f'{i:032x}',
            source_sha256='a'*64,generate_source_sha256='b'*64,readiness_receipt_sha256='c'*64)
            for i in range(8)])
    inp=tmp_path/'inputs';inp.write_text(json.dumps(dict(renderer=pool,anchors=[])))
    bc=tmp_path/'bc';bc.write_text(json.dumps(request()))
    sha=hashlib.sha256(bc.read_bytes()).hexdigest()
    spec=m.make_spec(request(),sha,seed=20261102,inputs_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())
    sf=tmp_path/'spec';sf.write_text(json.dumps(spec))
    pair=tmp_path/'pair';pair.mkdir()
    (pair/'plan.json').write_text(json.dumps(dict(schema=m.PAIR_SCHEMA,seeds=[20261102,20261103],
        bc_request_sha256=sha,expected_episodes=128,source_sha256='c'*64)))
    overlay=tmp_path/'overlay';overlay.mkdir();(overlay/'port-manifest.json').write_text('{"files":[]}')
    cmd=command('mixed',tmp_path/'out',overlay,tmp_path/'source/pipeline/dagger/rtx/code',32,
        mode='mixed-episodes',deployment_host='leo-rtx-6k',physical_gpu=0,memory_gib=32,
        compact_host_heap=True,seed=20261102,teacher_episode_spec=sf,teacher_bc_request=bc,
        teacher_pair_dir=pair,mixed_inputs=inp,mixed_checkpoint=cp)
    assert 'trainer._target_=pipeline.success_rate.mixed_episodes.MixedEpisodeEvaluator' in cmd
    assert 'simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]' in cmd
    assert '++domain_rand.image_augmentation.enabled=false' in cmd
    assert not any(v.startswith(('env.config.domain_rand.','++env.config.domain_rand.')) for v in cmd)
    assert not any('raytraced_mix_prob=' in v for v in cmd)
    assert any('dst=/mixed-assets/bootstrap.pt,readonly' in v for v in cmd)
    assert cmd[cmd.index('--network')+1]=='host'
