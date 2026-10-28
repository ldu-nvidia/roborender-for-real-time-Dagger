"""CPU-only gates for opt-in Image3F-only DAgger; never native qualification."""
from test.deployment.source_paths import PIPELINE_ROOT, REPO_ROOT
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
import ast
import inspect
from pathlib import Path
import textwrap
import threading

import numpy as np

import pytest

from pipeline.dagger.rtx.code.src.dagger_rtx.observation_contract import (
    GENERATED_ONLY_CONTRACT, configure_native_observations, require_generated_source,
    validate_generated_only,
)
from pipeline.dagger.services.contracts import ContractError
from pipeline.dagger.services.render_student import LiveRenderStudentBackend, FakeRenderStudentBackend, _Image3FLoopbackClient
from pipeline.dagger.services.render_student_server import RenderStudentHTTPServer, NotReadyError
from pipeline.dagger.services.store import StateStore
from pipeline.dagger.services.client import ServiceError
from test.pipeline.test_dagger_h200_services import running_service


@pytest.mark.parametrize('field,value', [('mode','mixed'), ('renderer_steps',50),
    ('renderer_cfg',True), ('teacache_threshold',0), ('raytraced_policy_fallback',True),
    ('renderer','fake'), ('unknown',1)])
def test_contract_rejects_any_downgrade(field,value):
    value_contract=deepcopy(GENERATED_ONLY_CONTRACT); value_contract[field]=value
    with pytest.raises(ValueError): validate_generated_only(value_contract)


def test_phase_local_mixture_selection_is_explicit_and_restored():
    trainer=SimpleNamespace(vg_raytraced_mix_prob=.5)
    configure_native_observations(trainer,GENERATED_ONLY_CONTRACT)
    assert trainer.vg_raytraced_mix_prob==0
    require_generated_source(trainer._roborender_student_observation_contract,'generated_controls')
    with pytest.raises(ValueError):
        require_generated_source(trainer._roborender_student_observation_contract,'raytraced')
    configure_native_observations(trainer,GENERATED_ONLY_CONTRACT)
    configure_native_observations(trainer,None)
    assert trainer.vg_raytraced_mix_prob==.5
    require_generated_source(None,'raytraced')  # Legacy is not reinterpreted.


def test_resident_boundary_clears_source_override_before_reset(monkeypatch):
    root=REPO_ROOT
    monkeypatch.syspath_prepend(str(root/'pipeline/dagger/rtx/code/src'))
    source=root/'pipeline/deployment/rtx_actor/resident_isaac_trainer.py'
    tree=ast.parse(source.read_text())
    method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                and node.name=='_prepare_training')
    scope={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
    trainer=SimpleNamespace(vg_raytraced_mix_prob=.5,env=object())
    configure_native_observations(trainer,GENERATED_ONLY_CONTRACT)
    def at_reset(env):
        assert env is trainer.env and trainer.vg_raytraced_mix_prob==.5
        assert trainer._roborender_student_observation_contract is None
        raise StopIteration('stop before any native reset')
    owner=SimpleNamespace(trainer=trainer,_require_collection_teacher=lambda:None,
        cold_phase_state=SimpleNamespace(restore=at_reset))
    with pytest.raises(StopIteration):scope['_prepare_training'](owner,42)


def register_gate(root, backend):
    store=StateStore(root)
    with store.transaction() as db:
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            ('run-generated','["actor0"]',81,'a'*64,'b'*64,'b'*64,1,.75,'collecting',100,0,0))
        db.execute('INSERT INTO run_observation_contracts VALUES(?,?)',
            ('run-generated',json.dumps(GENERATED_ONLY_CONTRACT)))
    obj=SimpleNamespace(store=store,backend=backend,observation_contracts={})
    key=SimpleNamespace(run_id='run-generated')
    return obj,key


def test_h200_generated_gate_blocks_raw_and_fake(tmp_path):
    obj,key=register_gate(tmp_path/'fake',FakeRenderStudentBackend())
    with pytest.raises(NotReadyError,match='genuine'):
        RenderStudentHTTPServer.admit_observation_sources(obj,[(key,{'observation_source':'generated_controls'})])
    backend=object.__new__(LiveRenderStudentBackend)
    backend.ready=Mock(return_value=(True,'CPU fixture only'))
    backend.require_generated_observations=Mock()
    obj,key=register_gate(tmp_path/'live-shape',backend)
    for _ in range(2):
        RenderStudentHTTPServer.admit_observation_sources(obj,[(key,{'observation_source':'generated_controls'})])
    backend.ready.assert_called_once()
    backend.require_generated_observations.assert_called_once_with(GENERATED_ONLY_CONTRACT)
    with pytest.raises(ContractError,match='generated_only'):
        RenderStudentHTTPServer.admit_observation_sources(obj,[(key,{'observation_source':'raytraced'})])


def test_missing_renderer_rejected_without_cached_downgrade(tmp_path):
    backend=object.__new__(LiveRenderStudentBackend)
    backend.ready=Mock(return_value=(False,'renderer missing'))
    obj,key=register_gate(tmp_path,backend)
    with pytest.raises(NotReadyError,match='renderer unavailable'):
        RenderStudentHTTPServer.admit_observation_sources(obj,[(key,{'observation_source':'generated_controls'})])
    assert not obj.observation_contracts


def test_registered_policy_is_immutable_and_retry_cannot_omit_it(tmp_path,monkeypatch):
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'CPU checkpoint fixture')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv('DAGGER_CHECKPOINT_SOURCE_ROOT',str(tmp_path))
    state=tmp_path/'service'
    with running_service('registry',state) as registry, running_service('coordinator',state) as coordinator:
        registry.post('/v1/checkpoints/register-local',dict(path=str(checkpoint),sha256=sha,metadata={}))
        body=dict(run_id='run-generated',actor_ids=['actor0'],rollout_steps=81,config_sha256='a'*64,
            initial_checkpoint_sha256=sha,validation_interval_iterations=100,
            student_observation=deepcopy(GENERATED_ONLY_CONTRACT))
        assert coordinator.post('/v1/runs',body)['ok'] is True
        assert coordinator.post('/v1/runs',body)['idempotent'] is True
        omitted=deepcopy(body);omitted.pop('student_observation')
        with pytest.raises(ServiceError,match='409'):
            coordinator.post('/v1/runs',omitted)
        changed=deepcopy(body);changed['student_observation']['raytraced_policy_fallback']=True
        with pytest.raises(ServiceError,match='400'):
            coordinator.post('/v1/runs',changed)


def attestation(contract=None):
    return dict(schema='roborender.image3f.observation-attestation.v1',
        contract=deepcopy(contract or GENERATED_ONLY_CONTRACT),
        backend_class='runtime.image3f_deploy.Image3FSession',generate_source_sha256='a'*64)


@pytest.mark.parametrize('reply', [{'ok':True}, {'ok':False},
    {'ok':True,'attestation':dict(attestation(),backend_class='FakeSession')}])
def test_old_or_fake_renderer_handshake_cannot_admit_new_run(reply):
    client=_Image3FLoopbackClient('127.0.0.1',18500)
    client._exchange=Mock(return_value=reply)
    with pytest.raises(RuntimeError):client.observation_contract()


def test_actual_renderer_settings_must_match_not_only_ping():
    backend=object.__new__(LiveRenderStudentBackend)
    backend.image3f=SimpleNamespace(observation_contract=Mock(return_value=attestation()))
    assert backend.require_generated_observations(GENERATED_ONLY_CONTRACT)==attestation()
    for field,value in [('renderer_steps',50),('renderer_cfg',5.),('teacache_threshold',.2)]:
        wrong=deepcopy(GENERATED_ONLY_CONTRACT);wrong[field]=value
        backend.image3f.observation_contract.return_value=attestation(wrong)
        with pytest.raises(ValueError):backend.require_generated_observations(GENERATED_ONLY_CONTRACT)


class FixtureSession:
    def __init__(self):
        self.config={'generation_contract':{'steps':5}}
        self.cfg_scale=1.;self.tea_cache_threshold=.3

    def generate(self, request, *, output_type='rgb'):
        return self.pipe(request, num_inference_steps=5)


def test_renderer_rpc_reads_actual_session_fields_and_method_without_cuda():
    source=PIPELINE_ROOT/'dagger/image3f_server.py'
    tree=ast.parse(source.read_text())
    helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                and node.name=='session_observation_contract')
    scope=dict(ast=ast,hashlib=hashlib,inspect=inspect,textwrap=textwrap,Image3FSession=FixtureSession)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])),str(source),'exec'),scope)
    session=FixtureSession()
    assert scope['session_observation_contract'](session)['contract']==GENERATED_ONLY_CONTRACT
    session.cfg_scale=5.
    assert scope['session_observation_contract'](session)['contract']['renderer_cfg']==5.
    session.config['generation_contract']['steps']=50
    with pytest.raises(RuntimeError,match='step count'):scope['session_observation_contract'](session)
    with pytest.raises(RuntimeError,match='genuine'):scope['session_observation_contract'](object())
    session.config['generation_contract']['steps']=5
    scope['inspect']=SimpleNamespace(getsource=lambda _: 'def generate(self,request):\n    return self.pipe(request,num_inference_steps=50)')
    with pytest.raises(RuntimeError,match='step count'):scope['session_observation_contract'](session)


def test_renderer_failure_never_falls_back_to_native_rgb():
    backend=object.__new__(LiveRenderStudentBackend)
    backend._lock=threading.Lock()
    backend.image3f=SimpleNamespace(render=Mock(side_effect=OSError('renderer lost')))
    backend._load_policy=Mock()
    arrays={f'{kind}__{view}':np.zeros((2,3,3),np.uint8)
        for view in ('head_color','hand_left','hand_right')
        for kind in ('depth_ctrl','rgb_robot','anchor_rgb','raytraced_rgb')}
    with pytest.raises(OSError,match='renderer lost'):
        backend.infer({'observation_source':'generated_controls','prompt':'CPU fixture'},arrays,'a'*64)
    backend._load_policy.assert_not_called()
