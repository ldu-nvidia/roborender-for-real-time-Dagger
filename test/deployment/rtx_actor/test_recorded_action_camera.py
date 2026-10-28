from test.deployment.source_paths import REPO_ROOT
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test.deployment.rtx_actor.test_leo_capacity_probe import Batch
from test.deployment.rtx_actor.test_warm_camera_diagnostic import plan
from recorded_action_camera import ACK, MODE, SCHEMA, RecordedActions, read_reference, validate_reference
from leo_capacity_probe import run_recorded_action_camera_soak
from warm_camera_diagnostic import canonical


def envelope(array):
    array=np.ascontiguousarray(array,dtype='<f4');raw=array.tobytes()
    return dict(dtype='float32',shape=list(array.shape),sha256=hashlib.sha256(raw).hexdigest(),
                data_b64=base64.b64encode(raw).decode())


def reference():
    teacher=np.tile(np.arange(26,dtype=np.float32),(16,1))
    selected=teacher.copy();selected[0,0]=-1
    return dict(schema=SCHEMA,reference_run_id='old-diagnostic',num_envs=16,reference_steps=129,
        reference_assignment_sha256=hashlib.sha256(canonical(plan()['arms']['0'][-1]['assignment'])).hexdigest(),
        production_replay_allowed=False,simulator_restore_claimed=False,
        rows=[dict(step=step,selected_actions=envelope(selected),teacher_actions=envelope(teacher),
            actor_obs=envelope(np.zeros((16,113),dtype=np.float32)),
            outcomes=[dict(done=False,timeout=False) for _ in range(16)],
            selected_sources=['student']+['teacher']*15) for step in range(129)])


@pytest.mark.parametrize('mutation',[
    lambda x:x.update(production_replay_allowed=True),
    lambda x:x.update(simulator_restore_claimed=True),
    lambda x:x['rows'].pop(),
    lambda x:x['rows'][0].update(step=1),
    lambda x:x['rows'][0]['actor_obs'].update(sha256='f'*64),
    lambda x:x['rows'][0]['selected_actions'].update(shape=[26,16]),
    lambda x:x['rows'][0]['selected_sources'].__setitem__(0,'teacher'),
    lambda x:x['rows'][0]['outcomes'][0].update(done=1),
    lambda x:x['rows'][0].update(teacher_actions=envelope(np.zeros((16,26),dtype=np.float32))),
])
def test_corrupt_reference_rejected(mutation):
    value=reference();mutation(value)
    with pytest.raises(ValueError):validate_reference(value)


def test_file_scene_and_digest_binding(tmp_path):
    value=reference();path=tmp_path/'trace.json';path.write_bytes(canonical(value))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert isinstance(read_reference(path,sha,value['reference_assignment_sha256']),RecordedActions)
    with pytest.raises(ValueError,match='scene'):read_reference(path,sha,'f'*64)
    with pytest.raises(ValueError,match='digest'):read_reference(path,'f'*64,value['reference_assignment_sha256'])
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError,match='regular'):read_reference(link,sha,value['reference_assignment_sha256'])


class ActionBatch(Batch):
    num_envs=16
    def step_batch(self,selected,teacher):
        expected=teacher.copy()
        if len(self.steps)<129:expected[0,0]=-1
        assert np.array_equal(selected,expected)
        self.steps.append(selected.copy())
        if self.fail_step:raise RuntimeError('uncertain native result')
        return [SimpleNamespace(reward=0.,done=False,timeout=False) for _ in range(16)]


def execute(batch,ref,events,calls):
    return run_recorded_action_camera_soak(batch,
        SimpleNamespace(begin=lambda:calls.append('begin'),finish=lambda:calls.append('finish')),
        steps=324,acknowledgement=ACK,report=events.append,reference=ref)


def test_initial_mismatch_prevents_first_step():
    batch=ActionBatch();batch.arrays['actor_obs'][0]=1
    events,calls=[],[]
    with pytest.raises(RuntimeError,match='initial state mismatch'):
        execute(batch,RecordedActions(validate_reference(reference()),'a'*64),events,calls)
    assert batch.steps==[] and calls==['begin','finish']
    assert not any(r['phase']=='env_step_start' for r in events)


def test_129_recorded_then_195_teacher_no_learning_and_continuous_history():
    batch=ActionBatch();events,calls=[],[]
    result=execute(batch,RecordedActions(validate_reference(reference()),'a'*64),events,calls)
    assert len(batch.steps)==324 and calls==['begin','finish']
    assert result['schema']=='roborender.leo.'+MODE+'.v1'
    assert result['optimizer_steps']==result['chunks_published']==0
    assert not result['production_qualified'] and not result['H200_renderer_tested']
    audit=result['recorded_action_diagnostic']
    assert audit['recorded_vector_steps']==129 and audit['outcome_mismatches']==0
    assert len(audit['numerical_comparisons'])==129
    comparisons=[r for r in events if r['phase']=='recorded_action_comparison']
    assert [r['completed_vector_steps'] for r in comparisons]==list(range(129))
    assert len(batch.bindings)==1
    assert batch.captures[129][0].episode_step_index==129


def test_uncertain_step_no_retry():
    batch=ActionBatch(fail_step=True);events,calls=[],[]
    with pytest.raises(RuntimeError,match='uncertain'):
        execute(batch,RecordedActions(validate_reference(reference()),'a'*64),events,calls)
    assert len(batch.steps)==1 and calls==['begin','finish']
    assert events[-1]['phase']=='env_step_start'


def test_later_numeric_and_outcome_drift_is_exposed():
    ref=RecordedActions(validate_reference(reference()),'a'*64)
    batch=ActionBatch();captures=[SimpleNamespace(arrays=batch.arrays) for _ in range(16)]
    teacher=np.tile(batch.arrays['teacher_action'],(16,1))
    ref.select(0,captures,teacher)
    batch.arrays['actor_obs'][0]=2
    ref.select(1,captures,teacher)
    ref.observe(1,[SimpleNamespace(done=True,timeout=False) for _ in range(16)])
    assert ref.metrics[1]['actor_max_abs_error']==2
    assert ref.summary()['outcome_mismatches']==16
    assert ref.summary()['simulator_restore_claimed'] is False


def test_launcher_requires_exclusive_mode_and_ro_mount(tmp_path):
    from qualify_native_isaac6 import command
    value=reference();trace=tmp_path/'trace.json';trace.write_bytes(canonical(value))
    warm=tmp_path/'plan.json';warm.write_bytes(canonical(plan()))
    (tmp_path/'port-manifest.json').write_text('{"files":[]}')
    root=REPO_ROOT
    options=dict(physical_gpu=0,deployment_host='leo-rtx-6k',capacity_steps=324,
        benchmark_barrier=tmp_path,warm_camera_plan=warm,warm_camera_source_sha256='a'*64,
        recorded_action_reference=trace,run_id='rtx-leo-migration-action-test')
    with pytest.raises(ValueError,match='exclusive'):
        command('test',tmp_path,tmp_path,root/'pipeline/dagger/rtx/code',16,mode='teacher-camera-soak',**options)
    cmd=command('test',tmp_path,tmp_path,root/'pipeline/dagger/rtx/code',16,mode=MODE,**options)
    assert f'type=bind,src={trace},dst=/run/recorded-action-reference.json,readonly' in cmd
    assert 'ROBORENDER_LEO_CAPACITY_ACK='+ACK in cmd
    assert not any('H200_TOKEN' in item or 'DAGGER_H200_BASE' in item for item in cmd)
    assert cmd[cmd.index('--network')+1]=='bridge'
