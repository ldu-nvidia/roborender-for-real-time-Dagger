"""CPU fixtures only: batching must preserve every native provenance check."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from pipeline.deployment.native_metadata_transfer import fetch_native_json
from pipeline.deployment.resident_heldout_collection import admit_native_cohort
from pipeline.deployment.resident_production_collection import copy_native_assignments
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical


class LocalSite:
    def __init__(self): self.calls=[]
    def command(self, argv, host, *, data=None):
        assert host=='beta'
        self.calls.append(json.loads(data))
        return subprocess.run(argv,input=data,capture_output=True,check=True,timeout=10).stdout
    def publish(self,path,raw,mode=0o444):
        assert mode==0o444
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): assert path.read_bytes()==raw
        else:path.write_bytes(raw)


def test_byte_preservation_and_bounded_grouping(tmp_path):
    site=LocalSite();names=[f'episode-{i}.json' for i in range(130)]
    for n in names:(tmp_path/n).write_bytes(b'{ "exact" : "spacing" }\n')
    result=fetch_native_json(site,tmp_path,names+[names[0]])
    assert len(result)==130 and [len(c['names']) for c in site.calls]==[64,64,2]
    assert all(raw==b'{ "exact" : "spacing" }\n' for raw in result.values())


@pytest.mark.parametrize('name',['../secret.json','/secret.json','payload.npz','x/y.json',None,'x'*256+'.json'])
def test_reject_escape_or_nonmetadata_before_transport(tmp_path,name):
    site=LocalSite()
    with pytest.raises(ValueError):fetch_native_json(site,tmp_path,[name])
    assert not site.calls


@pytest.mark.parametrize('kind',['link','large','directory','nonobject','missing','fifo'])
def test_remote_rejects_nonregular_or_unbounded_files(tmp_path,kind):
    import os
    p=tmp_path/'receipt.json'
    if kind=='link':
        target=tmp_path/'target';target.write_text('{}');p.symlink_to(target)
    elif kind=='large':p.write_bytes(b' '*65536)
    elif kind=='directory':p.mkdir()
    elif kind=='nonobject':p.write_text('[]')
    elif kind=='fifo':os.mkfifo(p)
    with pytest.raises(subprocess.CalledProcessError):fetch_native_json(LocalSite(),tmp_path,[p.name])


@pytest.mark.parametrize('change',['order','sha','extra','missing','base64','oversized','nonobject'])
def test_corrupt_response_rejected_before_publication(tmp_path,change):
    rows=[dict(name=n,sha256=hashlib.sha256(b'{}').hexdigest(),data_b64=base64.b64encode(b'{}').decode())
          for n in ('a.json','b.json')]
    if change=='order':rows.reverse()
    elif change=='sha':rows[0]['sha256']='0'*64
    elif change=='extra':rows[0]['extra']=True
    elif change=='missing':rows.pop()
    elif change=='base64':rows[0]['data_b64']='!'
    elif change=='oversized':
        raw=b' '*65536;rows[0].update(data_b64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    elif change=='nonobject':
        raw=b'[]';rows[0].update(data_b64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    site=SimpleNamespace(command=lambda *a,**kw:json.dumps(rows).encode())
    with pytest.raises(ValueError):fetch_native_json(site,tmp_path,['a.json','b.json'])


def fixture(tmp_path):
    site=LocalSite();runtime=tmp_path/'native';directory=runtime/'phases/example/native-assignments'
    directory.mkdir(parents=True)
    assignment=dict(lanes=[dict(lane=0)])
    raw=b'{ "episode_id": "ep-0" }\n';(directory/'ep.json').write_bytes(raw)
    reset=dict(lane=0,episode_id='ep-0',assignment_sha256=hashlib.sha256(canonical(assignment)).hexdigest())
    (directory/'ep.reset.json').write_bytes(canonical(reset))
    evidence=dict(native_assignment_path='/dagger-runtime/phases/example/native-assignments/ep.json',
        native_assignment_sha256=hashlib.sha256(raw).hexdigest(),episode_id='ep-0')
    site.pool=dict(workers=[dict(physical_gpu=0,runtime_root=str(runtime))])
    return site,assignment,evidence,raw,directory


def test_production_copies_identical_receipts_and_keeps_digest_gate(tmp_path):
    site,a,e,raw,_=fixture(tmp_path)
    response=dict(results=[dict(receipt=dict(actor_state=dict(physical_gpu=0),episode_assignments=[e]))])
    commands=[dict(gpu=0,payload=dict(output_name='example'))]
    refs=copy_native_assignments(site,response,commands,dict(assignments=[a]),tmp_path/'out')
    assert len(site.calls)==1 and len(refs)==1
    assert Path(refs[0]['path']).read_bytes()==raw
    e['native_assignment_sha256']='0'*64
    with pytest.raises(ValueError,match='assignment changed'):
        copy_native_assignments(site,response,commands,dict(assignments=[a]),tmp_path/'bad')


@pytest.mark.parametrize('change',['lane','episode_id','assignment_sha256'])
def test_heldout_still_rejects_reset_binding_mismatch(tmp_path,change):
    site,a,e,raw,directory=fixture(tmp_path)
    reset=json.loads((directory/'ep.reset.json').read_bytes())
    reset[change]=1 if change=='lane' else 'wrong'
    (directory/'ep.reset.json').write_bytes(canonical(reset))
    state=dict(phase='COMPLETE',registration_sha256='a'*64,actor_prefix='actor',transition_batches=81,
        physical_transition_rows=81,chunk_ids=['chunk'])
    receipt=dict(complete=True,purpose='validation',optimizer_steps=0,actor_state=state,episode_assignments=[e])
    with pytest.raises(RuntimeError,match='reset evidence'):
        admit_native_cohort(site,receipt,dict(physical_gpu_index=0,cohort_id='c0',lanes=[0]),
            dict(actor_prefix='actor'),a,'example',dict(registration_sha256='a'*64),tmp_path/'out',[],[])


def test_heldout_preserves_actor_and_reset_identity(tmp_path):
    site,a,e,raw,_=fixture(tmp_path);assignments=[];chunks=[]
    state=dict(phase='COMPLETE',registration_sha256='a'*64,actor_prefix='actor',transition_batches=81,
        physical_transition_rows=81,chunk_ids=['chunk'])
    receipt=dict(complete=True,purpose='validation',optimizer_steps=0,actor_state=state,episode_assignments=[e])
    admit_native_cohort(site,receipt,dict(physical_gpu_index=0,cohort_id='c0',lanes=[0]),
        dict(actor_prefix='actor'),a,'example',dict(registration_sha256='a'*64),tmp_path/'out',assignments,chunks)
    assert len(site.calls)==1 and chunks==['chunk'] and assignments[0]['actor_id']=='actor-lane-000'
    assert Path(assignments[0]['native_assignment_path']).read_bytes()==raw
    assert hashlib.sha256(Path(assignments[0]['native_reset_path']).read_bytes()).hexdigest()==assignments[0]['native_reset_sha256']
