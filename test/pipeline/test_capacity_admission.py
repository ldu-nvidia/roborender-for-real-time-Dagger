import json
from copy import deepcopy
import pytest
from pipeline.bc_training.handoff_common import publish,binding
from pipeline.deployment.capacity_admission import derive,validate,validate_production_memory_policy,GIB


def evidence(tmp_path,source_sha256='a'*64,production_memory_policy=False):
    result=dict(passed=True,failure=None,mode='teacher-capacity',physical_gpus=[0,1],
        exit_codes=[0,0],deployment_host='leo-rtx-6k',capacity_steps=10,conservative_capacity_guard=True,
        source_sha256=source_sha256,memory_gib_per_actor=21,num_envs_per_gpu=16)
    native=dict(result='PASS',mode='teacher-capacity',num_envs=16,environment_step_calls=10,
        physical_transition_rows=160,optimizer_steps=0,chunks_published=0,production_qualified=False)
    if production_memory_policy:
        scopes={}
        for gpu in (0,1):
            name='docker-'+str(gpu)*64+'.scope'
            scopes[str(gpu)]=dict(schema='roborender.native-systemd-scope-admission.v1',
                scope=name,cgroup_path='/sys/fs/cgroup/system.slice/'+name,
                passed=True,returncode=0,compute_release_authorized=False,global_host_policy_modified=False,
                properties=dict(ControlGroup='/system.slice/'+name,MemoryMax=str(21*GIB),
                    MemorySwapMax='0',CPUQuotaPerSecUSec='12s'))
        result.update(native_thp_policy='disable-v1',systemd_scope_admissions=scopes,
            production_memory_policy_matched=True)
        native.update(native_thp_policy=dict(schema='roborender.native-thp-policy.v1',policy='disable-v1',
            native_pid=1,disabled_after=1,global_host_policy_modified=False),
            cgroup_memory={'memory.max':str(21*GIB),'memory.swap.max':'0','memory.swap.current':'0'})
    pair=publish(tmp_path/'pair.json',result)
    refs=[publish(tmp_path/f'gpu{gpu}.json',native) for gpu in (0,1)]
    sample=dict(stat=dict(anon=10*GIB,shmem=GIB,kernel=GIB),events=dict(oom=0,oom_kill=0),
        **{'memory_swap.current':0})
    rows=[dict(elapsed_s=20+i,mem_available_kib=20*1024**2,container_memory=[sample,sample],
        gpu_samples=['0, 20000, 97887, 50, 150','1, 22000, 97887, 50, 150']) for i in range(3)]
    path=tmp_path/'resources.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    return dict(pair_result=pair,native_receipts=refs,resources=binding(path))


def test_legacy_memory_proof_does_not_admit_new_disable_v1_owner(tmp_path):
    old=derive(evidence(tmp_path),source_sha256='a'*64)
    with pytest.raises(ValueError):validate_production_memory_policy(old)


@pytest.mark.parametrize('fault',[None,'thp','scope_swap','native_swap','missing_scope','duplicate_scope'])
def test_actual_production_memory_policy_gate(tmp_path,fault):
    e=evidence(tmp_path,production_memory_policy=True)
    if fault:
        pair=json.loads(open(e['pair_result']['path']).read())
        if fault=='thp':pair['native_thp_policy']='inherited'
        if fault=='scope_swap':pair['systemd_scope_admissions']['0']['properties']['MemorySwapMax']='infinity'
        if fault=='missing_scope':pair['systemd_scope_admissions'].pop('1')
        if fault=='duplicate_scope':pair['systemd_scope_admissions']['1']=pair['systemd_scope_admissions']['0']
        e['pair_result']=publish(tmp_path/'bad-policy-pair.json',pair)
        if fault=='native_swap':
            native=json.loads(open(e['native_receipts'][0]['path']).read())
            native['cgroup_memory']['memory.swap.max']='max'
            e['native_receipts'][0]=publish(tmp_path/'bad-policy-native.json',native)
    admission=derive(e,source_sha256='a'*64)
    if fault:
        with pytest.raises(ValueError):validate_production_memory_policy(admission)
    else:assert validate_production_memory_policy(admission)==admission


def test_actual_paired_capacity_does_not_claim_h200_qualification(tmp_path):
    value=derive(evidence(tmp_path),source_sha256='a'*64)
    assert validate(value,source_sha256='a'*64,environments_per_actor=16)==value
    assert value['full_h200_qualification_required'] is True
    with pytest.raises(ValueError):validate(value,source_sha256='a'*64,environments_per_actor=32)


def test_lower_probe_cap_retains_four_gib_margin(tmp_path):
    e=evidence(tmp_path)
    pair=json.loads(open(e['pair_result']['path']).read());pair['memory_gib_per_actor']=20
    e['pair_result']=publish(tmp_path/'lower-cap.json',pair)
    assert derive(e,source_sha256='a'*64)['probe_memory_gib_per_actor']==20
    path=tmp_path/'resources.jsonl';rows=[json.loads(r) for r in path.read_text().splitlines()]
    rows[0]['container_memory'][0]['stat']['anon']=14*GIB
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n');e['resources']=binding(path)
    with pytest.raises(ValueError):derive(e,source_sha256='a'*64)


@pytest.mark.parametrize('fault',['source','failure','swap','private','host','gpu','tamper','short'])
def test_unsafe_evidence_cannot_promote_a_batch(tmp_path,fault):
    e=evidence(tmp_path)
    if fault in ('source','failure'):
        pair=json.loads(open(e['pair_result']['path']).read())
        pair.update(source_sha256='b'*64) if fault=='source' else pair.update(passed=False)
        e['pair_result']=publish(tmp_path/'bad-pair.json',pair)
    else:
        path=tmp_path/'resources.jsonl';rows=[json.loads(r) for r in path.read_text().splitlines()]
        if fault=='swap':rows[0]['container_memory'][0]['memory_swap.current']=1
        if fault=='private':rows[0]['container_memory'][0]['stat']['anon']=16*GIB
        if fault=='host':rows[0]['mem_available_kib']=8*1024**2
        if fault=='gpu':rows[0]['gpu_samples'][0]='0, 90000, 97887, 50, 150'
        if fault=='short':rows=rows[:1]
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        if fault=='tamper':path.write_text(path.read_text()+'\n')
        else:e['resources']=binding(path)
    with pytest.raises(ValueError):derive(e,source_sha256='a'*64)
