"""Synthetic CPU gate tests, not new native/H200 capacity certification."""
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline.bc_training.handoff_common import publish
from pipeline.dagger import coverage_queue as queue
from pipeline.dagger.coverage_production import digest
from pipeline.deployment import capacity_admission as capacity
from test.pipeline.test_capacity_admission import evidence
from test.pipeline.test_dagger_coverage_production import production


def learner_probe():
    files={name:'b'*64 for name in capacity.LEARNER_SOURCE_PATHS}
    return dict(schema=capacity.LEARNER_SCHEMA,passed=True,batch=32,frames=2592,
        checkpoint_sha256='c'*64,optimizer_steps=2,full_batch=True,microbatching=False,
        activation_checkpointing=False,production_checkpoint_saved=False,
        production_promoted=False,compute_released=False,
        current_source_paired_native_and_end_to_end_qualification_still_required=True,
        source_files=files,peak_allocated_bytes=69*capacity.GIB,
        peak_reserved_bytes=70*capacity.GIB,device_total_bytes=140*capacity.GIB,
        renderer_gateway_allowance_bytes=25*capacity.GIB,
        minimum_joint_headroom_fraction=.2,projected_joint_fraction=95/140,
        metrics=[dict(step=i,seconds=1.,mse=.1,gradient_norm=.2) for i in range(2)])


def source_manifest(probe):
    files=[dict(path=k,sha256=v,bytes=1,executable=False)
        for k,v in sorted(probe['source_files'].items())]
    return dict(schema='roborender.source-snapshot.v1',source_sha256=digest(files),
        file_count=len(files),files=files)


def check_probe(probe,manifest=None,**overrides):
    kwargs=dict(environments_per_actor=16,checkpoint_sha256='c'*64,
        source_manifest=manifest or source_manifest(learner_probe()))
    kwargs.update(overrides)
    return capacity.validate_learner(probe,**kwargs)


def test_full_batch_component_proof_does_not_admit_native_runtime():
    value=learner_probe()
    assert check_probe(value)==value
    assert value['current_source_paired_native_and_end_to_end_qualification_still_required']


@pytest.mark.parametrize('fault',['batch','parent','source','microbatch','recompute','promoted',
    'save','release','missing_joint_gate','memory','allowance','allocated','ratio','metrics','nan','no_backward'])
def test_unsafe_learner_probe_rejected(fault):
    value=learner_probe()
    if fault=='batch':value['batch']=64
    if fault=='parent':value['checkpoint_sha256']='d'*64
    if fault=='source':value['source_files'][next(iter(value['source_files']))]='d'*64
    if fault=='microbatch':value['microbatching']=True
    if fault=='recompute':value['activation_checkpointing']=True
    if fault=='promoted':value['production_promoted']=True
    if fault=='save':value['production_checkpoint_saved']=True
    if fault=='release':value['compute_released']=True
    if fault=='missing_joint_gate':value['current_source_paired_native_and_end_to_end_qualification_still_required']=False
    if fault=='memory':
        value['peak_reserved_bytes']=100*capacity.GIB
        value['projected_joint_fraction']=125/140
    if fault=='allowance':value['renderer_gateway_allowance_bytes']=24*capacity.GIB
    if fault=='allocated':value['peak_allocated_bytes']=71*capacity.GIB
    if fault=='ratio':value['projected_joint_fraction']=.1
    if fault=='metrics':value['metrics']=value['metrics'][:1]
    if fault=='nan':value['metrics'][0]['mse']=float('nan')
    if fault=='no_backward':value['metrics'][0]['gradient_norm']=0
    with pytest.raises(ValueError):check_probe(value)


@pytest.fixture
def scaled_queue(tmp_path,monkeypatch,production):
    _,plan=production
    monkeypatch.setattr(queue,'RUNTIME_ROOT',tmp_path)
    campaign=tmp_path/'fresh';campaign.mkdir()
    probe=learner_probe();manifest=source_manifest(probe)
    source=publish(campaign/'source.json',manifest)
    sampling=dict(schema='roborender.coverage-sampling-plan.v3',
        partitions={'train':{'cohorts':[{'process_seed':10}]},
                    'validation':{'cohorts':[{'process_seed':11}]}},
        selection_uses_labels_rewards_or_loss=False,
        motion_inventory_sha256=queue.digest(queue.bound(plan['inventory'])))
    admission=dict(schema='roborender.bc-data-admission.v3',purpose='production',
        production_admitted=True,metadata_independence_passed=True,motion_support_matches=True,
        test_payloads_opened=False,closed_loop_test_executed=False,missing_provenance={},warnings=[],
        protocol=dict(sampling_plan_canonical_sha256=digest(sampling),
            motion_inventory_canonical_sha256=sampling['motion_inventory_sha256']))
    paired=capacity.derive(evidence(tmp_path,manifest['source_sha256'],production_memory_policy=True),
        source_sha256=manifest['source_sha256'])
    selection=publish(campaign/'selected.json',{'checkpoint':{'sha256':'c'*64}})
    return dict(schema=queue.SCALED_FRESH_SCHEMA,run_id='dagger-leo-capacity-test',
        root=str(campaign/'dagger/run'),campaign_root=str(campaign),
        bc_selection_path=selection['path'],bc_success_path=str(campaign/'success.json'),
        runtime_bindings_path=str(campaign/'dagger/bindings.json'),inventory=plan['inventory'],
        panel=plan['panel'],excluded_process_seeds=[10,11],seed=42,motion_root='/motions',
        source_sha256=manifest['source_sha256'],teacher_sha256='d'*64,policy_spec_sha256='e'*64,
        wandb_entity='nvidia',wandb_key_file=None,source_manifest=source,
        bc_sampling_plan=publish(campaign/'sampling.json',sampling),
        bc_data_admission=publish(campaign/'admission.json',admission),environments_per_actor=16,
        capacity_admission=publish(campaign/'paired.json',paired),
        learner_capacity=publish(campaign/'learner.json',probe))


def test_scaled_fresh_queue_accepts_real_bound_component_evidence(scaled_queue):
    assert queue.validate_queue(scaled_queue)==scaled_queue


@pytest.mark.parametrize('fault',['missing_learner','missing_native','batch','old_parent','source',
    'bc_seed_overlap','v2_size_change','tamper'])
def test_scaled_queue_cannot_silently_bypass_capacity(scaled_queue,tmp_path,fault):
    value=deepcopy(scaled_queue)
    if fault=='missing_learner':value.pop('learner_capacity')
    if fault=='missing_native':value.pop('capacity_admission')
    if fault=='batch':value['environments_per_actor']=32
    if fault=='source':value['source_sha256']='f'*64
    if fault=='bc_seed_overlap':value['excluded_process_seeds']=[10,12]
    if fault=='old_parent':
        value['bc_selection_path']=publish(Path(value['campaign_root'])/'other.json',
            {'checkpoint':{'sha256':'f'*64}})['path']
    if fault=='v2_size_change':
        value['schema']=queue.FRESH_SCHEMA
        value.pop('capacity_admission');value.pop('learner_capacity')
    if fault=='tamper':
        path=Path(value['learner_capacity']['path']);path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):queue.validate_queue(value)


def test_legacy_fresh_smoke_still_uses_two_per_actor(scaled_queue):
    value=deepcopy(scaled_queue);value['schema']=queue.FRESH_SCHEMA
    value.pop('capacity_admission');value.pop('learner_capacity');value['environments_per_actor']=2
    assert queue.validate_queue(value)==value


def test_programmatic_staging_checks_capacity_before_any_execution(scaled_queue,monkeypatch):
    value=deepcopy(scaled_queue);value['environments_per_actor']=32
    def forbidden(*args,**kwargs):
        pytest.fail('unsafe queue reached handoff/operations')
    monkeypatch.setattr(queue,'validate_bc_handoff',forbidden)
    with pytest.raises(ValueError):queue.prepare_campaign(value)
