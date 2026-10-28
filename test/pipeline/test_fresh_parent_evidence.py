"""Fresh diagnostic parent admission checks; CPU fixtures are not live proof."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from pipeline.bc_training import train_bc_h200 as bc
from pipeline.bc_training.qualify_raven_bc32_lifecycle import memory_admission

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-dagger-alignment-20261026-7pYLxI/fresh_parent_evidence.py'
if not PATH.is_file():
    pytest.skip('optional run-local fresh-parent evidence helper absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('fresh_parent_evidence_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    expected=dict(bootstrap_sha256='a'*64,optimizer_updates=2,
        fresh_input_protocol=dict(schema='fresh-fixture-not-a-live-receipt',retired_data_used=False))
    device=dict(gpu_free_bytes_before=121<<30,gpu_total_bytes=141<<30,
        host_available_bytes_before=64<<30,cgroup_headroom_bytes_before=32<<30)
    batch=dict(bc.student_lifecycle_batch_fields(bc.REPLAY_STUDENT_LIFECYCLE,32),
        chunks_per_update=32,frames_per_chunk=81,policy_forward_population=2592,
        microbatching=False,gradient_accumulation=False,training_chunk_count=64,updates_per_epoch=None)
    result=dict(deepcopy(expected),result='PASS',initial_checkpoint_sha256='a'*64,
        initial_state_exact_saved_checkpoint=True,policy_reload_exact=True,optimizer_reload_exact=True,
        rng_restore_exact=True,training_batch_contract=batch,
        updates=[dict(update=i,bn_counter_delta=82*i,train_mse=.5/i,
            gradient_norm_before_clip=2.,update_seconds=5.) for i in (1,2)],
        batch_norm_counter_deltas={'vision.bn':164},optimizer_parameter_states=78,
        parameter_sha256_before={'std':'a','obj_pred_mlp.weight':'b','mlp.weight':'c'},
        parameter_sha256_after={'std':'a','obj_pred_mlp.weight':'b','mlp.weight':'d'},
        changed_parameter_names=['mlp.weight'],device=device,memory_policy=memory_admission(device),
        gpu_peak_reserved_bytes=90<<30,gpu_free_bytes_after=20<<30,
        memory_observations=[dict(phase=p,update=i,free_bytes=20<<30,peak_reserved_bytes=90<<30)
            for i in (1,2) for p in ('before_update','after_update')])
    return result,expected


def test_preserves_all_training_and_fresh_data_guards():
    result,expected=fixture();m.validate_training(result,expected)


@pytest.mark.parametrize('change',[
    lambda r:r.update(result='FAIL'), lambda r:r.update(optimizer_updates=1),
    lambda r:r.update(initial_checkpoint_sha256='b'*64),
    lambda r:r.update(initial_state_exact_saved_checkpoint=False),
    lambda r:r.update(optimizer_reload_exact=False), lambda r:r.update(rng_restore_exact=False),
    lambda r:r['training_batch_contract'].update(microbatching=True),
    lambda r:r['training_batch_contract'].update(chunks_per_update=16),
    lambda r:r['updates'][0].update(train_mse=float('nan')),
    lambda r:r['updates'][1].update(bn_counter_delta=82),
    lambda r:r['batch_norm_counter_deltas'].update({'vision.bn':163}),
    lambda r:r.update(changed_parameter_names=[]),
    lambda r:r['parameter_sha256_after'].update({'std':'changed'}),
    lambda r:r.update(gpu_free_bytes_after=(16<<30)-1),
    lambda r:r['memory_observations'].pop(),
    lambda r:r['memory_observations'][0].update(free_bytes=(16<<30)-1),
    lambda r:r['fresh_input_protocol'].update(retired_data_used=True),
])
def test_rejects_incomplete_or_relabelled_parent(change):
    result,expected=fixture();change(result)
    with pytest.raises(ValueError):m.validate_training(result,expected)


def test_parent_binding_rejects_alias_and_missing_bytes(tmp_path):
    parent=tmp_path/'parent.pt';parent.write_bytes(b'CPU fixture only')
    assert m.binding(parent)['path']==str(parent)
    alias=tmp_path/'alias.pt';alias.symlink_to(parent)
    with pytest.raises(ValueError):m.binding(alias)
    with pytest.raises(FileNotFoundError):m.binding(tmp_path/'missing.pt')
