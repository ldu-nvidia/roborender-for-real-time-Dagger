"""Pool-wrap/resume/prefetch comparisons to the pinned original offline cursor."""
from unittest import mock
import numpy as np
import pytest
from pipeline.bc_training import cyclic_sampling as current
from test.pipeline.test_bc_raven_offline_semantics import _original_offline_class

class Archive(dict):
    @property
    def files(self):
        return list(self)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass

@pytest.mark.parametrize('count,batch', [(104,32), (18,16), (64,32), (1,32), (7,16), (33,32)])
def test_multiple_reporting_epochs_match_original_cursor(count, batch):
    original = _original_offline_class()()
    original._offline_files = list(range(count))
    original._offline_cursor = 0
    sampler = current.CyclicBatchSampler(count, batch)
    accepted = 0
    with mock.patch.object(np, 'load', side_effect=lambda index: Archive(rgb_gen=np.asarray(index))):
        for _ in range(4):
            sampler.start_after(accepted)
            for indices in sampler:
                expected = [int(row['rgb_gen']) for row in original._next_offline_batch(batch)]
                assert indices == expected
                assert len(indices) == batch
                accepted += 1
                assert current.cursor_state(count,batch,accepted)['chunks_consumed'] == original._offline_cursor

def test_prefetch_does_not_commit_and_resume_does_not_reset():
    sampler = current.CyclicBatchSampler(104,32)
    prefetched = list(sampler)
    assert sampler.start == 0
    assert prefetched[-1] == list(range(96,104)) + list(range(24))
    saved = current.cursor_state(104,32,4)
    assert saved['next_index'] == 24
    resumed = current.CyclicBatchSampler(104,32)
    resumed.start_after(saved['accepted_updates'])
    assert next(iter(resumed)) == list(range(24,56))
    current.validate_cursor(saved,104,32,4)
    with pytest.raises(ValueError, match='cursor'):
        current.validate_cursor({**saved, 'next_index':0},104,32,4)

@pytest.mark.parametrize('count,batch', [(0,32), (True,32), (32,8), (18,False)])
def test_invalid_populations_rejected(count,batch):
    with pytest.raises(ValueError):
        current.CyclicBatchSampler(count,batch)

def test_reporting_epoch_not_claimed_exactly_once():
    from pathlib import Path
    from pipeline.bc_training.train_bc_h200 import training_sampling_contract
    root = Path('/pool')
    value = training_sampling_contract([root/f'{i:03d}.npz' for i in range(104)], root,
                                      shuffle=False,cyclic_batch_size=32)
    assert value['updates_per_reporting_epoch'] == 4
    assert 'epoch_boundary_cursor' not in value
    assert value['reporting_epoch'] == 'ceil_N_over_B_full_updates_cursor_never_reset'
