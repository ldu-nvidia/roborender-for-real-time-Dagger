import os
import stat

import pytest
import torch

from pipeline.dagger.services import raven_learner_worker as worker


@pytest.mark.parametrize('writer', [worker.atomic_torch_save, worker.atomic_json])
def test_checkpoint_and_receipt_fsync_before_and_after_atomic_publish(tmp_path, monkeypatch, writer):
    events=[];original_fsync=os.fsync;original_replace=os.replace
    def fsync(fd):
        events.append('directory_fsync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file_fsync')
        return original_fsync(fd)
    def replace(source,target):
        events.append('replace')
        return original_replace(source,target)
    monkeypatch.setattr(os,'fsync',fsync);monkeypatch.setattr(os,'replace',replace)
    target=tmp_path/'checkpoint.pt'
    writer({'step':100}, target)
    assert events==['file_fsync','replace','directory_fsync']
    if writer is worker.atomic_torch_save:
        assert torch.load(target, weights_only=True)=={'step':100}


@pytest.mark.parametrize('writer', [worker.atomic_torch_save, worker.atomic_json])
def test_failed_file_fsync_does_not_publish_checkpoint_or_receipt(tmp_path, monkeypatch, writer):
    def failure(fd):
        raise OSError('fixture durable storage failure')
    monkeypatch.setattr(os,'fsync',failure)
    target=tmp_path/'final'
    with pytest.raises(OSError, match='durable storage'):
        writer({'step':100}, target)
    assert not target.exists()
