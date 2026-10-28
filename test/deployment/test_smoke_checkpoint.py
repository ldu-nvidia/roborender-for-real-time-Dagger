from unittest.mock import Mock

import errno
import hashlib
from pathlib import Path

import pytest

from pipeline.deployment.prepare_beta_campaign import stage_round_parent


def test_disposable_bc_checkpoint_preserves_handoff_not_epoch_claim(monkeypatch, tmp_path):
    from pipeline.deployment.h200 import qualify_bc_resume as q
    save = Mock()
    monkeypatch.setattr(q.bc, "save_checkpoint", save)
    parent = {"epoch": 41, "global_step": 4100, "bootstrap_sha256": "a"*64,
              "train_split_sha256": "b"*64, "validation_split_sha256": "c"*64}
    policy, optimizer, generator = object(), object(), object()
    q.save_smoke_checkpoint(tmp_path / "candidate.pt", policy, optimizer, parent,
                            {"spec": "test"}, generator, .2, [.2]*11)
    args = save.call_args.args
    assert args[3:5] == (41, 4101)
    metadata = args[7]
    assert metadata["policy_spec"] == {"spec": "test"}
    assert not metadata["production_checkpoint"] and not metadata["completed_epoch"]
    assert metadata["sampled_validation_only"]
    assert metadata["source_global_step"] == 4100
    assert metadata["bootstrap_sha256"] == "a"*64


def test_runtime_parent_staging_passes_real_registry_without_widening_root(tmp_path, monkeypatch):
    from test.pipeline.test_dagger_h200_services import running_service
    from pipeline.dagger.services.client import ServiceError
    source = tmp_path / "runtime" / "migration" / "candidate.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"small checkpoint boundary fixture, not a policy")
    allowed = tmp_path / "checkpoints"
    allowed.mkdir()
    monkeypatch.setenv("DAGGER_CHECKPOINT_SOURCE_ROOT", str(allowed))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {"path": str(source), "sha256": digest, "metadata": {"purpose": "test"}}
    with running_service("registry", tmp_path / "ledger") as registry:
        with pytest.raises(ServiceError, match="checkpoint path must be below"):
            registry.post("/v1/checkpoints/register-local", payload)
        admission = stage_round_parent(source, allowed)
        assert admission == stage_round_parent(source, allowed)
        assert admission["sha256"] == digest and admission["production_promoted"] is False
        assert Path(admission["path"]).is_relative_to(allowed)
        registry.post("/v1/checkpoints/register-local", {**payload, "path": admission["path"]})
        assert registry.get("/v1/checkpoint-records/"+digest)["checkpoint"]["sha256"] == digest
        assert source.read_bytes() == Path(admission["path"]).read_bytes()


@pytest.mark.parametrize("unsafe", ["directory_symlink", "file_symlink", "corrupt_target"])
def test_parent_staging_rejects_unsafe_existing_target(tmp_path, unsafe):
    source = tmp_path / "candidate.pt"
    source.write_bytes(b"candidate")
    allowed = tmp_path / "checkpoints"
    allowed.mkdir()
    directory = allowed / "campaign_parents"
    if unsafe == "directory_symlink":
        directory.symlink_to(tmp_path, target_is_directory=True)
    else:
        directory.mkdir()
        target = directory / (hashlib.sha256(source.read_bytes()).hexdigest()+".pt")
        if unsafe == "file_symlink":
            target.symlink_to(source)
        else:
            target.write_bytes(b"different")
    with pytest.raises(ValueError):
        stage_round_parent(source, allowed)
    assert source.read_bytes() == b"candidate"


def test_parent_staging_cross_filesystem_copy(tmp_path, monkeypatch):
    from pipeline.deployment import prepare_beta_campaign as module
    source = tmp_path / "candidate.pt"
    source.write_bytes(b"candidate")
    allowed = tmp_path / "checkpoints"
    allowed.mkdir()
    original_link = module.os.link
    def cross_device(src, dst):
        if Path(src) == source:
            raise OSError(errno.EXDEV, "test different source filesystem")
        return original_link(src, dst)
    monkeypatch.setattr(module.os, "link", cross_device)
    admission = stage_round_parent(source, allowed)
    assert Path(admission["path"]).read_bytes() == source.read_bytes()
    assert list((allowed / "campaign_parents").glob("*.part")) == []
