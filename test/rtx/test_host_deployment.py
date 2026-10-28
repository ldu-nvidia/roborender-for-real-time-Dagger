import hashlib
import json
import pytest
from dagger_rtx.host_deployment import LEGACY, load_host_deployment


def fixture(tmp_path, **changes):
    value = dict(LEGACY, hostname="465d6b0-lcedt", physical_gpu_uuid="GPU-0c1949ae-379a-2f2e-0f23-621de2eee409")
    value.update(changes)
    path = tmp_path / "beta.json"
    path.write_text(json.dumps(value))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_beta_bindings_preserve_gpu1_and_source(tmp_path):
    p, h = fixture(tmp_path)
    value = load_host_deployment(path=p, digest=h, hostname="465d6b0-lcedt")
    assert value["physical_gpu_index"] == 1
    assert value["raven_root"] == LEGACY["raven_root"]


@pytest.mark.parametrize("changes", [
    {"physical_gpu_index": 0}, {"physical_gpu_index": True},
    {"image_id": "latest"}, {"physical_gpu_uuid": "1"},
    {"anchor_root": "/tmp/../other"}, {"anchor_root": "/tmp/path,readonly"},
    {"raven_root": "/tmp/wrong-source"}, {"teacher_beta": 0.5},
])
def test_reject_unsafe_or_scientific_override(tmp_path, changes):
    p, h = fixture(tmp_path, **changes)
    with pytest.raises(ValueError):
        load_host_deployment(path=p, digest=h, hostname="465d6b0-lcedt")


def test_hash_and_host_required(tmp_path):
    p, h = fixture(tmp_path)
    for digest, host in [(None, "465d6b0-lcedt"), ("0" * 64, "465d6b0-lcedt"), (h, "wrong-host")]:
        with pytest.raises(ValueError):
            load_host_deployment(path=p, digest=digest, hostname=host)


def test_legacy_without_override(monkeypatch):
    monkeypatch.delenv("DAGGER_HOST_DEPLOYMENT_FILE", raising=False)
    monkeypatch.delenv("DAGGER_HOST_DEPLOYMENT_SHA256", raising=False)
    assert load_host_deployment() == LEGACY
