import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.legacy.bc_collection.transfer_native_cohort import source_path, transfer_cohort


RUNTIME = "/home/ldu/roborender/runtime/bc-cohorts/qualification-123"


def fixture():
    payload = b"test bytes, not scientific training data"
    row = {"source_path": "/dagger-runtime/bc_isaac/chunk_run_0000000.npz",
        "source_sha256": hashlib.sha256(payload).hexdigest(), "source_bytes": len(payload),
        "request_sha256": "a" * 64, "cohort_id": "train-000", "selected_for_split": True,
        "overcollection": False}
    doc = {"schema": "roborender.bc-native-cohort.v1", "status": "complete", "teacher_only": True,
        "optimizer_steps": 0, "production_admitted": False, "request_sha256": "a" * 64,
        "request": {"cohort_id": "train-000", "output_dir": "/dagger-runtime/bc_isaac", "chunks_per_env": [1]},
        "selected_chunks": 1, "chunks": [row]}
    class Transport:
        copies = 0
        corrupt = False
        provenance = {**row, "rows": [{} for _ in range(81)]}

        def read(self, path):
            return json.dumps(self.provenance).encode()

        def copy(self, source, target):
            self.copies += 1
            Path(target).write_bytes(b"bad" if self.corrupt else payload)
    return doc, Transport()


def test_transfer_and_receipt_retry_dont_recopy(tmp_path):
    doc, transport = fixture()
    raw = json.dumps(doc).encode()
    result = transfer_cohort(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    retry = transfer_cohort(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert result == retry and transport.copies == 1
    assert result["source_payloads_rehashed"] is False
    assert Path(result["files"][0]["provenance_path"]).is_file()
    assert not list(tmp_path.rglob("*.part-*"))


@pytest.mark.parametrize("path,runtime", [
    ("/etc/passwd", RUNTIME), ("/dagger-runtime/../secret", RUNTIME),
    ("/dagger-runtime/bc_isaac/x", "/home/ldu"),
    ("/dagger-runtime/bc_isaac/x", "/home/ldu/roborender/runtime/bc-cohorts/../secret"),
])
def test_reject_out_of_scope_paths(path, runtime):
    with pytest.raises(ValueError):
        source_path(path, runtime)


def test_incomplete_cohort_not_copied(tmp_path):
    doc, transport = fixture()
    doc["status"] = "failed"
    with pytest.raises(ValueError):
        transfer_cohort(json.dumps(doc).encode(), remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert transport.copies == 0


def test_corrupt_transfer_not_published(tmp_path):
    doc, transport = fixture()
    transport.corrupt = True
    with pytest.raises(ValueError, match="transport size/hash"):
        transfer_cohort(json.dumps(doc).encode(), remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert not list(tmp_path.rglob("transfer-receipt.json"))
    assert not list(tmp_path.rglob("*.npz"))
    assert len(list(tmp_path.rglob("*.part-*"))) == 1


def test_sidecar_mismatch_fails_before_payload_transfer(tmp_path):
    doc, transport = fixture()
    transport.provenance = {**transport.provenance, "source_sha256": "b" * 64}
    with pytest.raises(ValueError, match="per-step provenance"):
        transfer_cohort(json.dumps(doc).encode(), remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert transport.copies == 0


def test_existing_changed_destination_is_never_overwritten(tmp_path):
    doc, transport = fixture()
    destination = tmp_path / "isaac/chunk_run_0000000.npz"
    destination.parent.mkdir()
    destination.write_bytes(b"user bytes")
    with pytest.raises(ValueError, match="refusing to replace"):
        transfer_cohort(json.dumps(doc).encode(), remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert destination.read_bytes() == b"user bytes"
    assert transport.copies == 0


def test_extra_output_excluded(tmp_path):
    doc, transport = fixture()
    extra = copy.deepcopy(doc["chunks"][0])
    extra.update(selected_for_split=False, overcollection=True, source_path="/dagger-runtime/bc_isaac/chunk_excess.npz")
    doc["chunks"].append(extra)
    result = transfer_cohort(json.dumps(doc).encode(), remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert len(result["files"]) == 1 and transport.copies == 1


@pytest.mark.parametrize('timeout', [False, True])
def test_interrupted_copy_same_process_retries_without_reusing_partial(tmp_path, timeout):
    import subprocess
    doc, transport = fixture()
    original = transport.copy
    attempts = []
    def interrupted(source, target):
        attempts.append(Path(target))
        if len(attempts) == 1:
            Path(target).write_bytes(b'incomplete')
            if timeout:
                raise subprocess.TimeoutExpired('scp', 1800)
            raise subprocess.CalledProcessError(255, 'scp')
        original(source, target)
    transport.copy = interrupted
    raw = json.dumps(doc).encode()
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        transfer_cohort(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert not list((tmp_path/'isaac').glob('*.npz'))
    result = transfer_cohort(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert result['verified'] is True
    assert attempts[0] != attempts[1]
    assert attempts[0].read_bytes() == b'incomplete'
    assert not attempts[1].exists()
    assert Path(result['files'][0]['destination_path']).stat().st_size == doc['chunks'][0]['source_bytes']


def test_old_pid_partial_is_preserved_not_adopted(tmp_path):
    import os
    doc, transport = fixture()
    directory = tmp_path/'isaac'; directory.mkdir()
    old = directory/f'chunk_run_0000000.npz.part-{os.getpid()}'
    old.write_bytes(b'old incomplete copy')
    result = transfer_cohort(json.dumps(doc).encode(), remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert result['verified'] is True and transport.copies == 1
    assert old.read_bytes() == b'old incomplete copy'
