"""Synthetic small receipt handoff; no GPU, transport or scientific data claim."""
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.legacy.bc_collection.transfer_native_cohort import (
    transfer_cohort, transfer_execution_evidence, _publish_exact_metadata,
)
from pipeline.deployment.rtx_actor.bc_execution_evidence import build_session_receipts, canonical
from test.pipeline.test_bc_native_transfer import fixture as payload_fixture, RUNTIME


def fixture(tmp_path):
    doc, transport = payload_fixture()
    doc['request']['motions'] = [{'motion_id': 'synthetic'}]
    doc.update(vector_steps=81, native_payload_writer_unchanged=True)
    raw = canonical(doc) + b'\n'
    transfer_cohort(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    session = dict(session_id='bc-session-leo-fixture', requests=[dict(sha256=doc['request_sha256'])])
    execution = dict(runtime_contract_sha256='c' * 64)
    launch = dict(collector_execution=execution, session_canonical_sha256=digest(canonical(session)),
        request_identities={doc['request_sha256']: digest(canonical(doc['request']))},
        source_manifest_raw_sha256='1' * 64, port_manifest_raw_sha256='2' * 64, session_sha256='3' * 64)
    native = dict(status='complete', failure=None, native_initializations=1, scene_initializations=1,
        session_id=session['session_id'], cohorts=[dict(cohort_id=doc['request']['cohort_id'],
            request_sha256=doc['request_sha256'], output_dir=doc['request']['output_dir'], vector_steps=81)])
    host = dict(passed=True, failure=None, exit_code=0)
    blobs = {name: canonical(value) + b'\n' for name, value in
             [('launch.json', launch), ('host-result.json', host), ('bc-session-result.json', native)]}
    sr, cr = build_session_receipts(launch=launch, launch_raw_sha256=digest(blobs['launch.json']), session=session,
        result=native, result_raw_sha256=digest(blobs['bc-session-result.json']), host_result=host,
        host_result_raw_sha256=digest(blobs['host-result.json']),
        cohort_envelopes=[dict(document=doc, raw_sha256=digest(raw))])
    blobs['session-execution.json'], blobs['execution-receipt.json'] = canonical(sr), canonical(cr[0])
    class MetadataTransport:
        def __init__(self): self.reads = []
        def read(self, path, limit):
            assert limit == 8 << 20
            self.reads.append(path)
            return blobs[Path(path).name]
        def copy(self, *args): raise AssertionError('no payload recopy allowed')
    return raw, blobs, MetadataTransport()


def test_handoff_is_exact_metadata_only_atomic_and_idempotent(tmp_path, monkeypatch):
    raw, blobs, transport = fixture(tmp_path)
    original_read = Path.read_bytes
    def metadata_only(path):
        assert path.suffix != '.npz'
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', metadata_only)
    first = transfer_execution_evidence(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    second = transfer_execution_evidence(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert first == second and first['payloads_rehashed'] is False
    assert set(first['files']) == set(blobs) and len(transport.reads) == 10
    for name, binding in first['files'].items():
        assert Path(binding['path']).read_bytes() == blobs[name]
        assert hashlib.sha256(blobs[name]).hexdigest() == binding['raw_sha256']
    assert not list(tmp_path.rglob('.execution-*'))


@pytest.mark.parametrize('fault', ['host', 'native', 'launch', 'cohort', 'session', 'oversized'])
def test_changed_or_missing_completed_binding_rejected(tmp_path, fault):
    raw, blobs, transport = fixture(tmp_path)
    if fault == 'oversized':
        blobs['session-execution.json'] = b' ' * ((8 << 20) + 1)
    else:
        name = {'host': 'host-result.json', 'native': 'bc-session-result.json', 'launch': 'launch.json',
                'cohort': 'execution-receipt.json', 'session': 'session-execution.json'}[fault]
        value = json.loads(blobs[name])
        if fault == 'cohort': value['actual_num_envs'] = 28
        elif fault == 'session': value['request_identities'] = {}
        else: value['changed'] = True
        blobs[name] = canonical(value)
    with pytest.raises(ValueError):
        transfer_execution_evidence(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert not list(tmp_path.rglob('execution-transfer-receipt.json'))


def test_existing_conflicting_metadata_is_not_replaced(tmp_path):
    raw, blobs, transport = fixture(tmp_path)
    conflict = tmp_path / 'manifests/sessions/bc-session-leo-fixture/launch.json'
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b'user metadata')
    with pytest.raises(ValueError, match='existing execution metadata differs'):
        transfer_execution_evidence(raw, remote_runtime=RUNTIME, dataset_root=tmp_path, transport=transport)
    assert conflict.read_bytes() == b'user metadata'


def test_publication_rejects_symlink(tmp_path):
    target = tmp_path / 'original.json'
    target.write_bytes(b'original')
    alias = tmp_path / 'alias.json'
    alias.symlink_to(target)
    with pytest.raises(ValueError): _publish_exact_metadata(alias, b'new')
    assert target.read_bytes() == b'original'


def test_cli_exposes_explicit_opt_in():
    import inspect
    from pipeline.legacy.bc_collection import transfer_native_cohort
    assert '--include-execution-evidence' in inspect.getsource(transfer_native_cohort.main)
