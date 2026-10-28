"""Metadata-only mixed-limit admission; synthetic receipts are not GPU proof."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.bc_training import assemble_production_protocol as assembler
from pipeline.bc_training import data_admission
from pipeline.bc_training import production_protocol as strict
from pipeline.bc_training.handoff_common import binding, bound, canonical, digest
from test.pipeline.test_bc_production_handoff import complete  # shared synthetic fixture


@contextmanager
def mixed_evidence(complete, tmp_path):
    """Rebind real synthetic transfer chains, restoring the shared fixture after use."""
    originals = {}
    def replace(path, doc):
        path = Path(path)
        originals.setdefault(path, path.read_bytes())
        path.write_bytes(canonical(doc))
        return binding(path)
    args = deepcopy(complete.args)
    capacity = bound(args['capacity_receipts'][0])
    capacity['cgroup_limit_bytes'] = 43 << 30
    extra_path = tmp_path / 'qualified-b6-43g.json'
    extra_path.write_bytes(canonical(capacity))
    extra = binding(extra_path)
    args['capacity_receipts'].append(extra)
    pool = Path(args['source_pool']['path']).parent
    chains = sorted((pool / 'manifests/cohorts').glob('*/execution-transfer-receipt.json'))
    sessions = {}
    try:
        for path in chains:
            transfer = json.loads(path.read_bytes())
            sid = transfer['session_id']
            files = transfer['files']
            if sid not in sessions:
                launch, host, session = (bound({k: files[name][k] for k in ('path', 'raw_sha256')})
                    for name in ('launch.json', 'host-result.json', 'session-execution.json'))
                if not sessions:  # One actual B6 session used the larger qualified limit.
                    launch['execution_topology']['cgroup_limit_bytes_per_collector'] = 43 << 30
                    launch['capacity_binding'] = {**extra, 'path': '/leo/qualified-b6-43g.json'}
                topology = launch['execution_topology']
                host.update(sampled_cgroup_peak_bytes=24 << 30, minimum_host_available_kib=(20 << 30) // 1024,
                    last_live_sample={'oom_killed': False, 'memory_swap.current': 0,
                        'events': dict(max=0, oom=0, oom_kill=0, oom_group_kill=0)})
                new_launch = replace(files['launch.json']['path'], launch)
                new_host = replace(files['host-result.json']['path'], host)
                session.update(launch_raw_sha256=new_launch['raw_sha256'], host_result_raw_sha256=new_host['raw_sha256'])
                new_session = replace(files['session-execution.json']['path'], session)
                sessions[sid] = ({'launch.json': new_launch, 'host-result.json': new_host,
                                  'session-execution.json': new_session}, digest(canonical(session)))
            updated, session_sha = sessions[sid]
            receipt = bound({k: files['execution-receipt.json'][k] for k in ('path', 'raw_sha256')})
            receipt['session_execution_canonical_sha256'] = session_sha
            new_receipt = replace(files['execution-receipt.json']['path'], receipt)
            for name, new_binding in {**updated, 'execution-receipt.json': new_receipt}.items():
                files[name] = {**new_binding, 'bytes': Path(new_binding['path']).stat().st_size}
            replace(path, transfer)
        yield args
    finally:
        for path, raw in originals.items():
            path.write_bytes(raw)


def audit(complete, protocol, tmp_path):
    path = tmp_path / 'candidate.json'
    path.write_bytes(canonical(protocol))
    return strict.audit_production_protocol(path, digest(path.read_bytes()), complete.manifest,
        complete.rb['path'], complete.rb['raw_sha256'])


def test_mixed_actual_profiles_assemble_and_independently_admit(complete, tmp_path, monkeypatch):
    real_open = Path.open
    def metadata_only(path, *args, **kwargs):
        assert path.suffix != '.npz', 'admission must not deserialize dataset tensors'
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    with mixed_evidence(complete, tmp_path) as args:
        result = assembler.assemble(**args, output=tmp_path / 'protocol.json')
        protocol = bound(result['protocol'])
        assert protocol['schema'] == strict.SCHEMA_V4
        assert 'cgroup_limit_bytes_per_collector' not in protocol['execution_topology']
        assert {p['cgroup_limit_bytes_per_collector'] for p in protocol['capacity_profiles'].values()} == {32 << 30, 43 << 30}
        assert len(protocol['session_resource_bindings']) == 28
        report = audit(complete, protocol, tmp_path)
        assert report['exact_role_quotas'] == {'train': 2560, 'validation': 320}
        assert report['payloads_opened'] is False
        admitted = data_admission.audit_split_files(complete.manifest, complete.splits,
            require_independent_render=True, purpose='production', run_name='prod-synthetic-mixed',
            protocol_path=result['protocol']['path'], protocol_sha256=result['protocol']['raw_sha256'],
            render_receipt_path=complete.rb['path'], render_receipt_sha256=complete.rb['raw_sha256'])
        assert admitted['production_admitted'] is True
        assert admitted['artifact_roles']['offline_test_chunks'] == 0


@pytest.mark.parametrize('change', [
    'relabel_limit', 'wrong_host', 'wrong_concurrency', 'lower_reserve', 'unbound_profile',
    'missing_session', 'duplicate_session', 'wrong_session_binding', 'wrong_launch_hash',
    'wrong_capacity', 'unsafe_host_peak', 'swap', 'oom', 'low_host_reserve', 'wrong_width',
    'downgrade_schema',
])
def test_mixed_profile_tampering_rejected(complete, tmp_path, change):
    with mixed_evidence(complete, tmp_path) as args:
        result = assembler.assemble(**args, output=tmp_path / 'protocol.json')
        protocol = bound(result['protocol'])
        row = protocol['session_resource_bindings'][0]
        profile = protocol['capacity_profiles'][row['capacity_receipt_raw_sha256']]
        if change == 'relabel_limit': profile['cgroup_limit_bytes_per_collector'] = 40 << 30
        elif change == 'wrong_host': profile['host_id'] = 'another-host'
        elif change == 'wrong_concurrency': profile['concurrent_collectors'] = 2
        elif change == 'lower_reserve': profile['minimum_host_reserve_bytes'] //= 2
        elif change == 'unbound_profile': protocol['capacity_profiles']['0' * 64] = profile
        elif change == 'missing_session': protocol['session_resource_bindings'].pop()
        elif change == 'duplicate_session': protocol['session_resource_bindings'][-1] = deepcopy(row)
        elif change == 'wrong_session_binding': row['session_execution'] = protocol['session_execution_receipts'][1]
        elif change == 'wrong_launch_hash': row['launch']['raw_sha256'] = '0' * 64
        elif change == 'wrong_capacity': row['capacity_receipt_raw_sha256'] = protocol['capacity_receipts'][0]['raw_sha256']
        elif change == 'downgrade_schema': protocol['schema'] = strict.SCHEMA_V3
        elif change == 'wrong_width':
            # A coherent capacity/profile hash change still cannot qualify B6 with B1.
            old = row['capacity_receipt_raw_sha256']
            receipt = bound(next(b for b in protocol['capacity_receipts'] if b['raw_sha256'] == old))
            receipt['num_envs'] = 1
            for sample in receipt['per_flush_memory']:
                sample['emitted_chunks_per_lane'] = sample['emitted_chunks_per_lane'][:1]
            path = tmp_path / 'wrong-width-capacity.json'; path.write_bytes(canonical(receipt)); new = binding(path)
            protocol['capacity_receipts'] = [new if b['raw_sha256'] == old else b for b in protocol['capacity_receipts']]
            protocol['capacity_profiles'][new['raw_sha256']] = protocol['capacity_profiles'].pop(old)
            row['capacity_receipt_raw_sha256'] = new['raw_sha256']
            launch = bound(row['launch']); launch['capacity_binding']['raw_sha256'] = new['raw_sha256']
            _rebind_session(protocol, row, tmp_path, launch=launch)
        else:
            host = bound(row['host_result'])
            if change == 'unsafe_host_peak': host['sampled_cgroup_peak_bytes'] = 44 << 30
            elif change == 'swap': host['last_live_sample']['memory_swap.current'] = 1
            elif change == 'oom': host['last_live_sample']['events']['oom'] = 1
            else: host['minimum_host_available_kib'] = 1
            _rebind_session(protocol, row, tmp_path, host=host)
        with pytest.raises((ValueError, KeyError)):
            audit(complete, protocol, tmp_path)


def _rebind_session(protocol, row, tmp_path, *, launch=None, host=None):
    """Update all metadata hashes, so negative tests exercise semantics, not stale hashes."""
    old_binding = row['session_execution']
    session = bound(old_binding)
    old_canonical = digest(canonical(session))
    for value, key, field in ((launch, 'launch', 'launch_raw_sha256'), (host, 'host_result', 'host_result_raw_sha256')):
        if value is not None:
            path = tmp_path / (key + '.json'); path.write_bytes(canonical(value)); row[key] = binding(path)
            session[field] = row[key]['raw_sha256']
    path = tmp_path / 'session.json'; path.write_bytes(canonical(session)); row['session_execution'] = binding(path)
    protocol['session_execution_receipts'] = [row['session_execution'] if b == old_binding else b
                                             for b in protocol['session_execution_receipts']]
    for index, b in enumerate(protocol['cohort_execution_receipts']):
        receipt = bound(b)
        if receipt['session_execution_canonical_sha256'] == old_canonical:
            receipt['session_execution_canonical_sha256'] = digest(canonical(session))
            path = tmp_path / f'attestation-{index}.json'; path.write_bytes(canonical(receipt))
            protocol['cohort_execution_receipts'][index] = binding(path)
