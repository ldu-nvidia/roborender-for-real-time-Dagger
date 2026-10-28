"""Control-only compatibility tests, not native runtime qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline.bc_training.handoff_common import publish
from pipeline.deployment.production_pool_control import (
    CONTROL_PATHS, REMOTE_VERIFY_AND_EXEC, SCHEMA, validate_control,
)
from pipeline.deployment.source_snapshot import canonical, file_record


def manifest(root, names):
    rows = [file_record(root, name) for name in sorted(names)]
    return dict(schema='roborender.source-snapshot.v1', files=rows, file_count=len(rows),
                source_sha256=hashlib.sha256(canonical(rows)).hexdigest())


def control_fixture(tmp_path):
    native = tmp_path / 'native'; overlay = tmp_path / 'overlay'
    helper = 'pipeline/deployment/rtx_actor/production_resident_pool.py'
    science = 'pipeline/dagger/services/raven_learner_worker.py'
    for root in (native, overlay):
        for name, raw in ((helper, b'print("original")\n'), (science, b'# immutable learner\n')):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    (overlay / helper).write_text('import json,sys;print(json.dumps({"args":sys.argv[1:]}))\n')
    for name in CONTROL_PATHS - {helper}:
        (overlay / name).write_text('# metadata reader only\n')
    old = manifest(native, (helper, science))
    new = manifest(overlay, CONTROL_PATHS | {science})
    before = {r['path']: r for r in old['files']}
    after = {r['path']: r for r in new['files']}
    oldref = publish(tmp_path / 'native.json', old)
    newref = publish(tmp_path / 'overlay.json', new)
    value = dict(schema=SCHEMA, native_source_sha256=old['source_sha256'], source_root=str(overlay),
        source_manifest=newref, native_manifest=oldref,
        changed_files=[dict(path=p, old_sha256=before.get(p, {}).get('sha256'), new_sha256=after[p]['sha256'])
                       for p in sorted(CONTROL_PATHS)])
    return value, science


def test_remote_reader_verifies_complete_unchanged_scientific_tree(tmp_path):
    control, science = control_fixture(tmp_path)
    argv = [sys.executable, '-c', REMOTE_VERIFY_AND_EXEC, canonical(control).decode(), '--action', 'status']
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'args': ['--action', 'status']}
    (Path(control['source_root']) / science).write_text('# forbidden learner change\n')
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode != 0 and not result.stdout


def test_remote_reader_rejects_changed_manifest_or_non_reader_delta(tmp_path):
    control, _ = control_fixture(tmp_path)
    for change in ('manifest', 'sha', 'native', 'removed'):
        bad = deepcopy(control)
        if change == 'manifest': bad['source_manifest']['raw_sha256'] = 'f' * 64
        if change == 'sha': bad['changed_files'][0]['new_sha256'] = 'e' * 64
        if change == 'native': bad['native_source_sha256'] = 'd' * 64
        if change == 'removed': bad['changed_files'].pop()
        result = subprocess.run([sys.executable, '-c', REMOTE_VERIFY_AND_EXEC, canonical(bad).decode()],
                                capture_output=True)
        assert result.returncode != 0 and not result.stdout


def test_control_schema_has_no_generic_source_exemption(tmp_path):
    control, _ = control_fixture(tmp_path)
    control['source_root'] = '/home/ldu/roborender/runtime/source-releases/reader-recovery/frozen'
    assert validate_control(control, control['native_source_sha256']) == control
    for change in ('path', 'extra', 'learner', 'native'):
        bad = deepcopy(control)
        if change == 'path': bad['source_root'] = '/home/ldu/roborender'
        if change == 'extra': bad['skip_source_check'] = True
        if change == 'learner': bad['changed_files'][0]['path'] = 'pipeline/dagger/services/raven_learner_worker.py'
        if change == 'native': bad['native_source_sha256'] = 'f' * 64
        with pytest.raises(ValueError): validate_control(bad, control['native_source_sha256'])


def test_v2_queue_forwards_exact_recovery_bindings_only(tmp_path):
    from pipeline.dagger.coverage_queue import operations_for
    control, _ = control_fixture(tmp_path)
    control['source_root'] = '/home/ldu/roborender/runtime/source-releases/reader-recovery/frozen'
    management = publish(tmp_path / 'management.json', {'native_source_sha256': control['native_source_sha256']})
    ref = publish(tmp_path / 'pool-control.json', control)
    site = publish(tmp_path / 'site.json', dict(schema='roborender.leo-site.v1', cpu_python='/usr/bin/python3'))
    retained = publish(tmp_path / 'retained.json', dict(schema='roborender.full295-retained-h200.v1'))
    runtime = dict(schema='roborender.full295-runtime-bindings.v2', site=site, retained=retained,
        resident_pool=dict(path='/home/ldu/recovery/pool.json', sha256='a' * 64),
        source_sha256=control['native_source_sha256'],
        control_only=dict(management_release=management, pool_control=ref))
    ops = operations_for(runtime, tmp_path)
    for row in ops['commands'].values():
        args = row['argv']
        assert args[args.index('--pool-control') + 1] == ref['path']
        assert args[args.index('--pool-control-sha256') + 1] == ref['raw_sha256']
        assert args[args.index('--management-release') + 1] == management['path']
    with pytest.raises(ValueError): operations_for({**runtime, 'ignore_source': True}, tmp_path)
    bad = deepcopy(runtime); bad['control_only']['skip_verification'] = True
    with pytest.raises(ValueError): operations_for(bad, tmp_path)


def test_retained_pool_calls_use_verified_overlay_without_native_replacement():
    from pipeline.deployment.retained_campaign_site import RetainedResidentSite
    site = object.__new__(RetainedResidentSite)
    site.config={'schema':'roborender.cross_gpu_dagger_mse_full295.leo.v1'}
    site.pool = {'schema': 'roborender.leo-production-resident-pool.v1'}
    site.site = {'rtx_python': '/usr/bin/python3'}
    site.pool_file = Path('/home/ldu/recovery/pool.json')
    site.pool_control = {'reviewed': 'descriptor'}
    calls = []
    site.json = lambda argv, host, **kwargs: calls.append((argv, host, kwargs)) or {'ok': True}
    assert site.pool_action('status') == {'ok': True}
    args, host, _ = calls[0]
    assert host == 'beta' and args[1:3] == ['-c', REMOTE_VERIFY_AND_EXEC]
    assert json.loads(args[3]) == site.pool_control
    assert args[-4:] == ['--pool', site.pool_file, '--action', 'status']
    with pytest.raises(ValueError): site.pool_action('abort')
