"""Canonical path plumbing is not scientific or live-capacity certification."""
import importlib
import json
from pathlib import Path

import pytest

from pipeline.deployment.rtx_actor import bc_runtime
from pipeline.bc_collection import run_continuous

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path('/data1/workspaces/roborender')


def test_paths_are_canonical_and_registry_bound():
    paths = bc_runtime.load_paths(ROOT)
    assert paths.mounts(0) == dict(
        raven=PROJECT / 'simulation_assets/rtx_BC_transfer',
        anchors=PROJECT / 'simulation_assets/anchor_pool_2048',
        cache=PROJECT / 'caches/isaac/leo-gpu0')
    assert paths.mounts(1)['cache'] == PROJECT / 'caches/isaac/leo-gpu1'
    assert paths.get('bc_collection_runs') == PROJECT / 'runs/bc-continuous'
    with pytest.raises(ValueError):
        paths.mounts(2)


@pytest.mark.parametrize('value', [
    '/home/ldu/roborender/runtime/bc-continuous/test',
    '/data1/workspaces/roborender/runs/legacy_home/test',
    '/data1/workspaces/roborender/runs/bc-continuous',
    '/data1/workspaces/roborender/runs/bc-continuous/nested/test',
    '/data1/workspaces/roborender/runs/bc-continuous/../test',
])
def test_new_run_rejects_old_aliases_and_non_scoped_paths(value):
    with pytest.raises(ValueError):
        bc_runtime.load_paths(ROOT).validate_run(value)


@pytest.mark.parametrize('key', bc_runtime.PATH_KEYS)
def test_registry_cannot_widen_execution_jail(tmp_path, key):
    doc = json.loads((ROOT / 'configs/storage_paths.json').read_text())
    host = doc['hosts']['leo']
    if key in host['roots']:
        host['roots'][key] = '/tmp/elsewhere'
    else:
        host['artifacts'].pop(key)
        host['roots'][key] = '/tmp/elsewhere'
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/storage_paths.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        bc_runtime.load_paths(tmp_path)


def test_paths_snapshot_is_immutable(tmp_path):
    paths = bc_runtime.load_paths(ROOT)
    first = paths.freeze(tmp_path)
    assert paths.freeze(tmp_path) == first
    assert first['registry_raw_sha256'] == paths.registry.raw_sha256
    (tmp_path / 'storage-paths.json').chmod(0o600)
    (tmp_path / 'storage-paths.json').write_text('{}')
    with pytest.raises(FileExistsError):
        paths.freeze(tmp_path)


def test_new_dependencies_are_in_source_admission():
    assert {'configs/storage_paths.json', 'roborender/infrastructure/paths.py',
            'pipeline/deployment/rtx_actor/bc_runtime.py'} <= run_continuous.REQUIRED_SOURCE
    assert 'pipeline/deployment/rtx_actor/run_bc_cohort.py' not in run_continuous.REQUIRED_SOURCE


@pytest.mark.parametrize('gpu', [0, 1])
def test_actual_native_argv_differs_only_in_host_storage_mounts(tmp_path, monkeypatch, gpu):
    actor = ROOT / 'pipeline/deployment/rtx_actor'
    monkeypatch.syspath_prepend(str(actor))
    native = importlib.import_module('qualify_native_isaac6')
    binding = importlib.import_module('leo_binding')
    monkeypatch.setattr(binding, 'checked_gpu', lambda *a, **k: 'GPU-unit-test')
    (tmp_path / 'port-manifest.json').write_text(json.dumps(dict(files=[])))
    paths = bc_runtime.load_paths(ROOT)
    options = dict(name='bc-cohort-leo-layout', output=tmp_path / 'run', overlay=tmp_path,
                   code_root=ROOT / 'pipeline/dagger/rtx/code', num_envs=32,
                   physical_gpu=gpu, mode='bc-collect', deployment_host='leo-rtx-6k')
    before = native.command(**options)
    after = native.command(**options, storage_paths=paths.mounts(gpu))
    replacements = {
        str(native.RAVEN): str(paths.mounts(gpu)['raven']),
        str(native.ANCHORS): str(paths.mounts(gpu)['anchors']),
        str(native.REPO / f'runtime/isaac6-cache-leo-gpu{gpu}'): str(paths.mounts(gpu)['cache']),
    }
    expected = []
    for arg in before:
        for old, new in replacements.items():
            arg = arg.replace('src=' + old + ',', 'src=' + new + ',')
        expected.append(arg)
    assert after == expected
    assert not any('/home/ldu/' in arg or '/data2/' in arg for arg in after)
    assert after[after.index('--memory')] == before[before.index('--memory')]


def test_collection_algorithms_do_not_import_runtime_plumbing():
    import ast
    for name in ('collector.py', 'chunk_writer.py'):
        tree = ast.parse((ROOT / 'pipeline/bc_collection' / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'deployment' not in (node.module or '')
