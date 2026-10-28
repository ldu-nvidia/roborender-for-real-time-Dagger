"""CPU-only source location and escape checks; no native/workload actions."""
from test.deployment.source_paths import REPO_ROOT
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest

from roborender.infrastructure import source_locations as locations


@pytest.mark.parametrize('root', [locations.LEGACY_RELEASE_ROOT, locations.CANONICAL_RELEASE_ROOT])
@pytest.mark.parametrize('leaf', ['code', 'frozen', 'code/pipeline/deployment/rtx_actor'])
def test_exact_release_layout(root, leaf):
    assert locations.is_source_release_location(root / 'release-1' / leaf)


@pytest.mark.parametrize('path', [
    '/data1/workspaces/roborender/source',
    '/data1/workspaces/roborender/code_releases',
    '/data1/workspaces/roborender/code_releases/release-1',
    '/data1/workspaces/roborender/code_releases/release-1/other',
    '/data1/workspaces/roborender/code_releases-other/release-1/code',
    '/data1/workspaces/roborender/runs/release-1/code',
    '/data1/workspaces/roborender/code_releases/release-1/code/../../outside',
    'relative/release-1/code',
    '/tmp/release-1/code',
])
def test_no_other_lexical_locations(path):
    assert not locations.is_source_release_location(path)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    legacy = tmp_path / 'legacy' / 'source-releases'
    canonical = tmp_path / 'canonical' / 'code_releases'
    monkeypatch.setattr(locations, 'LEGACY_RELEASE_ROOT', legacy)
    monkeypatch.setattr(locations, 'CANONICAL_RELEASE_ROOT', canonical)
    monkeypatch.setattr(locations, 'LEGACY_RUNTIME_ROOT', legacy.parent)
    for root in (legacy, canonical):
        (root / 'r1' / 'code').mkdir(parents=True)
    return legacy, canonical


def test_resolves_both_physical_roots(roots):
    for root in roots:
        source = root / 'r1' / 'code'
        assert locations.resolve_source_release(source) == source
        assert locations.resolve_native_code(source) == source


def test_legacy_nonstandard_descendants_keep_original_admission(roots):
    legacy = roots[0] / 'historical-layout' / 'other'
    legacy.mkdir(parents=True)
    assert locations.is_source_release_location(legacy)
    assert locations.resolve_source_release(legacy) == legacy
    assert locations.is_source_release_location(roots[0])


def test_missing_or_file_is_not_a_release(roots):
    root = roots[1]
    with pytest.raises(FileNotFoundError):
        locations.resolve_source_release(root / 'missing' / 'code')
    leaf = root / 'r1' / 'frozen'
    leaf.write_text('not a directory')
    with pytest.raises(ValueError, match='immutable source'):
        locations.resolve_source_release(leaf)


def test_release_symlink_escape_rejected(roots, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    for root in roots:
        escaped = root / 'r1' / 'frozen'
        escaped.symlink_to(outside, target_is_directory=True)
        assert locations.is_source_release_location(escaped)  # preflight is not admission
        with pytest.raises(ValueError, match='immutable source'):
            locations.resolve_source_release(escaped)


def test_allowed_root_itself_cannot_redirect_outside(roots, tmp_path, monkeypatch):
    target = tmp_path / 'unapproved'
    (target / 'r1' / 'code').mkdir(parents=True)
    alias = tmp_path / 'allowed-looking'
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(locations, 'CANONICAL_RELEASE_ROOT', alias)
    with pytest.raises(ValueError, match='immutable source'):
        locations.resolve_source_release(alias / 'r1' / 'code')


def test_alias_into_approved_physical_release_preserves_legacy_behavior(roots, tmp_path):
    alias = tmp_path / 'compatibility'
    alias.symlink_to(roots[0] / 'r1' / 'code', target_is_directory=True)
    assert locations.resolve_source_release(alias) == roots[0] / 'r1' / 'code'


def test_old_diagnostic_runtime_scope_not_extended_to_canonical_runs(roots, tmp_path):
    old = roots[0].parent / 'diagnostic' / 'actor-code'
    old.mkdir(parents=True)
    assert locations.resolve_native_code(old) == old
    with pytest.raises(ValueError):
        locations.resolve_source_release(old)
    new = roots[1].parent / 'runs' / 'actor-code'
    new.mkdir(parents=True)
    with pytest.raises(ValueError):
        locations.resolve_native_code(new)


def test_environment_cannot_widen_jail(roots, tmp_path, monkeypatch):
    outside = tmp_path / 'outside' / 'r1' / 'code'
    outside.mkdir(parents=True)
    for name in ('ROBORENDER_SOURCE_ROOT', 'ROBORENDER_RELEASE_ROOT', 'ROBORENDER_STORAGE_PATHS'):
        monkeypatch.setenv(name, str(outside.parent.parent))
    with pytest.raises(ValueError):
        locations.resolve_source_release(outside)


def test_legacy_error_text_is_preserved(roots, tmp_path):
    with pytest.raises(ValueError, match='standing workers require immutable source'):
        locations.resolve_source_release(tmp_path, error='standing workers require immutable source')


def test_native_source_gate_checks_actual_manifest_after_canonical_jail(roots, monkeypatch):
    repository = REPO_ROOT
    monkeypatch.syspath_prepend(str(repository / 'pipeline/deployment/rtx_actor'))
    from qualify_leo import source_gate
    from pipeline.deployment.source_snapshot import canonical, file_record
    source = roots[1] / 'r1' / 'code'
    (source / 'README.md').write_text('immutable CPU test fixture\n')
    rows = [file_record(source, 'README.md')]
    digest = hashlib.sha256(canonical(rows)).hexdigest()
    manifest = source.parent / 'source-manifest.json'
    manifest.write_bytes(canonical(dict(schema='roborender.source-snapshot.v1',
        files=rows, file_count=1, source_sha256=digest)))
    assert source_gate(source, manifest) == digest
    (source / 'README.md').write_text('changed\n')
    with pytest.raises(ValueError, match='immutable source release mismatch'):
        source_gate(source, manifest)


def test_lexical_site_preflight_extends_only_leo_not_beta():
    from pipeline.deployment.beta_site import validate_site
    profiles = REPO_ROOT / 'pipeline/deployment/profiles'
    for name in ('leo_site.direct.example.json', 'beta_site.direct.example.json'):
        site = json.loads((profiles / name).read_bytes())
        assert validate_site(site) is site
        root = str(locations.CANONICAL_RELEASE_ROOT / 'r1' / 'code')
        site.update(rtx_source_root=root,
            rtx_source_manifest=str(Path(root).parent / 'source-manifest.json'),
            broker_proxy=root + '/pipeline/deployment/connectivity/cloud_proxy.py')
        if name.startswith('leo'):
            assert validate_site(site) is site
        else:
            with pytest.raises(ValueError, match='frozen runtime source release'):
                validate_site(site)


@pytest.mark.parametrize('name', [
    'qualify_leo.py', 'run_resident_isaac.py', 'run_production_resident_isaac.py',
    'run_small_batch_success.py', 'run_paired_student_success.py',
    'run_guarded_bc_capacity.py', 'run_bc_session.py', 'run_bc_pair_capacity.py',
    'run_bc_cohort.py', 'qualify_resident_pool.py', 'qualify_native_isaac6.py',
    'run_throughput_pair.py',
])
def test_absolute_launcher_import_without_pythonpath(tmp_path, name, monkeypatch):
    # --help exits in argparse before workload/asset/host actions. This also
    # catches source helper imports that accidentally depend on shell cwd.
    script = REPO_ROOT / 'pipeline/deployment/rtx_actor' / name
    monkeypatch.delenv('PYTHONPATH', raising=False)
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_canonical_resident_qualification_remains_plan_only(tmp_path):
    repository = REPO_ROOT
    source = locations.CANONICAL_RELEASE_ROOT / 'r1' / 'code'
    base = Path('/home/ldu/roborender')
    run = 'standing-isaac-leo-source-location-test'
    spec = dict(schema='roborender.resident-qualification-request.v1', run_id=run,
        source_root=str(source), source_manifest=str(source.parent / 'source-manifest.json'),
        source_sha256='a'*64, root=str(base / 'runtime/standing-isaac' / run),
        old_round_root=str(base / 'runtime/old-round'),
        cold_evaluation_root=str(base / 'runtime/cold-test'), panel_sha256='b'*64,
        checkpoints=dict(bc='c'*64, dagger='d'*64), python=str(base / '.venv/bin/python'))
    raw = json.dumps(spec).encode()
    path = tmp_path / 'spec.json'
    path.write_bytes(raw)
    script = repository / 'pipeline/deployment/rtx_actor/qualify_resident_pool.py'
    result = subprocess.run([sys.executable, str(script), '--spec', str(path),
        '--spec-sha256', hashlib.sha256(raw).hexdigest()], cwd=tmp_path,
        capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(result.stdout)['execute'] is False
    assert not (tmp_path / run).exists()
