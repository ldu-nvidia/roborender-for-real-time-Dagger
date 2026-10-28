"""CPU-only create-time zero-swap property and exact owned-scope admission."""
from test.deployment.source_paths import REPO_ROOT
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys

import pytest

ROOT=REPO_ROOT
sys.path.insert(0,str(ROOT/'pipeline/deployment/rtx_actor'))
import run_production_resident_isaac as launcher

GROUP=Path('/sys/fs/cgroup/system.slice/docker-'+64*'a'+'.scope')


def properties(**changes):
    value=dict(ControlGroup=str(GROUP).removeprefix('/sys/fs/cgroup'),
        MemoryMax=str(21<<30),MemorySwapMax='0',CPUQuotaPerSecUSec='12s')
    value.update(changes)
    return ''.join(key+'='+content+'\n' for key,content in value.items())


def test_create_time_annotation_keeps_original_zero_swap_and_B2(monkeypatch, tmp_path):
    base=['docker','run','--memory','21g','--memory-swap','21g','--cpus','12',
        launcher.IMAGE,'num_envs=6','RESIDENT_MOTION_INVENTORY=old']
    monkeypatch.setattr(launcher,'build_resident_command',lambda **kwargs:list(base))
    actual=launcher.production_command(motion_inventory_sha256='b'*64, jobs=tmp_path/'jobs')
    at=actual.index('--annotation')
    assert actual[at+1]=='org.systemd.property.MemorySwapMax=uint64 0'
    assert at<actual.index(launcher.IMAGE)
    assert actual[actual.index('--memory')+1]==actual[actual.index('--memory-swap')+1]=='21g'
    assert actual[actual.index('--cpus')+1]=='12'
    assert 'num_envs=2' in actual and 'num_envs=6' not in actual
    assert 'RESIDENT_MOTION_INVENTORY=/resident/jobs/full-motion-inventory.json' in actual
    assert ('type=bind,src='+str(tmp_path/'scene-reconfigure.lock')+
            ',dst=/resident/scene-reconfigure.lock,readonly') in actual
    assert ('type=bind,src='+str(tmp_path/'population-barriers')+
            ',dst=/resident/population-barriers') in actual
    assert 'RESIDENT_POPULATION_BARRIERS=/resident/population-barriers' in actual


def test_population_journal_is_shared_private_and_does_not_mutate_existing_receipts(tmp_path):
    root=launcher.prepare_population_barriers(tmp_path)
    assert root==tmp_path/'population-barriers' and root.stat().st_mode&0o777==0o700
    (root/'existing-receipt.json').write_text('preserved')
    assert launcher.prepare_population_barriers(tmp_path)==root
    assert (root/'existing-receipt.json').read_text()=='preserved'


def test_population_journal_rejects_symlink_and_missing_pool(tmp_path):
    with pytest.raises(FileNotFoundError):launcher.prepare_population_barriers(tmp_path/'missing')
    (tmp_path/'elsewhere').mkdir()
    (tmp_path/'population-barriers').symlink_to(tmp_path/'elsewhere',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        launcher.prepare_population_barriers(tmp_path)


def test_ambiguous_inherited_swap_annotation_rejected(monkeypatch):
    monkeypatch.setattr(launcher,'build_resident_command',lambda **kwargs:[
        'docker','run','--annotation','org.systemd.property.MemorySwapMax=uint64 99',
        launcher.IMAGE,'num_envs=6'])
    with pytest.raises(RuntimeError,match='inherited'):
        launcher.production_command(motion_inventory_sha256='b'*64)


def test_exact_systemd_scope_zero_property_required_before_native_qualification():
    calls=[]
    def run(argv,**kwargs):
        calls.append((argv,kwargs))
        return SimpleNamespace(returncode=0,stdout=properties(),stderr='')
    result=launcher.attest_systemd_scope(GROUP,run=run)
    assert calls[0][0]==['systemctl','show',GROUP.name,
        '--property=ControlGroup,MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec']
    assert calls[0][1]==dict(capture_output=True,text=True,timeout=3)
    assert result['passed'] is True
    assert result['properties']['MemorySwapMax']=='0'
    assert result['global_host_policy_modified'] is False
    assert result['property_output_sha256']==hashlib.sha256(properties().encode()).hexdigest()


@pytest.mark.parametrize('change',[dict(MemorySwapMax='infinity'),dict(MemorySwapMax='1'),
    dict(MemoryMax='infinity'),dict(MemoryMax=str(28<<30)),dict(CPUQuotaPerSecUSec='infinity'),
    dict(CPUQuotaPerSecUSec='6s'),dict(ControlGroup='/system.slice/docker-'+64*'b'+'.scope')])
def test_effective_kernel_zero_cannot_waive_wrong_systemd_scope_policy(change):
    with pytest.raises(launcher.NativeScopeAdmissionError) as failure:
        launcher.attest_systemd_scope(GROUP,run=lambda *a,**k:SimpleNamespace(
            returncode=0,stdout=properties(**change),stderr=''))
    assert failure.value.diagnostic['reason']=='systemd_scope_resource_properties_differ'


@pytest.mark.parametrize('raw',[properties()+'MemorySwapMax=0\n',
    properties().replace('MemorySwapMax=0\n',''),properties()+'Other=sensitive-password\n',
    properties(MemorySwapMax='secret-auth-token')])
def test_ambiguous_missing_or_unexpected_properties_fail_without_exposing_text(raw):
    with pytest.raises(launcher.NativeScopeAdmissionError) as failure:
        launcher.attest_systemd_scope(GROUP,run=lambda *a,**k:SimpleNamespace(
            returncode=0,stdout=raw,stderr=''))
    diagnostic=json.dumps(failure.value.diagnostic)
    assert 'sensitive-password' not in diagnostic and 'secret-auth-token' not in diagnostic


@pytest.mark.parametrize('path',['/sys/fs/cgroup/system.slice',
    '/sys/fs/cgroup/user.slice/docker-'+64*'a'+'.scope','relative/docker-'+64*'a'+'.scope',
    '/sys/fs/cgroup/system.slice/docker-not-owned.scope'])
def test_ancestor_other_slice_and_unowned_scope_never_queried(path):
    def no_call(*a,**k):raise AssertionError('unowned systemctl query')
    with pytest.raises(launcher.NativeScopeAdmissionError):
        launcher.attest_systemd_scope(path,run=no_call)


@pytest.mark.parametrize('status,stdout',[(1,''),(0,'x'*4097)])
def test_denied_or_oversized_scope_observation_is_terminal_and_stderr_digest_only(status,stdout):
    with pytest.raises(launcher.NativeScopeAdmissionError) as failure:
        launcher.attest_systemd_scope(GROUP,run=lambda *a,**k:SimpleNamespace(
            returncode=status,stdout=stdout,stderr='password=SECRET'))
    result=launcher.sampling_diagnostic(failure.value)
    assert result['resource_read']['reason']=='systemd_scope_observation_unavailable'
    assert result['resource_read']['stderr_sha256']==hashlib.sha256(b'password=SECRET').hexdigest()
    assert 'SECRET' not in json.dumps(result)


def test_timeout_uses_constant_reason_without_raw_command_or_message():
    def timeout(*a,**k):raise subprocess.TimeoutExpired('secret command',3,stderr='auth-token=SECRET')
    with pytest.raises(launcher.NativeScopeAdmissionError) as failure:
        launcher.attest_systemd_scope(GROUP,run=timeout)
    assert failure.value.diagnostic['error_type']=='TimeoutExpired'
    assert 'SECRET' not in json.dumps(failure.value.diagnostic)


def test_source_does_not_repair_caps_or_reload_global_systemd():
    text=Path(launcher.__file__).read_text()
    assert "['systemctl','show',root.name" in text
    assert 'systemctl set-property' not in text and 'daemon-reload' not in text.replace(
        'daemon-reload can then restore infinity.','')
    assert "except NativeScopeAdmissionError:\n                        raise" in text
    assert "path=runtime/'systemd-scope-admission.json'" in text
