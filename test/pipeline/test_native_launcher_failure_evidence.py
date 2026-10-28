"""CPU-only cause-before-cleanup evidence; no GPU, signal or allocation calls."""
from test.deployment.source_paths import REPO_ROOT
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=REPO_ROOT
sys.path.insert(0,str(ROOT/'pipeline/deployment/rtx_actor'))
import run_production_resident_isaac as launcher


class CounterReadError(RuntimeError):
    def __init__(self,value):
        super().__init__('do-not-serialize arbitrary password=SECRET command')
        self.diagnostic=value


def test_bounded_field_diagnostic_kept_not_arbitrary_exception_text():
    detail={'field':'memory.pressure','error':'malformed_counter','attempts':3}
    value=launcher.sampling_diagnostic(CounterReadError(detail))
    assert value['error_type']=='CounterReadError'
    assert value['resource_read']==detail
    assert len(value['resource_read_diagnostic_sha256'])==64
    assert 'SECRET' not in json.dumps(value)
    assert launcher.sampling_diagnostic(ValueError('sensitive arbitrary text'))=={'error_type':'ValueError'}


@pytest.mark.parametrize('detail',[{'bad':float('nan')},{'bad':object()}])
def test_invalid_structured_read_diagnostic_not_serialized(detail):
    value=launcher.sampling_diagnostic(CounterReadError(detail))
    assert value['resource_read_diagnostic_unavailable']=='invalid_structured_value'
    assert 'resource_read' not in value


def test_oversized_read_diagnostic_is_hash_only():
    value=launcher.sampling_diagnostic(CounterReadError({'counter':'0'*40000}))
    assert value['resource_read_diagnostic_unavailable']=='bounded_size_exceeded'
    assert 'resource_read' not in value
    assert len(json.dumps(value))<512


def test_failure_is_durable_before_owned_cleanup_and_never_overwritten(tmp_path):
    observation={'counters_observed':False,'container':{'sampling_error':'native read failed',
        'sampling_diagnostic':launcher.sampling_diagnostic(CounterReadError({'field':'memory.max'}))}}
    proof=launcher.write_failure_evidence(tmp_path,RuntimeError('unknown counters'),
        container='standing-isaac-leo-prod-test-g0',source_sha256='1'*64,
        last_observation=observation,native_docker_pid=12,native_process_alive=True)
    path=Path(proof['path']);raw=path.read_bytes();value=json.loads(raw)
    assert proof['raw_sha256']==hashlib.sha256(raw).hexdigest()
    assert path.stat().st_mode&0o777==0o644
    assert value['native_process_alive'] is True
    assert value['cleanup_requested_by_launcher'] is True
    assert value['compute_release_authorized'] is False
    assert value['partial_native_state_replay_allowed'] is False
    assert value['last_resource_observation']==observation
    with pytest.raises(FileExistsError):
        launcher.write_failure_evidence(tmp_path,RuntimeError('later failure'),
            container='standing-isaac-leo-prod-test-g0',source_sha256='1'*64,
            last_observation=None,native_docker_pid=12,native_process_alive=False)
    assert path.read_bytes()==raw


def test_sustained_pressure_failure_keeps_exact_trigger_before_cleanup(tmp_path):
    from dagger_rtx.phase_resource_guard import PressureWindow, SustainedPressureError
    pressure=PressureWindow()
    sample={'some_avg10':8.34,'full_avg10':8.34}
    pressure.observe(sample,100.)
    with pytest.raises(SustainedPressureError) as caught:
        pressure.observe(sample,130.0145)
    diagnostic=launcher.sampling_diagnostic(caught.value)
    detail=diagnostic['resource_read']
    assert detail['reason_code']=='sustained_memory_pressure'
    assert detail['pressure_window']['failed'] is True
    assert detail['pressure_window']['maximum_warning_seconds']==pytest.approx(30.0145)
    assert detail['threshold_seconds']==30.
    assert detail['full_avg10']==8.34
    proof=launcher.write_failure_evidence(tmp_path,caught.value,
        container='standing-isaac-leo-prod-test-g0',source_sha256='1'*64,
        last_observation={'counters_observed':True},native_docker_pid=12,native_process_alive=True)
    result=json.loads(Path(proof['path']).read_bytes())
    assert result['error']==diagnostic
    assert result['compute_release_authorized'] is False
    assert result['partial_native_state_replay_allowed'] is False
