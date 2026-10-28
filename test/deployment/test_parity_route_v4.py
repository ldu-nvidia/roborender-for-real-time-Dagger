"""Re-run real observer logic against synthetic /proc with current source pin."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime'
PATH=ROOT/'raven-parity-overnight-20261026-t6oMpN/route-v4-tools/diagnostic_route_observer.py'
if not PATH.is_file():pytest.skip('runtime route adapter absent',allow_module_level=True)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
wrapper=load(PATH,'tested_route_v4_entry')
reference=load(ROOT/'live-pipeline-smoke-20261025/test_diagnostic_route_observer.py','route_v4_reference_fixtures')
reference.m=wrapper._core
for name,value in vars(reference).items():
    if name=='case' or name.startswith('test_'):globals()[name]=value

def test_admission_changes_source_only_not_observation_implementation():
    assert wrapper._core.NATIVE_SHA==wrapper.NATIVE
    assert wrapper._core.UPSTREAM_SUPERVISOR_SHA=='f741e888c885bc6b8525468073031e369bbc14d0b2c831eab6e63aaac86e0aa8'
    assert wrapper._core.__file__==str(PATH)
    assert wrapper.observe_upstream is wrapper._core.observe_upstream
