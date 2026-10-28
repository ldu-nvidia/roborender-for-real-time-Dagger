"""Metadata renewal cannot address GPU work or skip exact-owner checks."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('resource_renewal_test',ROOT/'host_resource_resumed.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_retains_prior_frozen_source_and_new_observer_namespace():
    assert m.u is not m.live.u
    assert m.u.SOURCE==m.live.u.SOURCE=='6052af310ef5db8b7d9a5d1dee1522f3e7375b2fe4b0c0966668b97760e06c14'
    assert m.m.OUT==m.OUT and m.m.PLAN==m.PLAN and m.m.SAMPLES==m.SAMPLES
    assert m.PLAN!=m.old.PLAN and m.REMOTE!=m.old.REMOTE
    assert m.m.load is m.load
    assert m.u.authority is m.auth.authority
    with pytest.raises(ValueError,match='horizon'):
        m.live.u.authority(new_work=True)

def test_stop_surface_is_cooperative_metadata_only():
    compile(m.STOP,'metadata-stop','exec')
    tree=ast.parse(m.STOP)
    imports={n.name for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names}
    assert imports=={'hashlib','json','pathlib','sys'}
    for required in ("owner['identity']==p['identity']", "owner['plan_sha256']==p['plan_sha256']",
        "fields[19]==p['identity']['start_ticks']", "p['identity']['cmdline_sha256']",
        "p['identity']['boot_id']", "plan['native_operations']==0", "plan['compute_release_authorized'] is False",
        "root/'stop'", "host-journal-", "plan['output']==str(root)"):
        assert required in m.STOP
    assert 'kill(' not in m.STOP and 'docker' not in m.STOP and 'http' not in m.STOP

def test_existing_plan_rejects_second_prepare(monkeypatch):
    monkeypatch.setattr(m,'PLAN',ROOT/'user-authorization.json')
    with pytest.raises(ValueError,match='already exists'):
        m.prepare()

def test_binding_change_blocks_load(monkeypatch):
    monkeypatch.setattr(m.s,'read',lambda _: {'controller':{'path':'x','raw_sha256':'old'}})
    monkeypatch.setattr(m.s,'binding',lambda _: {'path':'x','raw_sha256':'new'})
    with pytest.raises(ValueError,match='scope changed'):
        m.load()
