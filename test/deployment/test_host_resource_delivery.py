"""Actual read-only journal delivery: partial writes and terminal owner faults."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module():
    spec=importlib.util.spec_from_file_location('host_delivery_test',ROOT/'host_resource_transport.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def read(m,path,offset=0):
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',m.READ,str(path),str(offset)]))

def setup(m,path,state='RUNNING'):
    owner=m.w.identity()
    (path/'owner.json').write_text(json.dumps(dict(identity=owner,plan_sha256='a'*64)))
    (path/'status.json').write_text(json.dumps(dict(identity=owner,state=state)))
    return owner

def test_partial_tail_is_not_admitted_or_skipped(tmp_path):
    import base64
    m=module();setup(m,tmp_path)
    complete=b'{"actual":1}\n';partial=b'{"actual":'
    (tmp_path/'resources.jsonl').write_bytes(complete+partial)
    value=read(m,tmp_path)
    assert base64.b64decode(value['data'])==complete and value['next_offset']==len(complete)
    assert base64.b64decode(read(m,tmp_path,len(complete))['data'])==b''
    (tmp_path/'resources.jsonl').write_bytes(complete+partial+b'2}\n')
    assert base64.b64decode(read(m,tmp_path,len(complete))['data'])==b'{"actual":2}\n'

def test_terminal_host_fault_is_not_network_unknown(tmp_path):
    m=module();setup(m,tmp_path,state='FAILED')
    assert read(m,tmp_path)==dict(terminal_observer_fault='AssertionError')

def test_changed_owner_is_terminal_not_relabelled(tmp_path):
    m=module();owner=setup(m,tmp_path);owner['start_ticks']='not-this-process'
    (tmp_path/'owner.json').write_text(json.dumps(dict(identity=owner,plan_sha256='a'*64)))
    assert read(m,tmp_path)==dict(terminal_observer_fault='AssertionError')
