import copy
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).with_name('local_resource_journal.py')
spec=importlib.util.spec_from_file_location('journal_under_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def pair():
    rows=[]
    for kind in ('native','renderer'):
        rows.append(dict(schema=m.SCHEMA,kind=kind,slot=100,started_unix=500.1,
            finished_unix=501.0,owner={},value=dict(at_unix=501.0,native=[{'real':1}],
                renderers=[{'real':2}],host_available_bytes=123)))
    return rows

def test_merge_preserves_actual_observations():
    n,r=pair();row=m.merge_rows(n,r)
    assert row['native']==n['value']['native'] and row['renderers']==r['value']['renderers']
    assert row['host_available_bytes']==123
    assert row['schema']=='roborender.scaled-success-resource-sample.v1'

@pytest.mark.parametrize('field,value',[('slot',101),('slot',100.0),('kind','native'),
    ('started_unix',float('nan')),('finished_unix',float('inf')),('finished_unix',499),
    ('started_unix',490),('schema','untrusted')])
def test_no_interpolation_or_wrong_identity_time(field,value):
    n,r=pair();r[field]=value
    with pytest.raises(ValueError):m.merge_rows(n,r)

def plan(tmp_path):
    return dict(schema=m.SCHEMA,kind='native',period_seconds=5,lifetime_seconds=21600,
        output=str(tmp_path/'out'),probe_source=str(tmp_path/'probe'),source_root=str(tmp_path),
        native_operations=0,compute_release_authorized=False,helper_sha256=m.digest(PATH))

@pytest.mark.parametrize('key,value',[('period_seconds',10),('native_operations',1),
    ('compute_release_authorized',True),('kind','gpu-work'),('output','../bad')])
def test_readonly_bounded_plan(tmp_path,key,value):
    p=plan(tmp_path);p[key]=value
    with pytest.raises(ValueError):m.validate_plan(p)

def test_refuses_duplicate_launch(tmp_path):
    p=plan(tmp_path);Path(p['output']).mkdir();path=tmp_path/'plan.json';path.write_bytes(m.canonical(p))
    with pytest.raises(ValueError,match='accepted'):m.start(path)

def test_rejects_modified_probe(tmp_path):
    p=tmp_path/'probe.py';p.write_text("NATIVE='print(1)'")
    with pytest.raises(ValueError,match='changed'):m.probe_code(p,'native')
