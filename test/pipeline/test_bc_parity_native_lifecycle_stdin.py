"""Optional run-local SSH framing regression; no remote calls or GPU work."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runtime/raven-stage-certification-20261025-kBFQ3q'
NEEDED=[BASE/'native_lifecycle.py',BASE/'native_lifecycle_v2.py',
    ROOT/'runtime/teacher-scaled295-20261025/native_upgrade.py',
    ROOT/'runtime/curriculum-certification-20261025/native_upgrade.py',
    ROOT/'runtime/live-pipeline-smoke-20261025/paired_native_deploy.py',
    ROOT/'runtime/live-pipeline-smoke-20261025/native_upgrade_bc_smoke.py']
if any(not p.is_file() for p in NEEDED):
    pytest.skip('optional run-local lifecycle transport chain absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('bc_lifecycle_stdin_under_test',BASE/'native_lifecycle_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_multi_megabyte_payload_roundtrips_without_large_argv():
    args=[json.dumps({'expected_commands':'x'*(2<<20)})]
    program="import json,sys; p=json.loads(sys.argv[1]); print(json.dumps({'bytes':len(p['expected_commands'])}))"
    calls=[]
    def local_bootstrap(argv,**kwargs):
        calls.append((argv,kwargs))
        assert max(map(len,argv))<4096 and sum(map(len,argv))<8192
        assert 'x'*1000 not in ' '.join(argv)
        return subprocess.run([sys.executable,'-B','-c',m.BOOTSTRAP],**kwargs)
    assert m.invoke_stdin(program,args,run=local_bootstrap)=={'bytes':2<<20}
    assert len(calls)==1


def test_stdin_preserves_unicode_quotes_and_newlines():
    args=['a\n"quoted"; $(not-executed) αβ','second']
    program='import json,sys; print(json.dumps(sys.argv[1:]))'
    def local_bootstrap(argv,**kwargs):
        return subprocess.run([sys.executable,'-B','-c',m.BOOTSTRAP],**kwargs)
    assert m.invoke_stdin(program,args,run=local_bootstrap)==args


def test_timeout_is_one_attempt_only():
    calls=[]
    def failed(argv,**kwargs):
        calls.append(argv);raise subprocess.TimeoutExpired(argv,kwargs['timeout'])
    with pytest.raises(subprocess.TimeoutExpired):m.invoke_stdin('x',[],run=failed)
    assert len(calls)==1


def test_nonzero_redacts_stderr_and_never_retries():
    calls=[]
    def failed(argv,**kwargs):
        calls.append(argv);return subprocess.CompletedProcess(argv,1,b'',b'private-token-must-not-print')
    with pytest.raises(RuntimeError) as e:m.invoke_stdin('x',[],run=failed)
    assert 'private-token' not in str(e.value) and len(calls)==1


def test_bounded_input_rejected_before_subprocess(monkeypatch):
    monkeypatch.setattr(m,'MAX_BYTES',256)
    def never(*args,**kwargs):raise AssertionError('must not invoke')
    with pytest.raises(ValueError,match='bounded'):m.invoke_stdin('x'*300,[],run=never)


def test_bounded_output_rejected(monkeypatch):
    monkeypatch.setattr(m,'MAX_BYTES',128)
    def oversized(argv,**kwargs):return subprocess.CompletedProcess(argv,0,b'x'*129,b'')
    with pytest.raises(RuntimeError):m.invoke_stdin('x',[],run=oversized)


def test_transport_is_installed_for_inherited_audit_and_actions(monkeypatch):
    calls=[]
    def fake(program,args,**kwargs):calls.append((program,args,kwargs));return {'checked':True}
    monkeypatch.setattr(m,'invoke_stdin',fake)
    assert m.m.Transport is m.Transport
    assert m.m.__file__==m.__file__
    assert m.Transport().remote({'config':{}},'verify')=={'checked':True}
    assert calls[0][0]==m.m.STAGE
    assert m.Transport().remote({'huge_history':'x'*(1<<20)},'shutdown')=={'checked':True}
    assert calls[1][0]==m.m.n.REMOTE
    assert len(calls)==2


def test_v1_evidence_is_preserved_and_fresh_attempt_root_used():
    assert m.ROOT.resolve()==(BASE/'lifecycle-v2').resolve()
    assert m.m.ROOT==m.ROOT
    assert m.m.sha((BASE/'native_lifecycle.py').read_bytes())==m.V1_SHA
    compile(m.ABSENCE,'absence','exec')
