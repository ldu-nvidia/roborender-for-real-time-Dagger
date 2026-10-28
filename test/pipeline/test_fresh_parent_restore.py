"""Fresh parent owner-restoration tests; no remote/GPU activity."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'runtime/raven-dagger-alignment-20261026-7pYLxI/restore_warm_owners.py'
if not PATH.is_file():
    pytest.skip('optional run-local fresh-parent restoration helper absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('fresh_parent_restore_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_remote_guard_and_actual_launch_scope():
    compile(m.COLLECTOR_EXIT,'<no-execution>','exec')
    plan=m.read(PATH.parent/'upgrade-plan.json')
    for g in (0,1):
        p=m.parameters(plan,g)
        assert p['gpu']==g and p['config']['source_sha256']==m.m.SOURCE
        assert '--scaled-teacher' in p['argv'] and '--scaled-success' in p['argv']
        assert p['config']['checkpoint_loaded_at_startup'] is False
        assert p['config']['compute_release_authorized'] is False


@pytest.mark.parametrize('gpu',[-1,2,True])
def test_invalid_gpu_rejected(gpu):
    with pytest.raises(ValueError):m.parameters({},gpu)


def test_launch_requires_explicit_execute():
    with pytest.raises(ValueError,match='explicit execute'):m.operate('launch')


def test_uncertain_saved_intent_only_reconciles(tmp_path):
    class Transport:
        def __init__(self):self.actions=[]
        def remote(self,p,action):self.actions.append(action);raise TimeoutError()
    p=m.parameters(m.read(PATH.parent/'upgrade-plan.json'),0);t=Transport();path=tmp_path/'intent.json'
    with pytest.raises(TimeoutError):m.m.n.n.launch_once(path,p,t)
    with pytest.raises(TimeoutError):m.m.n.n.launch_once(path,p,t)
    assert t.actions==['launch','status']
    assert m.read(path.with_suffix('.unknown.json'))['replay_authorized'] is False
