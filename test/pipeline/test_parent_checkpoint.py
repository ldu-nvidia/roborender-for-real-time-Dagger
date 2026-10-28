import hashlib
from pathlib import Path
import pytest
from pipeline.dagger.services.parent_checkpoint import stage_parent

def fixture(tmp_path):
    source=tmp_path/'run-artifacts'/'checkpoint.pt';source.parent.mkdir();source.write_bytes(b'actual immutable checkpoint')
    return source,hashlib.sha256(source.read_bytes()).hexdigest(),tmp_path/'registry'/'parents'

def test_import_is_exact_and_idempotent_without_changing_original(tmp_path):
    source,digest,root=fixture(tmp_path);before=source.stat()
    target=stage_parent(source,digest,root)
    assert target==root/(digest+'.pt') and target.read_bytes()==source.read_bytes()
    assert target.stat().st_mode&0o222==0 and source.stat().st_mode==before.st_mode
    assert stage_parent(source,digest,root)==target
    assert list(root.iterdir())==[target]

@pytest.mark.parametrize('kind',['digest','source_symlink','target_symlink','target_bytes','root_symlink','relative_root'])
def test_import_fails_closed_without_overwriting(kind,tmp_path):
    source,digest,root=fixture(tmp_path)
    if kind=='digest':digest='0'*64
    if kind=='source_symlink':
        alias=tmp_path/'alias.pt';alias.symlink_to(source);source=alias
    if kind in ('target_symlink','target_bytes'):
        root.mkdir(parents=True);target=root/(digest+'.pt')
        if kind=='target_symlink':target.symlink_to(source)
        else:target.write_bytes(b'wrong')
    if kind=='root_symlink':root.parent.mkdir();root.symlink_to(source.parent,target_is_directory=True)
    if kind=='relative_root':root=Path('relative/parents')
    with pytest.raises(ValueError):stage_parent(source,digest,root)
    assert (tmp_path/'run-artifacts/checkpoint.pt').read_bytes()==b'actual immutable checkpoint'
