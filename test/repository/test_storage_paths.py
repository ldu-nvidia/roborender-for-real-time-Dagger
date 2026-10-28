"""CPU-only path-registry tests; no remote filesystem or service operations."""
from test.deployment.source_paths import REPO_ROOT
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from roborender.infrastructure.paths import SCHEMA, SNAPSHOT_SCHEMA, load_registry


@pytest.fixture
def document():
    return dict(schema=SCHEMA, hosts={
        "cosmos": dict(roots={"repo": "/workspace/project/roborender", "shared": "/mnt/shared"},
                       artifacts={"bc.raw": dict(root="shared", relative="datasets/bc")}),
        "leo": dict(roots={"repo": "/data1/workspaces/roborender", "legacy": "/home/ldu/roborender"},
                    artifacts={"source_releases": dict(root="repo", relative="code_releases")}),
    })


def write(tmp_path, document):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def test_explicit_hosts_roots_and_artifacts(tmp_path, document):
    registry = load_registry(write(tmp_path, document))
    assert registry.resolve("leo", "source_releases") == Path("/data1/workspaces/roborender/code_releases")
    assert registry.resolve("cosmos", "bc.raw") == Path("/mnt/shared/datasets/bc")
    assert registry.resolve("leo", "legacy") == Path("/home/ldu/roborender")


@pytest.mark.parametrize("host,key", [("unknown", "repo"), ("leo", "missing"), ("leo", "bc.raw"),
                                      (None, "repo"), ("", "repo"), ("leo", None)])
def test_no_host_or_key_fallback(tmp_path, document, host, key):
    registry = load_registry(write(tmp_path, document))
    with pytest.raises(ValueError):
        registry.resolve(host, key)


def test_raw_hash_not_reserialized_hash_and_no_filesystem_probe(tmp_path, document, monkeypatch):
    path = write(tmp_path, document)
    registry = load_registry(path)
    assert registry.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("physical canonicalization"))
    monkeypatch.setattr(Path, "exists", lambda *_args, **_kwargs: pytest.fail("remote existence probe"))
    assert str(registry.resolve("leo", "repo")) == "/data1/workspaces/roborender"


def test_resolution_preserves_symlink_spelling(tmp_path, document):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    document["hosts"]["leo"]["roots"]["repo"] = str(link)
    registry = load_registry(write(tmp_path, document))
    assert registry.resolve("leo", "source_releases") == link / "code_releases"
    assert not (actual / "code_releases").exists()


@pytest.mark.parametrize("root", ["relative/path", "~/data", "/home/$USER/data", "/x/${NAME}",
                                  "/tmp/../data", "/tmp/./data", "/tmp//data", "//host/data",
                                  "/tmp/data/", "/tmp/with\nnewline", "C:\\data", "", 3])
def test_invalid_root_paths(tmp_path, document, root):
    document["hosts"]["leo"]["roots"]["repo"] = root
    with pytest.raises(ValueError):
        load_registry(write(tmp_path, document))


@pytest.mark.parametrize("relative", ["/absolute", "../escape", "x/../../escape", "x/../y", ".", "",
                                      "x//y", "x/./y", "x/", "$NAME", "~user", "x\\y", None])
def test_invalid_artifact_paths(tmp_path, document, relative):
    document["hosts"]["leo"]["artifacts"]["source_releases"]["relative"] = relative
    with pytest.raises(ValueError):
        load_registry(write(tmp_path, document))


@pytest.mark.parametrize("change", [
    lambda x: x.update(schema="unknown"),
    lambda x: x.update(default_host="leo"),
    lambda x: x.update(hosts={}),
    lambda x: x["hosts"]["leo"].update(extra="unknown"),
    lambda x: x["hosts"]["leo"].update(roots={}),
    lambda x: x["hosts"]["leo"].update(artifacts=[]),
    lambda x: x["hosts"]["leo"]["artifacts"].update(repo={"root": "repo", "relative": "x"}),
    lambda x: x["hosts"]["leo"]["artifacts"]["source_releases"].update(root="missing"),
    lambda x: x["hosts"]["leo"]["artifacts"]["source_releases"].update(secret="not-allowed"),
    lambda x: x["hosts"]["leo"]["artifacts"].update(bad="/absolute"),
    lambda x: x["hosts"]["leo"]["roots"].update({"bad/key": "/tmp"}),
])
def test_strict_schema_and_references(tmp_path, document, change):
    change(document)
    with pytest.raises(ValueError):
        load_registry(write(tmp_path, document))


def test_artifacts_optional(tmp_path, document):
    del document["hosts"]["leo"]["artifacts"]
    assert load_registry(write(tmp_path, document)).resolve("leo", "legacy") == Path("/home/ldu/roborender")


@pytest.mark.parametrize("raw", [b'{"schema":"x","schema":"y"}', b'NaN', b'Infinity',
                                 b'{', b'\xff', b'[]'])
def test_rejects_malformed_duplicate_and_nonfinite_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_registry(path)


def test_registry_final_symlink_and_special_files_rejected(tmp_path, document):
    path = write(tmp_path, document)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_registry(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        load_registry(fifo)


def test_bounded_input(tmp_path, monkeypatch):
    from roborender.infrastructure import paths
    monkeypatch.setattr(paths, "MAX_REGISTRY_BYTES", 16)
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 17)
    with pytest.raises(ValueError):
        load_registry(path)


def test_loaded_registry_and_detached_snapshots_cannot_drift(tmp_path, document):
    path = write(tmp_path, document)
    registry = load_registry(path)
    original = registry.snapshot("leo")
    modified = registry.snapshot("leo")
    modified["paths"]["repo"] = "/other"
    document["hosts"]["leo"]["roots"]["repo"] = "/changed-on-disk"
    write(tmp_path, document)
    assert registry.snapshot("leo") == original
    with pytest.raises(TypeError):
        registry._locations["leo"]["repo"] = Path("/modified")
    with pytest.raises(FrozenInstanceError):
        registry.raw_sha256 = "0" * 64


@pytest.mark.parametrize("keys", [[], ["repo", "repo"], ["unknown"], "repo", {"repo": True}, [None]])
def test_snapshot_rejects_bad_key_selection(tmp_path, document, keys):
    registry = load_registry(write(tmp_path, document))
    with pytest.raises(ValueError):
        registry.snapshot("leo", keys=keys)


def test_snapshot_selection_and_freeze_are_deterministic(tmp_path, document):
    registry = load_registry(write(tmp_path, document))
    target = tmp_path / "run-paths.json"
    value = registry.freeze("leo", target, keys=["source_releases", "repo"])
    assert value["schema"] == SNAPSHOT_SCHEMA
    assert value["registry_raw_sha256"] == registry.raw_sha256
    assert list(value["paths"]) == ["repo", "source_releases"]
    assert json.loads(target.read_bytes()) == value
    before = (target.stat().st_ino, target.read_bytes())
    assert registry.freeze("leo", target, keys=["repo", "source_releases"]) == value
    assert (target.stat().st_ino, target.read_bytes()) == before
    assert not list(tmp_path.glob(".storage-paths-*"))


def test_freeze_never_overwrites_different_snapshot(tmp_path, document):
    registry = load_registry(write(tmp_path, document))
    target = tmp_path / "run-paths.json"
    registry.freeze("leo", target)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        registry.freeze("cosmos", target)
    assert target.read_bytes() == before


def test_freeze_requires_parent_and_rejects_symlink_target(tmp_path, document):
    registry = load_registry(write(tmp_path, document))
    with pytest.raises(ValueError):
        registry.freeze("leo", tmp_path / "missing" / "paths.json")
    target = tmp_path / "real.json"
    registry.freeze("leo", target)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    before = target.read_bytes()
    with pytest.raises(OSError):
        registry.freeze("leo", link)
    assert target.read_bytes() == before


@pytest.mark.parametrize("same", [True, False])
def test_atomic_publication_reconciles_concurrent_writer(tmp_path, document, monkeypatch, same):
    from roborender.infrastructure import paths
    registry = load_registry(write(tmp_path, document))
    target = tmp_path / "run-paths.json"
    def race(source, destination, **_kwargs):
        Path(destination).write_bytes(Path(source).read_bytes() if same else b"another-writer\n")
        raise FileExistsError("race")
    monkeypatch.setattr(paths.os, "link", race)
    if same:
        assert registry.freeze("leo", target)["host"] == "leo"
    else:
        with pytest.raises(FileExistsError):
            registry.freeze("leo", target)
        assert target.read_bytes() == b"another-writer\n"
    assert not list(tmp_path.glob(".storage-paths-*"))


def test_verify_uses_original_bytes_not_current_registry_or_old_locator(tmp_path, document):
    path = write(tmp_path, document)
    original = path.read_bytes()
    registry = load_registry(path)
    target = tmp_path / "run-paths.json"
    registry.freeze("leo", target)
    document["hosts"]["leo"]["roots"]["repo"] = "/new/location"
    write(tmp_path, document)
    with pytest.raises(ValueError, match="preserved original"):
        load_registry(path).verify_snapshot(target)
    archived = tmp_path / "preserved-registry.json"
    archived.write_bytes(original)
    result = load_registry(archived).verify_snapshot(target)
    assert result["verified"] is True
    assert result["registry_raw_sha256"] == registry.raw_sha256
    assert result["snapshot_raw_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", [
    lambda x: x.update(schema="unknown"),
    lambda x: x.update(host="unknown"),
    lambda x: x.update(registry_path="relative"),
    lambda x: x.update(extra=True),
    lambda x: x.update(paths={}),
    lambda x: x["paths"].update(repo="/redirected"),
    lambda x: x["paths"].update(unknown="/unregistered"),
])
def test_verify_rejects_tampering(tmp_path, document, change):
    registry = load_registry(write(tmp_path, document))
    value = registry.snapshot("leo")
    change(value)
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        registry.verify_snapshot(target)


def cli(*arguments):
    return subprocess.run([sys.executable, "-m", "roborender.infrastructure.paths", *map(str, arguments)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)


def test_cli_resolve_snapshot_verify(tmp_path, document):
    path = write(tmp_path, document)
    result = cli("resolve", "--registry", path, "--host", "leo", "--key", "source_releases")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/data1/workspaces/roborender/code_releases"
    target = tmp_path / "paths.json"
    result = cli("snapshot", "--registry", path, "--host", "leo", "--output", target,
                 "--key", "source_releases")
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)["paths"]) == {"source_releases"}
    result = cli("verify", "--registry", path, "--snapshot", target)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified"] is True


def test_cli_requires_explicit_registry_host_and_output(tmp_path, document):
    path = write(tmp_path, document)
    before = set(tmp_path.iterdir())
    for arguments in [("resolve", "--host", "leo", "--key", "repo"),
                      ("resolve", "--registry", path, "--key", "repo"),
                      ("snapshot", "--registry", path, "--host", "leo"),
                      ("resolve", "--registry", path, "--host", "missing", "--key", "repo")]:
        result = cli(*arguments)
        assert result.returncode == 2 and not result.stdout
    assert set(tmp_path.iterdir()) == before


def test_repository_registry_compatibility():
    registry_path = REPO_ROOT / "configs/storage_paths.json"
    registry = load_registry(registry_path)
    assert registry.resolve("leo", "release_root") == Path("/data1/workspaces/roborender/code_releases")
    assert registry.resolve("cosmos", "source_root") == Path("/workspace/project/roborender")
    assert registry.resolve("leo", "manipulation_teacher").name == "g1_privileged_action_teacher_wsdpt_step044500.pt"


@pytest.mark.parametrize("host,key", [
    ("leo", "bc_epoch116"), ("cosmos", "bc_epoch116"),
    *(('cosmos', key) for key in ('bc_clean_demonstrations', 'bc_clean_rendered',
        'bc_clean_manifests', 'bc_clean_splits', 'bc_train_union_splits')),
])
def test_repository_registry_has_no_retired_bc_selectors(host, key):
    registry = load_registry(REPO_ROOT / "configs/storage_paths.json")
    with pytest.raises(ValueError, match="unknown path key"):
        registry.resolve(host, key)
