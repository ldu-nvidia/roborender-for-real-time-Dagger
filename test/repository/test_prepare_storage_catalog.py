from test.deployment.source_paths import REPO_ROOT
from pathlib import Path
import json

import pytest

from pipeline.deployment.prepare_storage_catalog import BC116, plan
from roborender.infrastructure.paths import load_registry


@pytest.fixture
def historical_registry(tmp_path):
    """Explicit dated migration fixture, never reconstructed in production."""
    original = REPO_ROOT / "configs/storage_paths.json"
    value = json.loads(original.read_text())
    cosmos = value['hosts']['cosmos']['artifacts']
    cosmos.update({
        'bc_epoch116': {'root': 'student_checkpoints', 'relative': 'behavior_cloning/bc_epoch116_2ec87a3c7b76.pt'},
        'bc_clean_demonstrations': {'root': 'datasets', 'relative': 'teacher_demonstrations/clean_reset_20261020'},
        'bc_clean_rendered': {'root': 'datasets', 'relative': 'rendered_bc/clean_reset_20261020'},
        'bc_clean_manifests': {'root': 'datasets', 'relative': 'split_manifests/clean_reset_20261020/manifests'},
        'bc_clean_splits': {'root': 'datasets', 'relative': 'split_manifests/clean_reset_20261020/splits'},
    })
    value['hosts']['leo']['artifacts']['bc_epoch116'] = dict(cosmos['bc_epoch116'])
    path = tmp_path/'historical-20261024-registry.json'
    path.write_text(json.dumps(value))
    return load_registry(path)


@pytest.mark.parametrize("host", ["leo", "cosmos"])
def test_plan_covers_every_artifact_without_mutating_legacy_roots(host, historical_registry):
    registry = historical_registry
    value = plan(registry, host)
    roots = [Path(p) for p in value["roots"]]
    destinations = [Path(r["path"]) for r in value["links"] + value["copy_files"]]
    assert len(destinations) == len(set(destinations))
    for destination in destinations:
        assert any(destination.is_relative_to(root) for root in roots)
    for row in value["links"] + value["copy_files"]:
        assert not any(Path(row["target"]).is_relative_to(root) for root in roots)
    assert len(value["copy_files"]) == 4
    selected = next(r for r in value["copy_files"] if r["path"] == str(registry.resolve(host, "bc_epoch116")))
    assert selected["sha256"] == BC116
    assert not any(".ssh" in str(p) for p in destinations)


def test_unknown_host_is_not_inferred():
    registry = load_registry(REPO_ROOT / "configs/storage_paths.json")
    with pytest.raises(ValueError):
        plan(registry, "auto")


@pytest.mark.parametrize("host", ["leo", "cosmos"])
def test_current_registry_cannot_recreate_retired_bc_catalog(host):
    registry = load_registry(REPO_ROOT / "configs/storage_paths.json")
    with pytest.raises(ValueError, match="unknown path key"):
        plan(registry, host)
