from pathlib import Path

import pytest

from pipeline.deployment.h200 import qualify_fresh_bc as q


@pytest.mark.parametrize("migration", ["alpha-to-beta-20261008", "beta-to-leo-20261013"])
def test_existing_and_leo_isolated_artifacts_are_admitted(tmp_path, monkeypatch, migration):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    root = tmp_path / "runtime/migrations" / migration
    source = root / "h200/run/isaac/first.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    second = source.with_name("second.npz")
    second.touch()
    output = root / "h200/run/render"
    assert q.validate_artifact_paths(source, output, [second]) == (source, output, [second])


@pytest.mark.parametrize("destination", ["data/BC/rendered", "runtime/elsewhere", "runtime/migrations/alpha-to-beta-20261008/render"])
def test_leo_does_not_escape_or_write_another_migration(tmp_path, monkeypatch, destination):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    source = tmp_path / "runtime/migrations/beta-to-leo-20261013/source.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    with pytest.raises(RuntimeError, match="isolated migration scope"):
        q.validate_artifact_paths(source, tmp_path / destination)


def test_symlink_output_cannot_reach_production_data(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    scope = tmp_path / "runtime/migrations/beta-to-leo-20261013"
    scope.mkdir(parents=True)
    source = scope / "source.npz"
    source.touch()
    production = tmp_path / "data/BC"
    production.mkdir(parents=True)
    (scope / "escape").symlink_to(production, target_is_directory=True)
    with pytest.raises(RuntimeError, match="isolated migration scope"):
        q.validate_artifact_paths(source, scope / "escape/render")
