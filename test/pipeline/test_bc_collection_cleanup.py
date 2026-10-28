"""Retired collector entrypoints stay absent; the standalone path stays intact."""
import pytest

from test.deployment.source_paths import REPO_ROOT


STAGE = REPO_ROOT / "pipeline" / "bc_collection"


@pytest.mark.parametrize("name", [
    "publish_raven_bc_quota.py", "materialize_raven_bc_chunks.py",
    "seal_raven_bc_collection.py", "stream_render.py",
    "run_raven_bc_collection_8gpu.sh", "run_raven_bc_rank.sh",
    "audit_distribution.py", "audit_supervised_windows.py",
    "artifact_roles.py", "campaign_controller.py", "sampling_plan.py",
    "fresh_validation.py", "plan_native_sessions.py", "stage_campaign.py",
    "ingest_native_cohorts.py", "transfer_native_cohort.py",
    "materialize_raven_collection_runtime.sh", "raven_bc_collection_contract.v1.json",
    "validate_raven_bc_collection_contract.py",
])
def test_retired_collection_files_are_removed(name):
    assert not (STAGE / name).exists()


@pytest.mark.parametrize("name", [
    "collector.py", "chunk_writer.py", "continuous_plan.py",
    "run_continuous.py", "continuous_ingest.py", "storage_guard.py",
])
def test_standalone_collection_modules_remain(name):
    assert (STAGE / name).is_file()


def test_current_launcher_keeps_required_storage_helpers():
    from pipeline.bc_collection import run_continuous, storage_guard
    from pipeline.legacy.bc_collection import campaign_controller

    for name in ("validate_storage_policy", "storage_snapshot", "observe_storage"):
        assert getattr(run_continuous, name) is getattr(storage_guard, name)
        assert getattr(campaign_controller, name) is getattr(storage_guard, name)


def test_stage_contains_only_current_collection_implementation():
    assert {p.name for p in STAGE.iterdir() if p.is_file()} == {
        "README.md", "collector.py", "chunk_writer.py", "continuous_plan.py",
        "run_continuous.py", "continuous_ingest.py", "storage_guard.py",
        "streaming_storage.py", "episode_drain.py", "production_gate.py",
    }


def test_active_stage_does_not_import_legacy_collection():
    import ast

    for path in STAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pipeline.legacy")
            elif isinstance(node, ast.Import):
                assert not any(name.name.startswith("pipeline.legacy") for name in node.names)


def test_storage_guard_is_part_of_frozen_launcher_admission():
    from pipeline.bc_collection.run_continuous import REQUIRED_SOURCE

    assert "pipeline/bc_collection/storage_guard.py" in REQUIRED_SOURCE
    assert "pipeline/bc_collection/campaign_controller.py" not in REQUIRED_SOURCE
