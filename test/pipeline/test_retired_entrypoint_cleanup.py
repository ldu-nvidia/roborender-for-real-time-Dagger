"""Retired executable bodies stay absent; current shared helpers remain usable."""
from test.deployment.source_paths import REPO_ROOT
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = REPO_ROOT
TRIMMED = (
    "pipeline/deployment/beta_site.py",
    "pipeline/deployment/resident_campaign_site.py",
    "pipeline/deployment/plan.py",
    "pipeline/deployment/rtx_actor/run_paired_student_success.py",
    "pipeline/deployment/rtx_actor/run_small_batch_success.py",
    "pipeline/deployment/rtx_actor/run_success_diagnostic.py",
)


@pytest.mark.parametrize("relative", TRIMMED)
def test_retired_guard_has_no_unreachable_execution_body(relative):
    tree = ast.parse((ROOT / relative).read_text())
    for node in ast.walk(tree):
        for key in ("body", "orelse", "finalbody"):
            statements = getattr(node, key, None)
            if not isinstance(statements, list):
                continue
            assert not any(isinstance(item, (ast.Return, ast.Raise, ast.Break, ast.Continue))
                           for item in statements[:-1]), (relative, node.lineno)


def test_cold_legacy_evaluate_rejects_before_accessing_runtime():
    from pipeline.deployment.beta_site import Site
    # No site/config/files or native handles are supplied: retirement must be
    # the first operation, not a historical transfer or process launch.
    with pytest.raises(RuntimeError, match="execution is disabled"):
        Site.evaluate(SimpleNamespace())


def test_resident_explicit_route_survives_and_legacy_route_rejects(monkeypatch):
    from pipeline.deployment.resident_campaign_site import ResidentSite
    from pipeline.dagger.rtx.code.src.dagger_rtx import beta_round
    calls = []
    site = SimpleNamespace(config={}, evaluate_explicit=lambda: calls.append("rendered") or "kept")
    monkeypatch.setattr(beta_round, "uses_explicit_motion", lambda config: True)
    assert ResidentSite.evaluate(site) == "kept"
    assert calls == ["rendered"]
    monkeypatch.setattr(beta_round, "uses_explicit_motion", lambda config: False)
    with pytest.raises(ValueError, match="legacy ray-traced student execution is disabled"):
        ResidentSite.evaluate(site)
    assert calls == ["rendered"]


@pytest.mark.parametrize("relative,names", [
    ("pipeline/deployment/rtx_actor/run_paired_student_success.py", {"sha", "build_command", "main"}),
    ("pipeline/deployment/rtx_actor/run_small_batch_success.py", {"execute_container", "build_warm_command", "main"}),
    ("pipeline/deployment/rtx_actor/run_success_diagnostic.py", {"gpu_gate", "main"}),
])
def test_compatibility_and_production_helpers_remain(relative, names):
    tree = ast.parse((ROOT / relative).read_text())
    assert names <= {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


@pytest.mark.parametrize("relative", [
    "pipeline/evaluation/run_raven_raytraced_300_eval_h200.sh",
])
@pytest.mark.parametrize("arguments", [[], ["--execute", "/missing/checkpoint", "/missing/config", "32"]])
def test_retired_shells_refuse_before_tools_or_asset_access(relative, arguments, tmp_path):
    sentinel = tmp_path / "tools"
    sentinel.mkdir()
    marker = tmp_path / "external-command-ran"
    for command in ("python", "python3", "docker", "nvidia-smi", "mkdir", "cp", "find",
                    "unshare", "setsid", "date", "dirname", "sha256sum", "tee"):
        tool = sentinel / command
        tool.write_text('#!/bin/bash\nprintf touched > "' + str(marker) + '"\nexit 97\n')
        tool.chmod(0o755)
    before = set(tmp_path.rglob("*"))
    completed = subprocess.run(["/bin/bash", str(ROOT / relative), *arguments],
        cwd=tmp_path, env={"PATH": str(sentinel)}, capture_output=True, text=True, timeout=5)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Retired" in completed.stderr
    assert not marker.exists()
    assert set(tmp_path.rglob("*")) == before
