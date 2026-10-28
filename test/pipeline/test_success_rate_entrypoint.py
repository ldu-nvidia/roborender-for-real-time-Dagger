"""Canonical entry-point/import regressions; no live GPU or native command."""
import ast
import importlib
import json
import subprocess
import sys

import pytest

from test.deployment.source_paths import REPO_ROOT, SUCCESS_RATE_ROOT
from test.pipeline.test_raven_pooled_benchmark import material, completed_result
from test.deployment.rtx_actor.test_scaled_success_contract import accepted
from pipeline.success_rate.__main__ import main


@pytest.mark.parametrize("old", [
    "pipeline.evaluation.scaled_success_qualification",
    "pipeline.evaluation.scaled_success_resources",
    "pipeline.evaluation.compare_paired_success",
    "pipeline.deployment.raven_pooled_dispatch",
    "pipeline.deployment.scaled_success_dispatch",
    "pipeline.deployment.paired_scaled_evaluation",
    "pipeline.deployment.paired_scaled_runtime",
    "pipeline.deployment.rendered_success_dispatch",
    "pipeline.deployment.rtx_actor.raven_pooled_benchmark",
    "pipeline.deployment.rtx_actor.success_diagnostic_core",
    "pipeline.deployment.rtx_actor.rendered_success_observations",
    "pipeline.deployment.rtx_actor.rendered_student_success_contract",
    "pipeline.deployment.rtx_actor.scaled_success_contract",
    "pipeline.deployment.rtx_actor.success_cycle_timing",
    "pipeline.deployment.rtx_actor.production_success_contract",
    "pipeline.deployment.rtx_actor.smoke_success_contract",
    "pipeline.dagger.raven_success",
])
def test_compatibility_is_same_module_not_a_second_implementation(old):
    renamed = {
        "raven_pooled_dispatch": "episode_dispatch",
        "raven_pooled_benchmark": "episode_protocol",
        "raven_success": "episode_metrics",
        "success_diagnostic_core": "episode_metrics",
        "success_cycle_timing": "episode_metrics",
        "rendered_success_observations": "student_observations",
        "rendered_success_dispatch": "rendered_student_success_contract",
    }
    name = old.rsplit(".", 1)[1]
    canonical = importlib.import_module("pipeline.success_rate." + renamed.get(name, name))
    assert importlib.import_module(old) is canonical
    assert SUCCESS_RATE_ROOT == __import__("pathlib").Path(canonical.__file__).parent
    source = REPO_ROOT / (old.replace(".", "/") + ".py")
    assert not any(isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in ast.walk(ast.parse(source.read_text())))


def test_help_does_not_initialize_native_or_torch():
    code = "from pipeline.success_rate.__main__ import main; import sys\n"
    code += "try: main(['--help'])\nexcept SystemExit as e: assert e.code == 0\n"
    code += "assert 'torch' not in sys.modules; assert 'gr00t' not in sys.modules; assert 'numpy' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True)


def test_cli_round_trip_and_no_clobber(accepted, material, tmp_path, capsys):
    panel, config, owners = material
    for name, value in zip(("panel", "config", "owners"), material):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    plan_path = tmp_path / "plan.json"
    args = ["plan", "--protocol", "raven-pooled-b16x2-v1", "--panel", str(tmp_path / "panel.json"),
        "--config", str(tmp_path / "config.json"), "--owners", str(tmp_path / "owners.json"),
        "--arm", "dagger", "--qualification-only", "--command-prefix", "cpu-pooled-benchmark",
        "--output", str(plan_path)]
    assert main(args) == 0
    assert main(["validate-plan", "--plan", str(plan_path)]) == 0
    results = []
    for i in range(2):
        path = tmp_path / f"result-{i}.json"
        path.write_text(json.dumps(completed_result(accepted, material, index=i)))
        results.append(str(path))
    summary = tmp_path / "summary.json"
    assert main(["aggregate", "--plan", str(plan_path), "--results", *results,
                 "--output", str(summary)]) == 0
    value = json.loads(summary.read_text())
    assert value["completed"] == 64 and value["weighted_success_rate"] == .5
    assert value["benchmark_complete"] is False and value["qualification_only"] is True
    assert value["compute_release_authorized"] is False
    changed = list(args)
    changed[changed.index("cpu-pooled-benchmark")] = "different-plan"
    with pytest.raises(ValueError, match="immutable"):
        main(changed)
    capsys.readouterr()


def test_partial_results_cannot_be_published_as_complete(accepted, material, tmp_path):
    from pipeline.success_rate.episode_dispatch import build_plan, aggregate
    plan = build_plan(*material, command_prefix="cpu-pooled-benchmark", qualification_only=True)
    first = completed_result(accepted, material)
    with pytest.raises(ValueError, match="both"):
        aggregate(plan, [first])
    second = completed_result(accepted, material, index=1)
    second["complete"] = False
    with pytest.raises(ValueError):
        aggregate(plan, [first, second])
