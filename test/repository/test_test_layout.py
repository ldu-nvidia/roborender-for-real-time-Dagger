"""Keep first-party tests separate from importable production code."""
from test.deployment.source_paths import REPO_ROOT
import ast
import configparser
from pathlib import Path

import pytest


ROOT = REPO_ROOT
TEST_ROOTS = ("test",)
EXCLUDED_ROOTS = ("test", "pipeline/vendor", "pipeline/GR00T-VisualSim2Real", "runtime")
# This is an actual evaluation-plan builder, not a test module.
PRODUCTION_TEST_SUFFIX = {"pipeline/bc_training/plan_success_test.py"}


def excluded_source(relative):
    return any(relative.is_relative_to(Path(directory)) for directory in EXCLUDED_ROOTS)


def production_sources():
    from pipeline.deployment.source_snapshot import source_names
    for name in source_names(ROOT):
        relative = Path(name)
        if relative.suffix == ".py" and not excluded_source(relative):
            yield relative, ROOT / relative


def is_test_import(module):
    return any(part in {"test", "tests", "pytest", "unittest"} or part.startswith("test_")
               for part in module.split("."))


def test_first_party_test_files_are_not_embedded_in_production():
    assert excluded_source(Path("test/pipeline/test_fixture.py"))
    assert not excluded_source(Path("pipeline/foo/test/test_fixture.py"))
    embedded = [str(relative) for relative, _ in production_sources()
                if (relative.name == "conftest.py" or relative.name.startswith("test_")
                    or relative.name.endswith("_test.py"))
                and str(relative) not in PRODUCTION_TEST_SUFFIX]
    assert embedded == []


def test_production_does_not_import_test_modules():
    imports = []
    for relative, path in production_sources():
        for node in ast.walk(ast.parse(path.read_text(), filename=str(relative))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [alias.name for alias in node.names]
                       if isinstance(node, ast.Import) else [])
            imports.extend(f"{relative}:{node.lineno}: {module}"
                           for module in modules if is_test_import(module))
            if isinstance(node, ast.ImportFrom) and not node.module:
                imports.extend(f"{relative}:{node.lineno}: {alias.name}"
                               for alias in node.names if is_test_import(alias.name))
    assert imports == []


@pytest.mark.parametrize("module", ["test_actor", "pipeline.tests.fixture",
                                    "pipeline.deployment.test_actor",
                                    "pytest", "unittest.mock"])
def test_test_import_detection_includes_legacy_and_dedicated_paths(module):
    assert is_test_import(module)
    assert is_test_import("test.pipeline.fixture")


def test_default_discovery_includes_all_dedicated_test_roots_only():
    from pipeline.deployment.source_snapshot import source_names
    config = configparser.ConfigParser()
    config.read(ROOT / "pytest.ini")
    assert tuple(config["pytest"]["testpaths"].split()) == TEST_ROOTS
    assert (ROOT / "test/run_cpu_checks.sh").is_file()
    for folder in ("tests", "pipeline/tests", "pipeline/dagger/rtx/tests"):
        assert not list((ROOT / folder).rglob("*.py"))
    assert not (ROOT / "pipeline/deployment/run_cpu_checks.sh").exists()
    for name in source_names(ROOT):
        relative = Path(name)
        if name == "pytest.ini" or excluded_source(relative):
            continue
        if relative.suffix in {".ini", ".cfg", ".toml"}:
            text = (ROOT / relative).read_text()
            assert not any(section in text for section in
                           ("[pytest]", "[tool:pytest]", "[tool.pytest.ini_options]")), name
