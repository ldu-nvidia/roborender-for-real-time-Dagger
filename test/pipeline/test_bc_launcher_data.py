"""CPU-only path/default and shell wiring tests; never invoke a GPU learner."""
from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pipeline.bc_training.launcher_data import DATA_OPTIONS, resolve_selection


PROJECT = REPO_ROOT


def dump(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


@pytest.fixture
def published(tmp_path):
    root = tmp_path / "physical old pool"
    root.mkdir()
    union = tmp_path / "artifacts/split_manifests/merged"
    train = dump(union / "train.json", {"split": "train", "chunk_count": 32,
        "items": [{"stream_id": f"s{i}"} for i in range(32)]})
    validation = dump(union / "validation-retired.json", {
        "split": "validation", "chunk_count": 0, "items": [], "retired": True})
    report = dump(union / "data-admission.json", {
        "schema": "roborender.bc-train-union-admission.v1", "production_admitted": True,
        "training_protocol": "fixed_final", "validation_enabled": False,
        "dataset_root": str(root), "split_files": {
            role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in (("train", train), ("validation", validation))}})
    registry = dump(tmp_path / "storage_paths.json", {
        "schema": "roborender.storage-paths.v1", "hosts": {"cosmos": {
            "roots": {"datasets": str(tmp_path / "artifacts")},
            "artifacts": {"bc_train_union_splits": {
                "root": "datasets", "relative": "split_manifests/merged"}}}}})
    return dict(root=root, train=train, validation=validation, report=report,
                registry=registry, environ={"ROBORENDER_STORAGE_REGISTRY": str(registry)})


def select(published, *, environ=None, argv=()):
    return resolve_selection(project_root=PROJECT, asset_root=Path("/historical-assets"),
        environ=published["environ"] if environ is None else environ, argv=argv)


def test_fixed_final_default_uses_registry_and_admitted_physical_root(published):
    result = select(published)
    assert result == {"TRAINING_PROTOCOL": "fixed_final", "DATASET_ROOT": str(published["root"]),
        "TRAIN_SPLIT": str(published["train"]), "VALIDATION_SPLIT": str(published["validation"]),
        "DATA_ADMISSION_JSON": str(published["report"])}


def test_repository_retired_bc_default_fails_without_legacy_fallback(monkeypatch):
    from pipeline.bc_training import launcher_data
    monkeypatch.setattr(launcher_data, '_legacy_defaults',
                        lambda *_args, **_kwargs: pytest.fail('retired default fell back to legacy BC'))
    with pytest.raises(ValueError, match='unknown path key.*bc_train_union_splits'):
        resolve_selection(project_root=PROJECT, asset_root=Path('/historical-assets'), environ={})


@pytest.mark.parametrize("key", list(DATA_OPTIONS.values()))
def test_partial_fixed_final_override_fails_instead_of_mixing_pools(published, key):
    with pytest.raises(ValueError, match="partial BC data override"):
        select(published, environ={**published["environ"], key: "/other/pool"})


def test_explicit_environment_and_cli_form_one_complete_override(published):
    environ = {"DATASET_ROOT": "/explicit/root", "TRAIN_SPLIT": "/explicit/train",
               "VALIDATION_SPLIT": "/explicit/retired", "DATA_ADMISSION_JSON": "/explicit/admission"}
    result = select(published, environ=environ, argv=["--train-split=/override/train", "--max-epochs", "5"])
    assert result["TRAIN_SPLIT"] == "/override/train"
    assert result["DATASET_ROOT"] == "/explicit/root"
    assert result["DATA_ADMISSION_JSON"] == "/explicit/admission"


def test_complete_cli_overrides_need_no_registry(published):
    argv = [argument for option, key in DATA_OPTIONS.items() for argument in (option, f"/explicit/{key}")]
    result = select(published, environ={}, argv=argv)
    assert all(result[key] == f"/explicit/{key}" for key in DATA_OPTIONS.values())


def test_legacy_cli_protocol_controls_default_selection(published):
    result = select(published, environ={"TRAINING_PROTOCOL": "fixed_final"},
                    argv=["--training-protocol=legacy_validation"])
    assert result["TRAINING_PROTOCOL"] == "legacy_validation"
    assert result["DATASET_ROOT"] == "/historical-assets/data/BC"
    assert result["VALIDATION_SPLIT"] == "/historical-assets/data/BC/splits_independent_v2/validation.json"


def test_legacy_root_only_derives_all_old_siblings(published):
    result = select(published, environ={"TRAINING_PROTOCOL": "legacy_validation", "DATASET_ROOT": "/old"})
    assert result["TRAIN_SPLIT"] == "/old/splits_independent_v2/train.json"
    assert result["DATA_ADMISSION_JSON"] == "/old/splits_independent_v2/data-admission.json"


@pytest.mark.parametrize("argument", ["--train-s", "--dataset", "--training-prot"])
def test_abbreviated_controlled_flags_cannot_bypass_selection(published, argument):
    with pytest.raises(ValueError, match="spell the full"):
        select(published, argv=[argument, "/other"])


@pytest.mark.parametrize("change", ["missing_registry", "missing_admission", "wrong_schema",
                                   "wrong_root", "wrong_split_binding", "changed_index", "validation_enabled"])
def test_bad_union_default_never_falls_back_to_old2560(published, change):
    if change == "missing_registry":
        published["registry"].unlink()
    elif change == "missing_admission":
        published["report"].unlink()
    elif change == "changed_index":
        published["train"].write_text("{}")
    else:
        report = json.loads(published["report"].read_text())
        if change == "wrong_schema":
            report["schema"] = "roborender.bc-data-admission.v3"
        elif change == "wrong_root":
            report["dataset_root"] = "/missing/physical/pool"
        elif change == "wrong_split_binding":
            report["split_files"]["validation"]["path"] = str(published["train"])
        else:
            report["validation_enabled"] = True
        dump(published["report"], report)
    with pytest.raises((ValueError, OSError)):
        select(published)


def test_empty_override_and_control_characters_are_rejected(published):
    for value in ("", "/bad\npath"):
        with pytest.raises(ValueError, match="nonempty.*control"):
            select(published, environ={"TRAIN_SPLIT": value})


@pytest.fixture
def stub_launcher(tmp_path, published):
    """Copy only source and substitute the runtime/learner, not path selection."""
    source = tmp_path / "source"
    for relative in ("pipeline/bc_training/train_raven_bc_h200.sh", "pipeline/bc_training/launcher_data.py",
                     "roborender/infrastructure/paths.py"):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, destination)
    setup = source / "pipeline/common/setup_h200_isaac_runtime.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text('export GR00T_PYTHON ROBORENDER_ASSET_ROOT\n')
    python = tmp_path / "qualified-python-stub"
    python.write_text(f'#!{sys.executable}\n' + '''import json,os,pathlib,sys
args=sys.argv[1:]
if args[0].endswith('/train_bc_h200.py'):
    pathlib.Path(os.environ['CAPTURE_ARGUMENTS']).write_text(json.dumps(args[1:]))
elif args[0]=='-c':
    pass  # Dependency/key preflights only; never import GPU training libraries.
else:
    os.execv(sys.executable,[sys.executable,*args])
''')
    python.chmod(0o755)
    bootstrap = tmp_path / "bootstrap.pt"
    bootstrap.touch()
    spec = tmp_path / "policy.json"
    spec.write_text("{}")
    capture = tmp_path / "args.json"
    environment = {key: value for key, value in os.environ.items()
                   if key not in {*DATA_OPTIONS.values(), "TRAINING_PROTOCOL", "CHUNK_BATCH_SIZE", "PYTHONPATH", "PYTHONSAFEPATH"}}
    environment.update(published["environ"], GR00T_PYTHON=str(python), ROBORENDER_ASSET_ROOT=str(tmp_path),
        STUDENT_BOOTSTRAP=str(bootstrap), POLICY_SPEC=str(spec), WANDB_MODE="offline", CAPTURE_ARGUMENTS=str(capture))
    return source / "pipeline/bc_training/train_raven_bc_h200.sh", environment, capture


def test_shell_passes_union_root_retired_index_and_admission(stub_launcher, published):
    launcher, environment, capture = stub_launcher
    result = subprocess.run(["bash", str(launcher), "--max-epochs", "3"], env=environment,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[args.index("--dataset-root") + 1] == str(published["root"])
    assert args[args.index("--train-split") + 1] == str(published["train"])
    assert args[args.index("--validation-split") + 1] == str(published["validation"])
    assert args[args.index("--data-admission-json") + 1] == str(published["report"])
    assert "--require-production-data-admission" in args
    assert "train=32 validation=0" in result.stdout
    assert args[args.index("--chunk-batch-size") + 1] == "32"
    assert "batch_chunks=32 frames_per_update=2592 steps_per_epoch=1" in result.stdout


def test_shell_retired_repository_default_stops_before_learner(stub_launcher):
    launcher, environment, capture = stub_launcher
    environment['ROBORENDER_STORAGE_REGISTRY'] = str(PROJECT/'configs/storage_paths.json')
    result = subprocess.run(['bash', str(launcher), '--max-epochs', '3'], env=environment,
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'unknown path key' in result.stderr and 'bc_train_union_splits' in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("overrides,argv,protocol,batch", [
    ({"TRAINING_PROTOCOL": "legacy_validation"}, [], "legacy_validation", 16),
    ({"TRAINING_PROTOCOL": "fixed_final"}, ["--training-protocol=legacy_validation"], "legacy_validation", 16),
    ({"TRAINING_PROTOCOL": "legacy_validation"}, ["--training-protocol", "fixed_final"], "fixed_final", 32),
    ({"CHUNK_BATCH_SIZE": "16"}, [], "fixed_final", 16),
    ({"CHUNK_BATCH_SIZE": "16"}, ["--chunk-batch-size=32"], "fixed_final", 32),
    ({"CHUNK_BATCH_SIZE": "32"}, ["--chunk-batch-size", "16"], "fixed_final", 16),
])
def test_shell_batch_defaults_follow_effective_protocol_and_explicit_overrides(
        stub_launcher, published, overrides, argv, protocol, batch):
    launcher, environment, capture = stub_launcher
    # Explicit paths keep this CPU wiring test independent of historical assets.
    environment.update({key: str(published[name]) for key, name in (
        ("DATASET_ROOT", "root"), ("TRAIN_SPLIT", "train"),
        ("VALIDATION_SPLIT", "validation"), ("DATA_ADMISSION_JSON", "report"))})
    environment.update(overrides)
    result = subprocess.run(["bash", str(launcher), *argv, "--max-epochs", "3"],
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[args.index("--training-protocol") + 1] == protocol
    assert args[args.index("--chunk-batch-size") + 1] == str(batch)
    assert f"batch_chunks={batch} frames_per_update={batch * 81} steps_per_epoch={32 // batch}" in result.stdout


def test_shell_legacy_batch32_is_rejected_before_learner(stub_launcher):
    launcher, environment, capture = stub_launcher
    environment.update(TRAINING_PROTOCOL="legacy_validation", CHUNK_BATCH_SIZE="32")
    result = subprocess.run(["bash", str(launcher)], env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Historical validation BC retains CHUNK_BATCH_SIZE=16" in result.stderr
    assert not capture.exists()


def test_shell_default32_wraps_small_population_without_dropping_or_padding(stub_launcher, published):
    launcher, environment, capture = stub_launcher
    train = json.loads(published["train"].read_text())
    train.update(chunk_count=16, items=train["items"][:16])
    dump(published["train"], train)
    report = json.loads(published["report"].read_text())
    report["split_files"]["train"]["sha256"] = hashlib.sha256(published["train"].read_bytes()).hexdigest()
    dump(published["report"], report)
    result = subprocess.run(["bash", str(launcher), "--max-epochs", "3"],
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "steps_per_epoch=1 persistent_cyclic=True" in result.stdout
    assert capture.exists()


@pytest.mark.parametrize("argv", [["--chunk-batch-size", "32"], ["--chunk-batch-size=32"],
    ["--chunk-batch-size", "16", "--chunk-batch-size=32"]])
def test_shell_cli_batch32_controls_preflight_too(stub_launcher, argv):
    launcher, environment, capture = stub_launcher
    result = subprocess.run(["bash", str(launcher), *argv, "--max-epochs", "3"],
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "batch_chunks=32 frames_per_update=2592 steps_per_epoch=1" in result.stdout
    args = json.loads(capture.read_text())
    assert args[args.index("--chunk-batch-size") + 1] == "32"


@pytest.mark.parametrize("argv", [["--chunk-batch-size", "8"], ["--chunk-batch-size"],
                                 ["--chunk-batch-s", "32"]])
def test_shell_rejects_unsupported_or_ambiguous_batch_before_learner(stub_launcher, argv):
    launcher, environment, capture = stub_launcher
    result = subprocess.run(["bash", str(launcher), *argv], env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert not capture.exists()


def test_shell_partial_cli_override_exits_before_learner(stub_launcher):
    launcher, environment, capture = stub_launcher
    result = subprocess.run(["bash", str(launcher), "--train-split", "/other/train.json"],
                            env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert "partial BC data override" in result.stderr
    assert not capture.exists()
