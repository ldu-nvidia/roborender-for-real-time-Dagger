"""CPU-only BC launcher path selection; this is not data/compute admission.

Fixed-final defaults come from the Cosmos storage registry's published train
union. Explicit pools are all-or-nothing, so one overridden split cannot be
combined with unrelated registry defaults. The trainer still revalidates the
complete admission and strict resume identity before loading training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

DATA_OPTIONS = {
    "--dataset-root": "DATASET_ROOT",
    "--train-split": "TRAIN_SPLIT",
    "--validation-split": "VALIDATION_SPLIT",
    "--data-admission-json": "DATA_ADMISSION_JSON",
}
OPTIONS = {**DATA_OPTIONS, "--training-protocol": "TRAINING_PROTOCOL"}
OUTPUT_KEYS = ("TRAINING_PROTOCOL", *DATA_OPTIONS.values())


def _literal(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{label} must be nonempty and contain no control characters")
    return value


def _overrides(environ, argv):
    selected = {key: _literal(environ[key], key) for key in OPTIONS.values() if key in environ}
    index = 0
    while index < len(argv):
        argument = argv[index]
        name, equals, value = argument.partition("=")
        if name in OPTIONS:
            if not equals:
                index += 1
                if index >= len(argv) or argv[index].startswith("--"):
                    raise ValueError(f"{name} requires an explicit value")
                value = argv[index]
            selected[OPTIONS[name]] = _literal(value, name)
        elif name.startswith("--") and any(option.startswith(name) for option in OPTIONS):
            # argparse otherwise permits some abbreviated options after our
            # selected arguments, bypassing the coherent-pool check.
            raise ValueError(f"spell the full BC data/protocol option: {name}")
        index += 1
    return selected


def _legacy_defaults(asset_root, dataset_root=None):
    root = Path(dataset_root) if dataset_root is not None else Path(asset_root) / "data/BC"
    splits = root / "splits_independent_v2"
    return {"DATASET_ROOT": str(root), "TRAIN_SPLIT": str(splits / "train.json"),
            "VALIDATION_SPLIT": str(splits / "validation.json"),
            "DATA_ADMISSION_JSON": str(splits / "data-admission.json")}


def _union_defaults(project_root, environ):
    # Import the one central resolver, never duplicate host/path inference.
    from roborender.infrastructure.paths import load_registry

    registry = Path(environ.get("ROBORENDER_STORAGE_REGISTRY", Path(project_root) / "configs/storage_paths.json"))
    location = load_registry(registry).resolve("cosmos", "bc_train_union_splits")
    admission = location / "data-admission.json"
    if not admission.is_file() or admission.stat().st_size > 4 << 20:
        raise ValueError("published BC training-union admission is missing or oversized")
    report = json.loads(admission.read_bytes())
    if (report.get("schema") != "roborender.bc-train-union-admission.v1"
            or report.get("production_admitted") is not True
            or report.get("training_protocol") != "fixed_final"
            or report.get("validation_enabled") is not False):
        raise ValueError("fixed_final default requires the versioned admitted train-only union")
    root = Path(_literal(report.get("dataset_root"), "admitted dataset_root"))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("training-union physical dataset root is absent or not absolute")
    values = {"DATASET_ROOT": str(root), "DATA_ADMISSION_JSON": str(admission)}
    for role, key, filename in (("train", "TRAIN_SPLIT", "train.json"),
                                ("validation", "VALIDATION_SPLIT", "validation-retired.json")):
        path = location / filename
        binding = report.get("split_files", {}).get(role, {})
        if (binding.get("path") != str(path.resolve()) or not path.is_file()
                or path.stat().st_size > 32 << 20
                or hashlib.sha256(path.read_bytes()).hexdigest() != binding.get("sha256")):
            raise ValueError(f"published training-union {role} index binding differs")
        values[key] = str(path)
    return values


def resolve_selection(*, project_root: Path, asset_root: Path, environ, argv=()):
    overrides = _overrides(environ, argv)
    protocol = overrides.get("TRAINING_PROTOCOL", "fixed_final")
    if protocol not in {"fixed_final", "legacy_validation"}:
        raise ValueError("TRAINING_PROTOCOL must be fixed_final or legacy_validation")
    explicit = {key: overrides[key] for key in DATA_OPTIONS.values() if key in overrides}
    if not explicit:
        values = (_union_defaults(project_root, environ) if protocol == "fixed_final"
                  else _legacy_defaults(asset_root))
    elif protocol == "legacy_validation" and set(explicit) == {"DATASET_ROOT"}:
        # Historical root-only selection derives all siblings from that root.
        values = _legacy_defaults(asset_root, explicit["DATASET_ROOT"])
    elif set(explicit) == set(DATA_OPTIONS.values()):
        values = explicit
    else:
        missing = ", ".join(sorted(set(DATA_OPTIONS.values()) - set(explicit)))
        raise ValueError("partial BC data override would mix pools; explicitly provide " + missing)
    return {"TRAINING_PROTOCOL": protocol, **values}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("trainer_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    trainer_arguments = args.trainer_arguments
    if trainer_arguments[:1] == ["--"]:
        trainer_arguments = trainer_arguments[1:]
    try:
        result = resolve_selection(project_root=args.project_root, asset_root=args.asset_root,
                                   environ=os.environ, argv=trainer_arguments)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for key in OUTPUT_KEYS:
        print(_literal(result[key], key))


if __name__ == "__main__":
    # The qualified Python uses PYTHONSAFEPATH and runs this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
