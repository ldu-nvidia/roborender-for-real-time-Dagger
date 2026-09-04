#!/usr/bin/env python3
"""Finalize the external Lepton backend contract after source/environment seal.

The checked-in contract is a source-inventoried template and intentionally
contains zero SHA placeholders.  Writing the source inventory SHA back into
that template would be self-referential.  This command instead verifies the
already sealed source and live Python environment, substitutes only the two
external evidence hashes, and publishes a create-only mode-0444 contract in
``PROJECT_ROOT/manifests`` outside the sealed source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

# This tool runs from the sealed source tree and imports ``round_dagger``
# below.  Never let that import create ``__pycache__`` in the immutable tree:
# a cache directory is both an unexpected mutable artifact and causes the
# independent source verifier to reject an otherwise valid seal.  Set this
# before importing any repository module so callers do not need to remember a
# process-environment flag.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.lepton_job import (  # noqa: E402
    LeptonBackendContract,
    LeptonJobError,
    load_contract,
    require_finalized_supply_chain,
    sha256_file,
    verify_python_environment_freeze,
    verify_sealed_contract_source,
)


def _immutable_sha(path: Path, label: str) -> str:
    path = Path(os.path.abspath(path))
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise LeptonJobError(f"{label} does not exist") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise LeptonJobError(f"{label} must be a regular mode-0444 file")
    digest = sha256_file(path)
    after = os.lstat(path)
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise LeptonJobError(f"{label} changed while it was hashed")
    return digest


def finalized_document(
    template: Mapping[str, Any],
    *,
    source_inventory_sha256: str,
    python_freeze_sha256: str,
) -> dict[str, Any]:
    """Return a deep-copy with only the two zero placeholders substituted."""

    document = json.loads(json.dumps(template, allow_nan=False))
    try:
        supply = document["supply_chain"]
    except (KeyError, TypeError) as exc:
        raise LeptonJobError("backend template supply_chain is malformed") from exc
    if (
        supply.get("source_inventory_sha256") != "0" * 64
        or supply.get("python_freeze_sha256") != "0" * 64
    ):
        raise LeptonJobError(
            "backend template must retain both all-zero seal placeholders"
        )
    supply["source_inventory_sha256"] = source_inventory_sha256
    supply["python_freeze_sha256"] = python_freeze_sha256
    return document


def _publish(path: Path, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise LeptonJobError(
                "finalized backend contract destination appeared concurrently"
            ) from exc
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_contract(
    *,
    template_path: Path,
    output_path: Path,
) -> LeptonBackendContract:
    template = Path(os.path.abspath(template_path.expanduser()))
    if template.is_symlink() or not template.is_file():
        raise LeptonJobError("backend template is missing or unsafe")
    contract = load_contract(template, immutable=True)
    if (
        contract.source_inventory_sha256 != "0" * 64
        or contract.python_freeze_sha256 != "0" * 64
    ):
        raise LeptonJobError("backend template is already finalized")

    output = Path(os.path.abspath(output_path.expanduser()))
    expected_parent = Path(str(contract.project_root)) / "manifests"
    if output.parent != expected_parent or output.suffix != ".json":
        raise LeptonJobError(
            "finalized backend contract must be a JSON file directly in PROJECT_ROOT/manifests"
        )
    source_root = Path(str(contract.source_root))
    if output == source_root or source_root in output.parents:
        raise LeptonJobError("finalized backend contract must be outside sealed source")
    if output.exists() or output.is_symlink():
        raise LeptonJobError("finalized backend contract destination already exists")

    inventory_sha = _immutable_sha(
        Path(str(contract.source_inventory)), "sealed source inventory"
    )
    freeze_sha = _immutable_sha(
        Path(str(contract.python_freeze)), "Python environment freeze"
    )
    finalized = finalized_document(
        contract.document,
        source_inventory_sha256=inventory_sha,
        python_freeze_sha256=freeze_sha,
    )

    # Validate both external attestations before publishing any final contract.
    verified_contract = replace(
        contract,
        document=finalized,
        source_inventory_sha256=inventory_sha,
        python_freeze_sha256=freeze_sha,
    )
    verify_sealed_contract_source(verified_contract)
    verify_python_environment_freeze(verified_contract)

    project_root = Path(str(contract.project_root))
    if project_root.is_symlink() or not project_root.is_dir():
        raise LeptonJobError("Lepton project root is missing or unsafe")
    if expected_parent.exists():
        if expected_parent.is_symlink() or not expected_parent.is_dir():
            raise LeptonJobError("Lepton manifests directory is unsafe")
    else:
        expected_parent.mkdir(mode=0o750)
    _publish(output, finalized)
    loaded = load_contract(output, immutable=True)
    require_finalized_supply_chain(loaded)
    if (
        loaded.source_inventory_sha256 != inventory_sha
        or loaded.python_freeze_sha256 != freeze_sha
    ):
        raise LeptonJobError("published backend contract identities changed")
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    contract = finalize_contract(
        template_path=arguments.template,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "schema": contract.document["schema"],
                "contract_path": str(arguments.output.resolve()),
                "contract_sha256": contract.sha256,
                "source_inventory_sha256": contract.source_inventory_sha256,
                "python_freeze_sha256": contract.python_freeze_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeptonJobError as exc:
        print(f"Lepton backend finalization failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
