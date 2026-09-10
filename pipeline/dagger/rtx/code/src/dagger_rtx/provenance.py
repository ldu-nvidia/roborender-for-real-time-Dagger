from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import PINNED_RAVEN_COMMIT, PINNED_RAVEN_ROOT, ContractError

ROLE_TO_PROVENANCE = {
    "raven_videogen_trainer": "raven_videogen_trainer_sha256",
    "raven_teacher_homie_trainer": "raven_teacher_homie_trainer_sha256",
    "isaac_capture": "isaac_capture_sha256",
    "image_dagger_3view_config": "image_dagger_config_sha256",
    "privileged_teacher": "teacher_checkpoint_sha256",
    "homie_walk": "homie_walk_sha256",
    "homie_stand": "homie_stand_sha256",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_assets(
    manifest_path: Path,
    *,
    runtime_raven_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the pinned assets on the host or its certified container mount.

    Manifest paths always retain the canonical host identity.  The optional
    runtime root changes only the location from which bytes are read inside the
    read-only container mount.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("raven_root") != PINNED_RAVEN_ROOT or manifest.get("raven_commit") != PINNED_RAVEN_COMMIT:
        raise ContractError("asset manifest does not pin the certified Raven source")
    results = []
    for artifact in manifest.get("artifacts", []):
        canonical_path = Path(artifact["path"])
        path = canonical_path
        if runtime_raven_root is not None:
            try:
                relative = canonical_path.relative_to(PINNED_RAVEN_ROOT)
            except ValueError as exc:
                raise ContractError(
                    f"manifest artifact is outside pinned Raven root: {canonical_path}"
                ) from exc
            path = runtime_raven_root / relative
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = sha256_file(path) if exists and size == artifact["bytes"] else None
        ok = exists and size == artifact["bytes"] and digest == artifact["sha256"]
        results.append(
            {
                "role": artifact["role"],
                "path": str(canonical_path),
                "verified_runtime_path": str(path),
                "expected_bytes": artifact["bytes"],
                "actual_bytes": size,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": digest,
                "ok": ok,
            }
        )
    if not results or not all(row["ok"] for row in results):
        raise ContractError("one or more pinned Raven assets failed verification")
    roles = {item["role"]: item for item in manifest["artifacts"]}
    if not set(ROLE_TO_PROVENANCE).issubset(roles):
        raise ContractError("asset manifest lacks required provenance roles")
    static = {"raven_root": PINNED_RAVEN_ROOT, "raven_commit": PINNED_RAVEN_COMMIT}
    static.update({target: roles[role]["sha256"] for role, target in ROLE_TO_PROVENANCE.items()})
    return static, results
