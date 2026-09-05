#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
VENDOR=$ROOT/vendor/gr00t-native-h200
PATCH=$VENDOR/changes.patch
MANIFEST=$VENDOR/manifest.json
GR00T_ROOT=${1:-$ROOT/GR00T-VisualSim2Real}

if (( $# > 1 )); then
    echo "usage: $0 [GR00T-CHECKOUT]" >&2
    exit 2
fi
[[ $GR00T_ROOT == /* && -d $GR00T_ROOT && ! -L $GR00T_ROOT ]] || {
    echo "GR00T checkout must be an absolute regular directory" >&2
    exit 2
}
[[ -f $PATCH && ! -L $PATCH && -f $MANIFEST && ! -L $MANIFEST ]] || {
    echo "native-H200 GR00T vendor bundle is incomplete" >&2
    exit 2
}

python3 - "$PATCH" "$MANIFEST" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

patch, manifest_path = map(Path, sys.argv[1:])
with manifest_path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
required = {
    "schema", "upstream_repository", "base_revision", "patch_file",
    "patch_sha256", "file_count", "files",
}
if set(manifest) != required:
    raise SystemExit("vendor manifest key set differs")
if manifest["schema"] != "fast_roborender.vendor.gr00t_native_h200_patch.v1":
    raise SystemExit("vendor manifest schema differs")
if manifest["patch_file"] != patch.name:
    raise SystemExit("vendor patch filename differs")
digest = hashlib.sha256(patch.read_bytes()).hexdigest()
if digest != manifest["patch_sha256"]:
    raise SystemExit("vendor patch SHA-256 differs")
rows = manifest["files"]
if not isinstance(rows, list) or manifest["file_count"] != len(rows):
    raise SystemExit("vendor file count differs")
paths = []
for row in rows:
    path = row.get("path") if isinstance(row, dict) else None
    parsed = PurePosixPath(path) if isinstance(path, str) else None
    if (
        parsed is None
        or parsed.is_absolute()
        or parsed.as_posix() != path
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or row.get("status") not in {"added", "modified", "deleted"}
    ):
        raise SystemExit("vendor file inventory is malformed")
    paths.append(path)
if paths != sorted(set(paths)):
    raise SystemExit("vendor file paths must be sorted and unique")
PY

BASE=$(python3 - "$MANIFEST" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    print(json.load(stream)["base_revision"])
PY
)
[[ $(git -C "$GR00T_ROOT" rev-parse HEAD) == "$BASE" ]] || {
    echo "GR00T checkout is not at the vendor bundle's base revision" >&2
    exit 2
}
[[ -z $(git -C "$GR00T_ROOT" status --porcelain=v1 --untracked-files=all) ]] || {
    echo "GR00T checkout must be clean before applying the native-H200 patch" >&2
    exit 2
}

git -C "$GR00T_ROOT" apply --check --binary --whitespace=error-all "$PATCH"
git -C "$GR00T_ROOT" apply --binary "$PATCH"

python3 - "$GR00T_ROOT" "$MANIFEST" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, manifest_path = map(Path, sys.argv[1:])
with manifest_path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
expected_codes = {"added": "??", "modified": " M", "deleted": " D"}
expected = {row["path"]: expected_codes[row["status"]] for row in manifest["files"]}
raw = subprocess.check_output(
    ["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
)
actual = {}
for item in raw.split(b"\0"):
    if not item:
        continue
    code = item[:2].decode("ascii")
    path = item[3:].decode("utf-8")
    actual[path] = code
if actual != expected:
    raise SystemExit("patched GR00T status inventory differs from manifest")
for row in manifest["files"]:
    target = root / row["path"]
    if row["status"] == "deleted":
        if target.exists() or target.is_symlink():
            raise SystemExit(f"deleted vendor path remains: {row['path']}")
        continue
    if target.is_symlink() or not target.is_file():
        raise SystemExit(f"patched vendor path is unsafe: {row['path']}")
    payload = target.read_bytes()
    if len(payload) != row["size_bytes"]:
        raise SystemExit(f"patched vendor file size differs: {row['path']}")
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise SystemExit(f"patched vendor file SHA-256 differs: {row['path']}")
print(
    "NATIVE_H200_GR00T_PATCH_APPLIED "
    f"base={manifest['base_revision']} files={manifest['file_count']}"
)
PY
