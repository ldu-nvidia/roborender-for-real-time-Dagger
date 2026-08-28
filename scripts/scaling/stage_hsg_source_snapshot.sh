#!/bin/bash
# Build, attest, and atomically publish one hydrated tracked-only HSG source snapshot.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FAST_REPO=${FAST_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
GR00T_REPO=${GR00T_REPO:-$FAST_REPO/GR00T-VisualSim2Real}
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-/mnt/data2/ldu/roborender_source_snapshots}
HSG_HOST=${HSG_HOST:-oci-hsg}
STAGE_ROOT=${STAGE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ldu/roborender_gb200_sm100}
REMOTE_OUTPUT_ROOT=${REMOTE_OUTPUT_ROOT:-$STAGE_ROOT/src_snapshots}
REMOTE_SEED_ROOT=${REMOTE_SEED_ROOT:-NONE}
ALLOW_UNTRACKED_FAST=${ALLOW_UNTRACKED_FAST:-}
ALLOW_UNTRACKED_GR00T=${ALLOW_UNTRACKED_GR00T:-}
LOCAL_ONLY=${LOCAL_ONLY:-0}

fatal() { echo "FATAL: $*" >&2; exit 2; }
case "$LOCAL_ONLY" in 0|1) ;; *) fatal "LOCAL_ONLY must be 0 or 1" ;; esac
[[ "$HSG_HOST" =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "HSG_HOST is malformed"
for path in "$LOCAL_OUTPUT_ROOT" "$STAGE_ROOT" "$REMOTE_OUTPUT_ROOT"; do
    [[ "$path" =~ ^/[A-Za-z0-9_./-]+$ ]] || fatal "unsafe absolute path: $path"
done
if [ "$REMOTE_SEED_ROOT" != NONE ]; then
    [[ "$REMOTE_SEED_ROOT" =~ ^/[A-Za-z0-9_./-]+$ ]] || \
        fatal "unsafe absolute remote seed path: $REMOTE_SEED_ROOT"
fi
for command in git install find chmod mv python3 realpath mktemp tar sha256sum \
               ssh stat awk wc cmp dd grep unlink zstd rsync; do
    command -v "$command" >/dev/null 2>&1 || fatal "required command is missing: $command"
done

PUBLISH_COMPLETE=0
REMOTE_PREPARED=0
LOCAL_TEMPORARY=
LOCAL_PUBLICATION_TEMP=
LOCAL_PUBLICATION=
LOCAL_DESTINATION=
LOCAL_ATTESTATION_TEMP=
LOCAL_ATTESTATION=
LOCAL_INVENTORY_TEMP=
LOCAL_INVENTORY=
LOCAL_LOCK=
LOCAL_TRANSPORT_ARCHIVE=
LOCAL_LOCK_OWNED=0
LOCAL_DESTINATION_OWNED=0
LOCAL_ATTESTATION_OWNED=0
LOCAL_INVENTORY_OWNED=0
REMOTE_TEMPORARY=
REMOTE_DESTINATION=
REMOTE_ATTESTATION_TEMP=
REMOTE_ATTESTATION=
REMOTE_INVENTORY_TEMP=
REMOTE_INVENTORY=
REMOTE_LOCK=
REMOTE_TRANSPORT_ARCHIVE=
INDEX_LISTS=()

cleanup_incomplete_publication() {
    local rc=$?
    trap - EXIT
    if [ "$PUBLISH_COMPLETE" -eq 0 ]; then
        if [ "$REMOTE_PREPARED" -eq 1 ]; then
            ssh -o BatchMode=yes -o ConnectTimeout=15 "$HSG_HOST" /bin/bash -s -- \
                "$REMOTE_OUTPUT_ROOT" "$REMOTE_TEMPORARY" "$REMOTE_DESTINATION" \
                "$REMOTE_ATTESTATION_TEMP" "$REMOTE_ATTESTATION" \
                "$REMOTE_INVENTORY_TEMP" "$REMOTE_INVENTORY" "$REMOTE_LOCK" \
                "$REMOTE_TRANSPORT_ARCHIVE" \
                <<'REMOTE_CLEANUP' || true
set -Eeuo pipefail
root=$1; temporary=$2; destination=$3
attestation_temporary=$4; attestation=$5; inventory_temporary=$6; inventory=$7
lock=$8; transport_archive=$9
case "$temporary" in "$root"/.fast_roborender_*_sealed.uploading.*) ;; *) exit 2 ;; esac
case "$destination" in "$root"/fast_roborender_*_sealed) ;; *) exit 2 ;; esac
if [ ! -e "$destination" ]; then
    for path in "$attestation_temporary" "$attestation" \
                "$inventory_temporary" "$inventory" "$transport_archive"; do
        [ ! -e "$path" ] || unlink "$path"
    done
    # A TERM can race the remote extraction process. Retry after restoring
    # owner write permission so files created just after the first chmod do
    # not strand the invocation's private temporary publication.
    for _attempt in 1 2 3 4 5; do
        if [ -d "$temporary" ] && [ ! -L "$temporary" ]; then
            chmod -R u+w "$temporary" || true
            find "$temporary" -xdev -depth -delete 2>/dev/null || true
        fi
        [ ! -e "$temporary" ] && break
        sleep 1
    done
    if [ -d "$lock" ] && [ ! -L "$lock" ]; then
        find "$lock" -xdev -depth -delete
    fi
fi
REMOTE_CLEANUP
        fi
        if [ "$LOCAL_ATTESTATION_OWNED" -eq 1 ] && [ -e "$LOCAL_ATTESTATION" ]; then
            unlink "$LOCAL_ATTESTATION"
        fi
        if [ "$LOCAL_INVENTORY_OWNED" -eq 1 ] && [ -e "$LOCAL_INVENTORY" ]; then
            unlink "$LOCAL_INVENTORY"
        fi
        for path in "${INDEX_LISTS[@]}"; do
            [ ! -e "$path" ] || unlink "$path"
        done
        if [ -n "$LOCAL_TRANSPORT_ARCHIVE" ]; then
            case "$LOCAL_TRANSPORT_ARCHIVE" in
                "$LOCAL_OUTPUT_ROOT"/.fast_roborender_*_sealed.transport.*) \
                    [ ! -e "$LOCAL_TRANSPORT_ARCHIVE" ] || \
                        unlink "$LOCAL_TRANSPORT_ARCHIVE" ;;
            esac
        fi
        for directory in "$LOCAL_PUBLICATION_TEMP"; do
            [ -z "$directory" ] && continue
            case "$directory" in
                "$LOCAL_OUTPUT_ROOT"/.fast_roborender_*_sealed.publication.staging.*) ;;
                *) continue ;;
            esac
            if [ -d "$directory" ] && [ ! -L "$directory" ]; then
                chmod -R u+w "$directory"
                find "$directory" -xdev -depth -delete
            fi
        done
        if [ "$LOCAL_DESTINATION_OWNED" -eq 1 ] && \
           [ -d "$LOCAL_PUBLICATION" ] && [ ! -L "$LOCAL_PUBLICATION" ]; then
            chmod -R u+w "$LOCAL_PUBLICATION"
            find "$LOCAL_PUBLICATION" -xdev -depth -delete
        fi
        if [ "$LOCAL_LOCK_OWNED" -eq 1 ] && \
           [ -d "$LOCAL_LOCK" ] && [ ! -L "$LOCAL_LOCK" ]; then
            find "$LOCAL_LOCK" -xdev -depth -delete
        fi
    fi
    exit "$rc"
}
trap cleanup_incomplete_publication EXIT

FAST_REPO=$(realpath -e "$FAST_REPO")
GR00T_REPO=$(realpath -e "$GR00T_REPO")
[ "$GR00T_REPO" = "$FAST_REPO/GR00T-VisualSim2Real" ] || \
    fatal "GR00T_REPO must be the exact nested GR00T checkout"

assert_untracked_allowlist() {
    local repo=$1 csv=$2
    python3 - "$repo" "$csv" <<'PY'
import subprocess
import sys

repo, csv = sys.argv[1:]
allowed = {".source_revision"}
allowed.update(path for path in csv.split(",") if path)
try:
    output = subprocess.run(
        ["git", "-C", repo, "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
except subprocess.CalledProcessError as exc:
    raise SystemExit(f"FATAL: could not enumerate untracked source in {repo}") from exc
unexpected = [
    raw.decode("utf-8", "surrogateescape")
    for raw in output.split(b"\0")
    if raw and raw.decode("utf-8", "surrogateescape") not in allowed
]
if unexpected:
    raise SystemExit(
        "FATAL: nonignored untracked source is not explicitly allowed: "
        + f"{repo}/{unexpected[0]}"
    )
PY
}

assert_repo_state() {
    local repo=$1 expected_head=$2 expected_index=$3 allow_untracked=$4
    [ "$(git -C "$repo" rev-parse --verify HEAD^{commit})" = "$expected_head" ] || \
        fatal "repository HEAD changed while staging: $repo"
    [ "$(git -C "$repo" write-tree)" = "$expected_index" ] || \
        fatal "repository index changed while staging: $repo"
    git -C "$repo" diff --quiet --no-ext-diff --ignore-submodules -- || \
        fatal "tracked working-tree changes are not releasable: $repo"
    git -C "$repo" diff --cached --quiet --no-ext-diff --ignore-submodules -- || \
        fatal "staged changes are not releasable: $repo"
    assert_untracked_allowlist "$repo" "$allow_untracked"
}

for repo in "$FAST_REPO" "$GR00T_REPO"; do
    git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
        fatal "not a Git working tree: $repo"
done
FAST_REVISION=$(git -C "$FAST_REPO" rev-parse --verify HEAD^{commit})
GR00T_REVISION=$(git -C "$GR00T_REPO" rev-parse --verify HEAD^{commit})
FAST_INDEX_TREE=$(git -C "$FAST_REPO" write-tree)
GR00T_INDEX_TREE=$(git -C "$GR00T_REPO" write-tree)
[[ "$FAST_REVISION:$GR00T_REVISION:$FAST_INDEX_TREE:$GR00T_INDEX_TREE" =~ \
^[0-9a-f]{40}:[0-9a-f]{40}:[0-9a-f]{40}:[0-9a-f]{40}$ ]] || \
    fatal "source Git identities are malformed"
assert_repo_state "$FAST_REPO" "$FAST_REVISION" "$FAST_INDEX_TREE" "$ALLOW_UNTRACKED_FAST"
assert_repo_state "$GR00T_REPO" "$GR00T_REVISION" "$GR00T_INDEX_TREE" "$ALLOW_UNTRACKED_GR00T"

mkdir -p "$LOCAL_OUTPUT_ROOT"
[ -d "$LOCAL_OUTPUT_ROOT" ] && [ ! -L "$LOCAL_OUTPUT_ROOT" ] || \
    fatal "local source snapshot root is unsafe"
LOCAL_OUTPUT_ROOT=$(realpath -e "$LOCAL_OUTPUT_ROOT")
SNAPSHOT_NAME=fast_roborender_${FAST_REVISION:0:7}_${GR00T_REVISION:0:7}_sealed
LOCAL_LOCK=$LOCAL_OUTPUT_ROOT/.${SNAPSHOT_NAME}.publish.lock
LOCAL_PUBLICATION=$LOCAL_OUTPUT_ROOT/${SNAPSHOT_NAME}.publication
LOCAL_DESTINATION=$LOCAL_PUBLICATION/$SNAPSHOT_NAME
LOCAL_ATTESTATION=$LOCAL_PUBLICATION/${SNAPSHOT_NAME}.snapshot_attestation.json
LOCAL_INVENTORY=$LOCAL_PUBLICATION/${SNAPSHOT_NAME}.inventory.json
mkdir "$LOCAL_LOCK" 2>/dev/null || fatal "another local publisher owns $SNAPSHOT_NAME"
LOCAL_LOCK_OWNED=1
[ ! -e "$LOCAL_PUBLICATION" ] || \
    fatal "refusing to replace an existing local snapshot publication"
LOCAL_PUBLICATION_TEMP=$(mktemp -d \
    "$LOCAL_OUTPUT_ROOT/.${SNAPSHOT_NAME}.publication.staging.XXXXXX")
LOCAL_TEMPORARY=$LOCAL_PUBLICATION_TEMP/$SNAPSHOT_NAME
mkdir "$LOCAL_TEMPORARY"
LOCAL_ATTESTATION_TEMP=$LOCAL_PUBLICATION_TEMP/${SNAPSHOT_NAME}.snapshot_attestation.json
LOCAL_INVENTORY_TEMP=$LOCAL_PUBLICATION_TEMP/${SNAPSHOT_NAME}.inventory.json

LFS_FILE_COUNT=0
LFS_TOTAL_BYTES=0
copy_tracked_tree() {
    local repo=$1 destination=$2 entry metadata path mode rest object stage target_mode
    local source target blob_size pointer_version pointer_oid_line pointer_size_line
    local pointer_oid pointer_size target_sha target_size index_list
    mkdir -p "$destination"
    index_list=$(mktemp "$LOCAL_OUTPUT_ROOT/.tracked-index.XXXXXX")
    INDEX_LISTS+=("$index_list")
    git -C "$repo" ls-files --stage -z > "$index_list" || \
        fatal "could not enumerate the stage-zero tracked tree: $repo"
    while IFS= read -r -d '' entry; do
        metadata=${entry%%$'\t'*}
        path=${entry#*$'\t'}
        [ "$path" != .source_revision ] && \
        [ "$path" != .source_snapshot_manifest.json ] || \
            fatal "generated provenance file is tracked in $repo: $path"
        case "$path" in
            __pycache__/*|*/__pycache__/*|*.pyc|*.pyo)
                fatal "tracked executable bytecode is forbidden: $repo/$path" ;;
        esac
        mode=${metadata%% *}
        rest=${metadata#* }
        object=${rest%% *}
        stage=${rest##* }
        [ "$stage" = 0 ] || fatal "non-stage-zero index entry: $repo/$path"
        case "$mode" in
            100644) target_mode=0644 ;;
            100755) target_mode=0755 ;;
            120000) fatal "tracked source symlinks are forbidden: $repo/$path" ;;
            160000) fatal "nested Git links are forbidden: $repo/$path" ;;
            *) fatal "unsupported tracked mode $mode for $repo/$path" ;;
        esac
        source=$repo/$path
        [ -f "$source" ] && [ ! -L "$source" ] || \
            fatal "tracked working file is missing or unsafe: $source"
        target=$destination/$path
        mkdir -p "$(dirname "$target")"
        blob_size=$(git -C "$repo" cat-file -s "$object")
        pointer_version=
        pointer_oid_line=
        pointer_size_line=
        if [ "$blob_size" -le 1024 ]; then
            {
                IFS= read -r pointer_version || true
                IFS= read -r pointer_oid_line || true
                IFS= read -r pointer_size_line || true
                if [ "$pointer_version" = \
                     "version https://git-lfs.github.com/spec/v1" ]; then
                    IFS= read -r extra_line && \
                        fatal "LFS pointer has unexpected fields: $repo/$path"
                fi
            } < <(git -C "$repo" cat-file blob "$object")
        fi
        if [ "$pointer_version" = "version https://git-lfs.github.com/spec/v1" ]; then
            [[ "$pointer_oid_line" =~ ^oid\ sha256:([0-9a-f]{64})$ ]] || \
                fatal "malformed LFS object ID: $repo/$path"
            pointer_oid=${BASH_REMATCH[1]}
            [[ "$pointer_size_line" =~ ^size\ ([0-9]+)$ ]] || \
                fatal "malformed LFS object size: $repo/$path"
            pointer_size=${BASH_REMATCH[1]}
            [ "$(stat -c %s "$source")" = "$pointer_size" ] || \
                fatal "LFS working file size does not match committed pointer: $repo/$path"
            [ "$(sha256sum "$source" | awk '{print $1}')" = "$pointer_oid" ] || \
                fatal "LFS working file hash does not match committed pointer: $repo/$path"
            install -m "$target_mode" "$source" "$target"
            target_size=$(stat -c %s "$target")
            target_sha=$(sha256sum "$target" | awk '{print $1}')
            [ "$target_size:$target_sha" = "$pointer_size:$pointer_oid" ] || \
                fatal "copied LFS object failed verification: $repo/$path"
            LFS_FILE_COUNT=$((LFS_FILE_COUNT + 1))
            LFS_TOTAL_BYTES=$((LFS_TOTAL_BYTES + pointer_size))
        else
            git -C "$repo" cat-file blob "$object" > "$target"
            chmod "$target_mode" "$target"
            [ "$(sha256sum "$source" | awk '{print $1}')" = \
              "$(sha256sum "$target" | awk '{print $1}')" ] || \
                fatal "working file bytes differ from committed Git blob: $repo/$path"
        fi
    done < "$index_list"
    unlink "$index_list"
}

copy_tracked_tree "$FAST_REPO" "$LOCAL_TEMPORARY"
mkdir -p "$LOCAL_TEMPORARY/GR00T-VisualSim2Real"
copy_tracked_tree "$GR00T_REPO" "$LOCAL_TEMPORARY/GR00T-VisualSim2Real"
printf '%s\n' "$FAST_REVISION" > "$LOCAL_TEMPORARY/.source_revision"
printf '%s\n' "$GR00T_REVISION" > \
    "$LOCAL_TEMPORARY/GR00T-VisualSim2Real/.source_revision"
python3 - "$LOCAL_TEMPORARY/.source_snapshot_manifest.json" "$SNAPSHOT_NAME" \
    "$FAST_REVISION" "$GR00T_REVISION" "$FAST_INDEX_TREE" "$GR00T_INDEX_TREE" \
    "$LFS_FILE_COUNT" "$LFS_TOTAL_BYTES" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "fast_roborender.hsg_source_snapshot.v2",
    "status": "sealed_tracked_hydrated_sources",
    "snapshot_name": sys.argv[2],
    "fast_roborender_revision": sys.argv[3],
    "gr00t_revision": sys.argv[4],
    "fast_roborender_index_tree": sys.argv[5],
    "gr00t_index_tree": sys.argv[6],
    "export_method": "stage0-git-blobs-plus-verified-lfs-v2",
    "lfs_file_count": int(sys.argv[7]),
    "lfs_total_bytes": int(sys.argv[8]),
    "tracked_only": True,
    "lfs_pointer_oids_verified": True,
    "symlinks_rejected": True,
    "bytecode_rejected": True,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

# Close index/worktree TOCTOU after every source byte has been copied.
assert_repo_state "$FAST_REPO" "$FAST_REVISION" "$FAST_INDEX_TREE" "$ALLOW_UNTRACKED_FAST"
assert_repo_state "$GR00T_REPO" "$GR00T_REVISION" "$GR00T_INDEX_TREE" "$ALLOW_UNTRACKED_GR00T"
FORBIDDEN=$(find "$LOCAL_TEMPORARY" \
    \( -type l -o -type b -o -type c -o -type p -o -type s \
       -o -type d -name __pycache__ \
       -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) -print -quit)
[ -z "$FORBIDDEN" ] || fatal "forbidden staged source entry: $FORBIDDEN"
find "$LOCAL_TEMPORARY" -type f -perm /111 -exec chmod 0555 {} +
find "$LOCAL_TEMPORARY" -type f ! -perm /111 -exec chmod 0444 {} +
find "$LOCAL_TEMPORARY" -type d -exec chmod 0555 {} +

PYTHONDONTWRITEBYTECODE=1 python3 - "$LOCAL_TEMPORARY" "$LOCAL_INVENTORY_TEMP" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
records = []
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise SystemExit(f"unsafe inventory entry: {relative}")
    if stat.S_ISDIR(mode):
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append({
        "path": relative,
        "mode": oct(mode & 0o777),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
payload = {
    "schema": "fast_roborender.hsg_source_inventory.v1",
    "file_count": len(records),
    "total_file_bytes": sum(record["bytes"] for record in records),
    "files": records,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
chmod 0444 "$LOCAL_INVENTORY_TEMP"

FAST_TREE_SHA256=$(PYTHONDONTWRITEBYTECODE=1 python3 \
    "$LOCAL_TEMPORARY/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash \
    --root "$LOCAL_TEMPORARY")
GR00T_TREE_SHA256=$(PYTHONDONTWRITEBYTECODE=1 python3 \
    "$LOCAL_TEMPORARY/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash \
    --root "$LOCAL_TEMPORARY/GR00T-VisualSim2Real")
INVENTORY_SHA256=$(sha256sum "$LOCAL_INVENTORY_TEMP" | awk '{print $1}')
ARCHIVE_SHA256=$(tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -cf - -C "$LOCAL_TEMPORARY" . | sha256sum | awk '{print $1}')
read -r FILE_COUNT TOTAL_FILE_BYTES < <(python3 - "$LOCAL_INVENTORY_TEMP" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["file_count"], payload["total_file_bytes"])
PY
)
python3 - "$LOCAL_ATTESTATION_TEMP" "$SNAPSHOT_NAME" "$FAST_REVISION" \
    "$GR00T_REVISION" "$FAST_INDEX_TREE" "$GR00T_INDEX_TREE" \
    "$FAST_TREE_SHA256" "$GR00T_TREE_SHA256" "$INVENTORY_SHA256" \
    "$ARCHIVE_SHA256" "$FILE_COUNT" "$TOTAL_FILE_BYTES" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema": "fast_roborender.hsg_source_snapshot_attestation.v2",
    "status": "PASS",
    "snapshot_name": sys.argv[2],
    "fast_roborender_revision": sys.argv[3],
    "gr00t_revision": sys.argv[4],
    "fast_roborender_index_tree": sys.argv[5],
    "gr00t_index_tree": sys.argv[6],
    "fast_roborender_source_tree_sha256": sys.argv[7],
    "gr00t_source_tree_sha256": sys.argv[8],
    "inventory_sha256": sys.argv[9],
    "deterministic_archive_sha256": sys.argv[10],
    "file_count": int(sys.argv[11]),
    "total_file_bytes": int(sys.argv[12]),
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
chmod 0444 "$LOCAL_ATTESTATION_TEMP" "$LOCAL_INVENTORY_TEMP"
chmod 0555 "$LOCAL_PUBLICATION_TEMP"
mv "$LOCAL_PUBLICATION_TEMP" "$LOCAL_PUBLICATION"
LOCAL_PUBLICATION_TEMP=
LOCAL_DESTINATION_OWNED=1

if [ "$LOCAL_ONLY" -eq 1 ]; then
    printf 'ROOT=%s\nGR00T_ROOT=%s\nATTESTATION=%s\nINVENTORY=%s\n' \
        "$LOCAL_DESTINATION" "$LOCAL_DESTINATION/GR00T-VisualSim2Real" \
        "$LOCAL_ATTESTATION" "$LOCAL_INVENTORY"
    find "$LOCAL_LOCK" -xdev -depth -delete
    LOCAL_LOCK_OWNED=0
    PUBLISH_COMPLETE=1
    exit 0
fi

REMOTE_DESTINATION=$REMOTE_OUTPUT_ROOT/$SNAPSHOT_NAME
REMOTE_TEMPORARY=$REMOTE_OUTPUT_ROOT/.${SNAPSHOT_NAME}.uploading.$$
REMOTE_ATTESTATION=$REMOTE_OUTPUT_ROOT/${SNAPSHOT_NAME}.snapshot_attestation.json
REMOTE_ATTESTATION_TEMP=$REMOTE_OUTPUT_ROOT/.${SNAPSHOT_NAME}.attestation.uploading.$$.json
REMOTE_INVENTORY=$REMOTE_OUTPUT_ROOT/${SNAPSHOT_NAME}.inventory.json
REMOTE_INVENTORY_TEMP=$REMOTE_OUTPUT_ROOT/.${SNAPSHOT_NAME}.inventory.uploading.$$.json
REMOTE_LOCK=$REMOTE_OUTPUT_ROOT/.${SNAPSHOT_NAME}.publish.lock
REMOTE_TRANSPORT_ARCHIVE=$REMOTE_OUTPUT_ROOT/.${SNAPSHOT_NAME}.transport.uploading.$$.tar.zst
ssh -o BatchMode=yes "$HSG_HOST" /bin/bash -s -- "$STAGE_ROOT" "$REMOTE_OUTPUT_ROOT" \
    "$REMOTE_DESTINATION" "$REMOTE_TEMPORARY" "$REMOTE_ATTESTATION" \
    "$REMOTE_ATTESTATION_TEMP" "$REMOTE_INVENTORY" "$REMOTE_INVENTORY_TEMP" \
    "$REMOTE_LOCK" "$REMOTE_SEED_ROOT" "$REMOTE_TRANSPORT_ARCHIVE" <<'REMOTE_PREPARE'
set -Eeuo pipefail
stage_root=$1; output_root=$2; destination=$3; temporary=$4
attestation=$5; attestation_temporary=$6; inventory=$7; inventory_temporary=$8
lock=$9; seed=${10}; transport_archive=${11}
lock_owned=0
cleanup_prepare() {
    local rc=$?
    trap - EXIT
    if [ "$rc" -ne 0 ] && [ "$lock_owned" -eq 1 ] && \
       [ -d "$lock" ] && [ ! -L "$lock" ]; then
        find "$lock" -xdev -depth -delete
    fi
    exit "$rc"
}
trap cleanup_prepare EXIT
[ -d "$stage_root" ] && [ ! -L "$stage_root" ]
[ -x /usr/bin/zstd ]
[ -x /usr/bin/rsync ]
mkdir -p "$output_root"
[ -d "$output_root" ] && [ ! -L "$output_root" ]
[ "$(realpath -e "$output_root")" = "$output_root" ]
[ "$(dirname "$output_root")" = "$stage_root" ]
[ "$(basename "$output_root")" = src_snapshots ]
mkdir "$lock"
lock_owned=1
for path in "$destination" "$temporary" "$attestation" "$attestation_temporary" \
            "$inventory" "$inventory_temporary" "$transport_archive"; do
    [ ! -e "$path" ]
done
mkdir "$temporary"
if [ "$seed" != NONE ]; then
    [ -d "$seed" ] && [ ! -L "$seed" ]
    canonical_seed=$(realpath -e "$seed")
    [ "$canonical_seed" = "$seed" ]
    case "$canonical_seed" in "$output_root"/fast_roborender_*_sealed) ;; *) exit 2 ;; esac
    [ "$canonical_seed" != "$destination" ] && [ "$canonical_seed" != "$temporary" ]
    cp -a --reflink=auto "$canonical_seed/." "$temporary/"
    chmod -R u+w "$temporary"
fi
REMOTE_PREPARE
REMOTE_PREPARED=1
if [ "$REMOTE_SEED_ROOT" = NONE ]; then
    LOCAL_TRANSPORT_ARCHIVE=$(mktemp \
        "$LOCAL_OUTPUT_ROOT/.${SNAPSHOT_NAME}.transport.XXXXXX")
    tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
        -cf - -C "$LOCAL_DESTINATION" . | zstd -q -1 -T0 -c \
        > "$LOCAL_TRANSPORT_ARCHIVE"
    [ "$(zstd -q -d -c "$LOCAL_TRANSPORT_ARCHIVE" | sha256sum | awk '{print $1}')" = \
      "$ARCHIVE_SHA256" ] || fatal "local transport archive identity mismatch"
    TRANSPORT_SHA256=$(sha256sum "$LOCAL_TRANSPORT_ARCHIVE" | awk '{print $1}')
    chmod 0444 "$LOCAL_TRANSPORT_ARCHIVE"
    rsync -a --checksum --no-owner --no-group \
        --rsync-path=/usr/bin/rsync \
        -e "ssh -o BatchMode=yes" \
        "$LOCAL_TRANSPORT_ARCHIVE" "$HSG_HOST:$REMOTE_TRANSPORT_ARCHIVE"
    ssh -o BatchMode=yes "$HSG_HOST" /bin/bash -s -- \
        "$REMOTE_TRANSPORT_ARCHIVE" "$REMOTE_TEMPORARY" \
        "$TRANSPORT_SHA256" "$ARCHIVE_SHA256" \
        < "$LOCAL_DESTINATION/scripts/scaling/receive_source_snapshot_archive.sh"
    unlink "$LOCAL_TRANSPORT_ARCHIVE"
    LOCAL_TRANSPORT_ARCHIVE=
else
    rsync -a --checksum --delete --no-owner --no-group \
        --rsync-path=/usr/bin/rsync \
        -e "ssh -o BatchMode=yes" \
        "$LOCAL_DESTINATION/" "$HSG_HOST:$REMOTE_TEMPORARY/"
fi
ssh -o BatchMode=yes "$HSG_HOST" \
    "/bin/bash -c 'dd of=\"$REMOTE_ATTESTATION_TEMP\" status=none'" \
    < "$LOCAL_ATTESTATION"
ssh -o BatchMode=yes "$HSG_HOST" \
    "/bin/bash -c 'dd of=\"$REMOTE_INVENTORY_TEMP\" status=none'" \
    < "$LOCAL_INVENTORY"

ssh -o BatchMode=yes "$HSG_HOST" /bin/bash -s -- "$REMOTE_TEMPORARY" \
    "$REMOTE_DESTINATION" "$REMOTE_ATTESTATION_TEMP" "$REMOTE_ATTESTATION" \
    "$REMOTE_INVENTORY_TEMP" "$REMOTE_INVENTORY" "$FAST_REVISION" "$GR00T_REVISION" \
    "$FAST_TREE_SHA256" "$GR00T_TREE_SHA256" "$INVENTORY_SHA256" "$ARCHIVE_SHA256" \
    "$SNAPSHOT_NAME" "$REMOTE_LOCK" <<'REMOTE_VERIFY'
set -Eeuo pipefail
temporary=$1; destination=$2; attestation_temporary=$3; attestation=$4
inventory_temporary=$5; inventory=$6; fast_revision=$7; gr00t_revision=$8
fast_tree_expected=$9; gr00t_tree_expected=${10}; inventory_expected=${11}
archive_expected=${12}; snapshot_name=${13}
lock=${14}
[ -d "$temporary" ] && [ ! -L "$temporary" ]
[ -f "$attestation_temporary" ] && [ ! -L "$attestation_temporary" ]
[ -f "$inventory_temporary" ] && [ ! -L "$inventory_temporary" ]
[ "$(cat "$temporary/.source_revision")" = "$fast_revision" ]
[ "$(cat "$temporary/GR00T-VisualSim2Real/.source_revision")" = "$gr00t_revision" ]
[ -z "$(find "$temporary" \( -type l -o -type b -o -type c -o -type p -o -type s \
    -o -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
    -print -quit)" ]
[ -z "$(find "$temporary" -perm /222 -print -quit)" ]
fast_tree_actual=$(PYTHONDONTWRITEBYTECODE=1 python3 \
    "$temporary/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash --root "$temporary")
gr00t_tree_actual=$(PYTHONDONTWRITEBYTECODE=1 python3 \
    "$temporary/scripts/scaling/hsg_gb200_smoke_gate.py" tree-hash \
    --root "$temporary/GR00T-VisualSim2Real")
[ "$fast_tree_actual" = "$fast_tree_expected" ]
[ "$gr00t_tree_actual" = "$gr00t_tree_expected" ]
[ "$(sha256sum "$inventory_temporary" | awk '{print $1}')" = "$inventory_expected" ]
python3 - "$attestation_temporary" "$snapshot_name" "$fast_revision" \
    "$gr00t_revision" "$fast_tree_expected" "$gr00t_tree_expected" \
    "$inventory_expected" "$archive_expected" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema": "fast_roborender.hsg_source_snapshot_attestation.v2",
    "status": "PASS",
    "snapshot_name": sys.argv[2],
    "fast_roborender_revision": sys.argv[3],
    "gr00t_revision": sys.argv[4],
    "fast_roborender_source_tree_sha256": sys.argv[5],
    "gr00t_source_tree_sha256": sys.argv[6],
    "inventory_sha256": sys.argv[7],
    "deterministic_archive_sha256": sys.argv[8],
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"snapshot attestation {key} mismatch")
PY
chmod 0444 "$attestation_temporary" "$inventory_temporary"
mv "$inventory_temporary" "$inventory"
mv "$attestation_temporary" "$attestation"
mv "$temporary" "$destination"
find "$lock" -xdev -depth -delete
printf 'REMOTE_ROOT=%s\nREMOTE_ATTESTATION=%s\nREMOTE_INVENTORY=%s\n' \
    "$destination" "$attestation" "$inventory"
REMOTE_VERIFY
find "$LOCAL_LOCK" -xdev -depth -delete
LOCAL_LOCK_OWNED=0
PUBLISH_COMPLETE=1

printf 'LOCAL_ROOT=%s\nLOCAL_ATTESTATION=%s\nLOCAL_INVENTORY=%s\n' \
    "$LOCAL_DESTINATION" "$LOCAL_ATTESTATION" "$LOCAL_INVENTORY"
