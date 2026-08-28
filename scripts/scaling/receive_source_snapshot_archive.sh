#!/bin/bash
# Verify one canonical source archive byte-for-byte, then extract it.
set -Eeuo pipefail

[ "$#" -eq 4 ] || {
    echo "FATAL: expected ARCHIVE DESTINATION TRANSPORT_SHA256 ARCHIVE_SHA256" >&2
    exit 2
}
archive=$1
destination=$2
transport_expected=$3
archive_expected=$4

[[ "$transport_expected:$archive_expected" =~ ^[0-9a-f]{64}:[0-9a-f]{64}$ ]] || {
    echo "FATAL: malformed source archive digest" >&2
    exit 2
}
[ -f "$archive" ] && [ ! -L "$archive" ] || {
    echo "FATAL: source transport archive is missing or unsafe" >&2
    exit 2
}
[ -d "$destination" ] && [ ! -L "$destination" ] || {
    echo "FATAL: source extraction destination is missing or unsafe" >&2
    exit 2
}
[ -z "$(find "$destination" -mindepth 1 -print -quit)" ] || {
    echo "FATAL: source extraction destination is not empty" >&2
    exit 2
}
[ -x /usr/bin/zstd ] && [ -x /usr/bin/tar ] || {
    echo "FATAL: source archive runtime is incomplete" >&2
    exit 2
}

transport_actual=$(sha256sum "$archive" | awk '{print $1}')
[ "$transport_actual" = "$transport_expected" ] || {
    echo "FATAL: compressed source transport SHA-256 mismatch" >&2
    exit 2
}
archive_actual=$(/usr/bin/zstd -q -d -c "$archive" | sha256sum | awk '{print $1}')
[ "$archive_actual" = "$archive_expected" ] || {
    echo "FATAL: canonical source archive SHA-256 mismatch" >&2
    exit 2
}

# The canonical archive bytes were produced once on the publisher.  Extraction
# intentionally does not ask this host's tar version to reproduce those bytes.
/usr/bin/zstd -q -d -c "$archive" | /usr/bin/tar -xf - -C "$destination"
unlink "$archive"
