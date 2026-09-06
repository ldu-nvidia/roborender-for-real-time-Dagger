#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TOOLS=$ROOT/containers/tools
VERSION=v0.33.0
ARCHIVE=buildkit-${VERSION}.linux-amd64.tar.gz
EXPECTED_SHA256=b6242896d343100808dcbe37565caf381e0a444a6a83d7255926bb1519248ead
URL=https://github.com/moby/buildkit/releases/download/${VERSION}/${ARCHIVE}

if [[ -x $TOOLS/bin/buildctl && -x $TOOLS/bin/buildkitd && -x $TOOLS/bin/buildkit-runc ]]; then
  "$TOOLS/bin/buildctl" --version
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 2; }
command -v tar >/dev/null || { echo "tar is required" >&2; exit 2; }

mkdir -p "$TOOLS"
temporary=$(mktemp -d "$ROOT/containers/.buildkit-download.XXXXXX")
cleanup() { rm -rf -- "$temporary"; }
trap cleanup EXIT

curl --fail --location --retry 3 --output "$temporary/$ARCHIVE" "$URL"
printf '%s  %s\n' "$EXPECTED_SHA256" "$temporary/$ARCHIVE" | sha256sum --check --strict
tar -xzf "$temporary/$ARCHIVE" -C "$TOOLS"

"$TOOLS/bin/buildctl" --version
"$TOOLS/bin/buildkitd" --version
