#!/usr/bin/env bash
# Pull one official base, record its immutable identity, build a diagnostic image.
set -Eeuo pipefail
(( $# == 1 )) || { echo 'usage: build_graphics_probe.sh 5.1.0|6.0.0' >&2; exit 2; }
case "$1" in 5.1.0|6.0.0) version=$1 ;; *) exit 2 ;; esac
[[ $(hostname) == 465d6b0-lcedt ]] || { echo 'Build this beta diagnostic on RTX beta.' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
base="nvcr.io/nvidia/isaac-sim:$version"
image="roborender-graphics-probe:isaac-$version-beta"
docker pull "$base"
digest=$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')
[[ $digest =~ ^nvcr.io/nvidia/isaac-sim@sha256:[0-9a-f]{64}$ ]] || exit 2
docker build --pull=false --build-arg "ISAAC_BASE=$digest" \
  -f "$ROOT/Dockerfile.graphics-probe" -t "$image" "$ROOT"
docker image inspect "$image" --format 'DIAGNOSTIC_IMAGE {{.Id}} {{json .RepoTags}}'
printf 'PINNED_BASE %s\n' "$digest"
echo 'Image built only. Run qualify_graphics.py; a camera PASS is not Raven qualification.'
