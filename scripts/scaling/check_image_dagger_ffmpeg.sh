#!/bin/bash
# Fail closed unless the telemetry encoder explicitly selected by the launcher
# is usable in the trainer runtime.  This is intentionally independent of PATH:
# the sealed ARM64 trainer keeps imageio-ffmpeg under its Python environment.
set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 2
}

[ -n "${IMAGE_DAGGER_FFMPEG:-}" ] || \
    fatal "IMAGE_DAGGER_FFMPEG is unset"
case "$IMAGE_DAGGER_FFMPEG" in
    /*) ;;
    *) fatal "IMAGE_DAGGER_FFMPEG must be an absolute path: $IMAGE_DAGGER_FFMPEG" ;;
esac
[ -f "$IMAGE_DAGGER_FFMPEG" ] || \
    fatal "IMAGE_DAGGER_FFMPEG is missing or not a regular file: $IMAGE_DAGGER_FFMPEG"
[ -x "$IMAGE_DAGGER_FFMPEG" ] || \
    fatal "IMAGE_DAGGER_FFMPEG is not executable: $IMAGE_DAGGER_FFMPEG"

# Executability alone does not catch a missing ELF interpreter or shared
# library.  Exercise the encoder without producing output before Isaac starts.
FFMPEG_VERSION=$(
    "$IMAGE_DAGGER_FFMPEG" -version 2>&1 | sed -n '1p'
) || fatal "IMAGE_DAGGER_FFMPEG failed its version probe: $IMAGE_DAGGER_FFMPEG"
[ -n "$FFMPEG_VERSION" ] || \
    fatal "IMAGE_DAGGER_FFMPEG returned an empty version probe"

# MP4 telemetry is encoded with H.264.  A runnable ffmpeg binary without the
# libx264 encoder would still fail only after an expensive rollout, so make the
# codec capability part of this launch-time contract as well.
FFMPEG_ENCODERS=$(
    "$IMAGE_DAGGER_FFMPEG" -hide_banner -encoders 2>&1
) || fatal "IMAGE_DAGGER_FFMPEG failed its encoder probe: $IMAGE_DAGGER_FFMPEG"
printf '%s\n' "$FFMPEG_ENCODERS" | \
    grep -Eq '(^|[[:space:]])libx264([[:space:]]|$)' || \
    fatal "IMAGE_DAGGER_FFMPEG does not advertise the required libx264 encoder: $IMAGE_DAGGER_FFMPEG"

printf 'IMAGE_DAGGER_FFMPEG=%s\n' "$IMAGE_DAGGER_FFMPEG"
printf 'IMAGE_DAGGER_FFMPEG_VERSION=%s\n' "$FFMPEG_VERSION"
printf 'IMAGE_DAGGER_FFMPEG_ENCODER=libx264\n'
