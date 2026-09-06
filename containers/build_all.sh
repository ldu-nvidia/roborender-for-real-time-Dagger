#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
CONTAINERS=$ROOT/containers
TOOLS=$CONTAINERS/tools/bin
IMAGES=$CONTAINERS/images
LOGS=$CONTAINERS/logs
STATE=$CONTAINERS/state
SOCKET=$STATE/buildkitd.sock
QROOT=${ROBORENDER_QUALIFIED_ACTOR_RUNTIME:-$ROOT/runtime/isaac_actor/qualified_h200_v1}

fatal() { echo "container build: $*" >&2; exit 2; }

roles=("${@:-collector renderer learner}")
if (( $# == 0 )); then
  roles=(collector renderer learner)
fi
for role in "${roles[@]}"; do
  case "$role" in collector|renderer|learner) ;; *) fatal "unknown role: $role" ;; esac
done

[[ -d $ROOT/pipeline && -d $ROOT/renderer ]] || fatal "canonical project layout is incomplete"
[[ -d $QROOT/runtime/isaacsim-5.1 ]] || fatal "qualified Isaac 5.1 runtime is missing: $QROOT"
[[ -d $ROOT/renderer/online_image3f/runtime && -d $ROOT/renderer/envs ]] || \
  fatal "renderer runtime/environment is missing"

if [[ ! -x $TOOLS/buildctl || ! -x $TOOLS/buildkitd ]]; then
  "$CONTAINERS/bootstrap_buildkit.sh"
fi

mkdir -p "$IMAGES" "$LOGS" "$STATE/root"
rm -f -- "$SOCKET"
PATH=$TOOLS:/usr/sbin:/usr/bin:/sbin:/bin \
  "$TOOLS/buildkitd" \
  --root "$STATE/root" \
  --addr "unix://$SOCKET" \
  --oci-worker=true \
  --containerd-worker=false \
  --oci-worker-snapshotter=native \
  >"$LOGS/buildkitd.log" 2>&1 &
daemon_pid=$!
cleanup() {
  if kill -0 "$daemon_pid" 2>/dev/null; then
    kill "$daemon_pid"
    wait "$daemon_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=false
for _ in $(seq 1 60); do
  if "$TOOLS/buildctl" --addr "unix://$SOCKET" debug workers >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
[[ $ready == true ]] || {
  tail -n 80 "$LOGS/buildkitd.log" >&2 || true
  fatal "project-local BuildKit daemon did not become ready"
}

build_role() {
  local role=$1
  local output=$IMAGES/roborender-${role}-h200.oci.tar
  local partial=$output.partial
  local log=$LOGS/${role}.build.log
  local -a contexts=()
  [[ ! -e $output ]] || fatal "refusing to overwrite existing image archive: $output"
  [[ ! -e $partial ]] || fatal "remove the incomplete archive before retrying: $partial"

  case "$role" in
    collector)
      contexts+=(
        --local "qualified_actor_runtime=$QROOT/runtime"
        --local "qualified_actor_packages=$QROOT/packages"
        --opt context:qualified_actor_runtime=local:qualified_actor_runtime
        --opt context:qualified_actor_packages=local:qualified_actor_packages
      )
      ;;
    renderer)
      contexts+=(
        --local "renderer_env=$ROOT/renderer/envs"
        --local "renderer_runtime=$ROOT/renderer/online_image3f/runtime"
        --opt context:renderer_env=local:renderer_env
        --opt context:renderer_runtime=local:renderer_runtime
      )
      ;;
  esac

  echo "Building $role -> $output"
  "$TOOLS/buildctl" --addr "unix://$SOCKET" build \
    --frontend dockerfile.v0 \
    --local "context=$ROOT" \
    --local "dockerfile=$CONTAINERS/$role" \
    --opt filename=Dockerfile \
    --opt platform=linux/amd64 \
    "${contexts[@]}" \
    --output "type=oci,name=roborender/${role}:h200,dest=$partial" \
    --progress plain 2>&1 | tee "$log"
  [[ -s $partial ]] || fatal "$role build produced an empty archive"
  mv "$partial" "$output"
  sha256sum "$output" | tee "$output.sha256"
}

for role in "${roles[@]}"; do
  build_role "$role"
done

find "$IMAGES" -maxdepth 1 -type f -name '*.oci.tar' -print0 \
  | sort -z | xargs -0 sha256sum > "$IMAGES/SHA256SUMS"
echo "Built ${#roles[@]} OCI image archive(s) under $IMAGES"
