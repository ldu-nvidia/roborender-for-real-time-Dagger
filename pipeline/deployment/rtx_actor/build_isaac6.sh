#!/usr/bin/env bash
# CPU-only image build: no GPU reservation, model changes, or runtime launch.
set -Eeuo pipefail
[[ $(hostname) == 465d6b0-lcedt ]] || { echo 'Build on RTX beta only.' >&2; exit 2; }
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
image=roborender-round-actor:isaac6-lab3-migration-candidate-20261009
docker build --pull=false --progress=plain -f "$root_dir/Dockerfile.isaac6" -t "$image" "$root_dir"
docker image inspect "$image" --format 'CANDIDATE_IMAGE {{.Id}} {{json .RepoTags}}'
echo 'Candidate built only. Native Raven qualification is still required.'
