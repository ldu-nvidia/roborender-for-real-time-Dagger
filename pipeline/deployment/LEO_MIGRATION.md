# Dedicated Leo RTX migration

Destination is `ldu@10.110.17.160`, hostname `leo-rtx-6k`. The explicitly pinned
RTX PRO6000 devices are physical **0 and 1**. Physical **2** is the A400 display
device and is excluded. Beta's old GPU0 restriction remains unchanged on Beta;
this is a new host-specific binding, not a global weakening of that guard.

The [host profile](profiles/leo_2rtx_migration.v1.json) retains the exact Beta
Isaac6 container and scientific callback code. Container, Raven assets, teacher
and Homie checkpoints, motion data, anchors and selected student checkpoints
are copied/verified separately from the code-only source release. Heavy H200
rendering weights, shared BC data and canonical training checkpoints stay on
the shared H200 volume. Never copy SSH private keys or cloud credentials.

Migration evidence lives under `runtime/migrations/beta-to-leo-20261013` on the
CPU and `/home/ldu/roborender_migration_beta_to_leo_20261013` on the workstations.
Initial migration SSH uses Beta as a LAN hop; that alone does not prove an
independent Leo-to-Cosmos production connection. A separate cloud enrollment
and actual endpoint check are needed before removing the Beta hop.

## Bounded native acceptance

Use the same source release on CPU and Leo. Replace RELEASE below with the
actual immutable directory; it must include the source manifest beside frozen.

```bash
/home/ldu/roborender/.venv/bin/python \
  /home/ldu/roborender/runtime/source-releases/RELEASE/frozen/pipeline/deployment/rtx_actor/qualify_leo.py \
  --source-root /home/ldu/roborender/runtime/source-releases/RELEASE/frozen \
  --source-manifest /home/ldu/roborender/runtime/source-releases/RELEASE/source-manifest.json \
  --physical-gpu 0 --num-envs 2 --mode capture-only
```

Repeat on physical GPU1. `--mode bc-collect` exercises 243 teacher-only vector
steps and verifies complete 81-frame BC chunk alignment, with no optimizer.
Authenticated `isolated-one-step`/`single-window` require a fresh run ID beginning
`rtx-leo-migration-`, a scoped token and the actual live H200 gateway PID.

Only the named diagnostic container is cleaned up. Base assets, source, other
containers, display GPU and checkpoints are preserved. No driver reset/change,
production learner or checkpoint promotion is performed by these commands.

## Certification boundary

File checksums and image-ID matching certify the copy, not a new runtime.
Capture certifies cameras/teacher inputs, not a complete training run. This
migration binding rejects Beta's host identity and the historical multi-window
qualification contract. The explicit Leo round/smoke schemas use the same
controller algorithm with GPU0+GPU1, six environments each, and four H200s.
Record each native acceptance result before describing Leo as ready; do not
reuse Beta's historical success percentages as a Leo result.

## Leo end-to-end smoke and access

Use `profiles/leo_site.direct.example.json` with an actual frozen source release.
`roborender.cross_gpu_dagger_smoke.leo.v1` runs fresh teacher-only BC collection,
50-step rendering, one full 16x81 BC update, paired ten-case success evaluation,
then three full 12x81 updates at each of teacher75/50/25/0 with paired success
evaluation after each round. The B5 evaluation panel remains the same versioned
scientific protocol; its Beta-era protocol name is not a routing dependency.
No smoke candidate is promoted and ten cases are not a full 256-case result.

```bash
python -m pipeline.deployment.beta_e2e_smoke \
  --site /absolute/path/leo-site.json \
  --run-id dagger-leo-smoke-UNIQUE_LOWERCASE_ID \
  --output /workspace/project/roborender/runtime/leo-smokes/UNIQUE_ID --execute
```

The shared implementation retains some `beta_*` module/field names for backward
compatibility. Active Leo paths are `runtime/leo-campaigns`, `leo-production`,
`leo-pods`, and `isaac6-cache-leo-gpu0/1`; old Beta run state is not reused.
The independent `roborender-leo-access.service` owns reverse port 22033 and uses
Leo's local cloud login, not copied Beta credentials. Its CPU SSH alias is
`roborender-leo-direct`. Both access host keys and GPU UUIDs are pinned.

Passing source tests alone does not admit this new topology: retain actual
native, transfer, renderer, learner, checkpoint and success receipts.
