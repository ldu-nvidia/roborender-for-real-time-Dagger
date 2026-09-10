---
name: roborender-distributed-pipeline
description: Operate, audit, debug, or extend the canonical RoboRender behavioral-cloning and distributed RTX-to-H200 DAgger repository, including Isaac teacher collection, offline rendering, Image3F, student learning, checkpoint provenance, and closed-loop success validation.
---

# RoboRender distributed pipeline

Work from the canonical Git checkout and read `README.md` before acting. Then
read the README for the relevant stage. Treat versioned JSON contracts,
artifact manifests, and the live run's immutable config as authoritative; old
logs are evidence, not defaults.

## Preserve ownership

- RTX physical GPU1 owns persistent vectorized Isaac, the privileged teacher,
  Homie, episode lifecycle, and the only `env.step([B,26])` call.
- H200 GPUs 0–7 own Image3F and student inference.
- H200 services own learner chunks, optimization, checkpoint registry,
  coordination, evaluation ledger, and W&B training metrics.
- `pipeline/dagger/rtx` is source within the same repository, not a second
  independently maintained repo.

Never move the irreversible simulator step to H200, let a partial lane batch
step, replay an action after `CRASH_UNCERTAIN`, or publish an RTX audit seal as
a learner-complete chunk.

## Fixed scientific contracts

- Policy views: `head_color`, `hand_left`, `hand_right`, vertically stacked at
  108×192 per view.
- Online Image3F mapping: `ext1=head_color`, `ext2=hand_left`,
  `wrist=hand_right`; offline teacher mapping is left/right/head.
- Teacher label: 11-D privileged manipulation plus the exact 15-D Homie tail.
  Student predicts the first 11 dimensions.
- Online controls: Raven `h5_v4` depth and textured robot-on-black RGB at
  240×416, with one fixed three-view photorealistic anchor per episode.
- Observation assignment: 50% ray-traced and 50% Image3F per episode.
- Rollout/update unit: 81 sequential vector steps; B=28 yields 2,268 rows and
  28 chunks, followed by one current-window optimizer update.
- Replay: none across iterations or rounds.
- Round student probabilities: 25%, 50%, 75%, 100%; the teacher labels every
  state regardless of selected behavior.
- Validation: deterministic closed-loop goal-reached rate on the sealed fixed
  panel; no validation MSE and no backpropagation.

Read `pipeline/RAVEN_EQUIVALENCE_AUDIT.md` before claiming equivalence. Raven's
published percentages were single-view and her recorded behavior was
teacher-only; the three-view policy, beta schedule, B=28, v4 transport, and
fixed 256-case panel are explicit project extensions.

## Change workflow

1. Inspect Git status and preserve unrelated or runtime-owned state.
2. Locate the owning stage; do not create a parallel implementation.
3. Check checkpoint/renderer identity in `checkpoints/manifest.json` and
   `renderer/manifests/registry.v1.json` before loading or moving an artifact.
4. Keep source portable. Put absolute host asset paths only in explicit
   deployment manifests or ignored runtime configs.
5. Make protocol/scientific changes append-only and versioned. Preserve v1–v3
   compatibility when v4 imports their validators or codecs.
6. Run focused tests, Python compilation, shell syntax checks, JSON parsing,
   manifest hashes, secret/large-file scans, and `git diff --check`.
7. Require matching Git commit/tree identity on RTX and H200 before a live run.

## Safety gates

- Never start production from a fake endpoint or an unattested gateway PID.
- Require the exact run/config/checkpoint hashes and mode-0600 bearer token.
- Keep datasets, checkpoints, engines, containers, secrets, W&B state, and
  runtime journals out of Git.
- Do not stop, reset, or probe GPU0 from RTX production tooling.
- Do not restart Isaac at an 81-row boundary unless a real episode/lifecycle
  boundary or an explicit continuation contract permits it.
- Preserve a stopped run's last fully durable transition and exclude partial
  windows from learning.

## Verification commands

From the repository root:

```bash
python -m pytest -q \
  pipeline/tests/test_dagger_h200_services.py \
  pipeline/tests/test_dagger_render_student_service.py \
  pipeline/tests/test_dagger_round_saturation.py \
  pipeline/tests/test_dagger_stream_v4.py \
  pipeline/tests/test_raven_success_validation.py

python -m pytest -q pipeline/dagger/rtx/tests
bash -n pipeline/dagger/rtx/scripts/*.sh
python pipeline/bc_collection/validate_raven_bc_collection_contract.py
python pipeline/dagger/validate_raven_image3f_contract.py
```

Do not call a run production-ready from unit tests alone. Require a fresh native
preflight, live gateway identity, one all-lane prepare barrier, exactly-once
step/commit evidence, final chunk reconciliation, learner receipt, and the
scheduled success-rate evaluation.
