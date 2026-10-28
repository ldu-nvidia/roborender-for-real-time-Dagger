# Working repository and deployment entrypoint

## Current source locations (October 25)

The editable Git repositories are `/workspace/project/roborender` on Cosmos
and `/data1/workspaces/roborender/source` on Leo. Host-local asset roots resolve
through `configs/storage_paths.json`. Live owners execute pinned releases under
the corresponding `code_releases` root, not either mutable checkout. A Git
commit or source sync does not upgrade an already-running Isaac process.

Use the latest [pipeline contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md)
for current behavior and per-run receipts for actual hardware certification.
The dated profiles below are historical evidence, not current launch authority.
All requested Cosmos H200 compute is retained unless the user explicitly
authorizes releasing the exact resource; stopping a workload is not release.

## Historical October 15 protocol

Use the current [pipeline contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md)
and [MSE qualification report](QUALIFICATION_20261015_MSE.md). The latest complete
native run used **Leo GPUs 0+1, B=2 each, and four approved H200 workers**. It
completed BC, all four mixed-policy smoke rounds, held-out MSE selection and
independent final success tests. This is not full-population production data or
a success-rate improvement claim. The historical B6+6 run below had an owned
limit-pressure event and is not the current comfortable-memory admission.

New BC production uses a separately guarded single-collector profile, exact
full-buffer capacity receipts and an independent motion/seed protocol. Do not
infer its memory limits from online DAgger or GPU VRAM. Current source changes
are tested in the CPU checkout and deployed as immutable per-run source
releases; neither the mutable checkout nor a historical Git HEAD proves which
source an existing native process loaded.

## Historical warm smoke (before the current MSE protocol)

Earlier entrypoint: dedicated **Leo GPUs 0+1**, two B6 actors and four reserved
H200s. The warm four-round smoke completed on **2026-10-14 Pacific** with
12 optimizer updates, 144 chunks, 11,664 transitions and ten-case success
evaluation after each round. Both native workers retained one initialization;
H200 allocations were released. See [qualification](QUALIFICATION_20261015.md),
[standing Isaac](STANDING_ISAAC.md) and [Leo deployment](LEO_MIGRATION.md).
The smoke reused completed BC work and does not establish production-scale
reliability or policy improvement.

## Historical Beta qualification

Status date: **2026-10-12**. The following records the beta entrypoint. September
alpha deployment/certification documents remain historical reference, not
authorization to run those launchers on beta.

**Latest: selected two-RTX/four-H200 runtime PASS.** Three persistent B6+ B6
windows, 2,916 real transitions, three full-batch updates, checkpoint continuity,
lost-ack recovery and cloud W&B readback passed. The full paired v4 B5 evaluation
also completed: BC 0/256, three-update candidate 11/256, all 52 cohort starts
matched and no accounting discrepancies. These low scores are not a
policy-quality endorsement. See
[`QUALIFICATION_20261012.md`](QUALIFICATION_20261012.md) for exact scope,
identities, memory settings, evidence and remaining production decisions.

**New source integration:** [beta round automation](BETA_ROUND_AUTOMATION.md)
adds a separate four-worker controller contract, two persistent B6 actors,
50-update round budgets, final-256 success import, per-round W&B and final
checkpoint handoff through teacher75/50/25/0. It explicitly keeps current-
iteration-only data. CPU regression coverage is not a new native deployment
qualification; do not bypass `plan.py`'s production admission guard.

**October 10 bounded functional result: PASS.** One beta GPU1 actor with B8
and one reserved H200 completed three persistent 81-step windows, 1,944 real
transitions and three full-batch learner updates, including lost-ack recovery.
Fresh pre-BC collection, 50-step teacher rendering and exact BC resume also
passed. This does not admit the two-RTX profile, certify success rate, or verify
online W&B. Detailed receipts are in the ignored shared directory
`runtime/single-actor-qualification-20261010/STATUS.md` and
`runtime/single-actor-retry-20261010/`.

Read the [end-to-end pipeline contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md)
for pre-BC, BC and DAgger pseudocode, assets, GPU ownership, exact update
semantics and scientific guardrails. This runbook remains the deployment
companion; neither document promotes a bounded test to production.

## One source tree, host-local assets

| Location | Role |
|---|---|
| CPU access pod: `/workspace/project/roborender` | editing, orchestration, shared H200 code/data |
| Dedicated RTX Leo: `/data1/workspaces/roborender/source` | canonical Git source, Isaac/teacher actor code; GPUs 0+1 |
| Git remote: `ldu-nvidia/roborender-for-real-time-Dagger` | reviewed source-of-truth history; RTX is the intended push host |

The source-only manifest checks every code/config/document byte and executable
bit on both hosts. It does not hash datasets or checkpoints. Verify matching
source digests and Git HEADs separately. A reviewed Git bundle and fast-forward
can align local commits without a push. The legacy `/home/ldu/roborender`
directory and path-bound historical runs remain distinct from the Git checkout.

Keep model weights, TensorRT engines, datasets, container images, secret
tokens, caches, run journals and old continuation receipts local/ignored.
Never synchronize `.git` or use `rsync --delete` over either workspace. Before
reconciliation, preserve both source inventories under an ignored runtime
directory. Transfer only the explicit source file list, then verify again.
An unexpected file or concurrent source edit must fail the verification.

The October reconciliation retains the corrected three-view BC/render/replay
code from H200, merges beta's round-specific logging, continuation seed/budget
handling and associated regressions, and preserves the bounded Isaac 6/four-
gateway additions. Historical per-run continuation configurations are excluded
from reusable source, not discarded from the audit archive.

## Selected hardware profile

The October 10 B6/B4 attempts hit their 23 GiB container caps and remain recorded
as failed. The October 12 bounded B6+ B6 test passed after native-boundary CPU
heap compaction and limited glibc arenas, with 21 GiB caps and the same host
safety floors. Lower environment counts alone did not solve the earlier OOM.
This closes the three-window runtime/memory check, not production controller,
replay, or simulator-distribution qualification. All counts and learner batches
remain derived from each immutable run configuration.

The machine-readable entry is
[`profiles/beta_2rtx_4h200.v1.json`](profiles/beta_2rtx_4h200.v1.json). It binds
the exact tested candidate by SHA-256, the corrected BC checkpoint, beta GPU
UUIDs and remaining qualification flags.

- RTX beta (`465d6b0-lcedt`): physical GPUs **1 and 2**, six environments each.
  Two separate persistent simulator processes; **GPU0 is forbidden**.
- Actors: 21 GiB host-RAM limit and 12 CPU quota each, no container swap;
  stop the scoped test if available host RAM falls below 6 GiB or swap grows
  by more than 128 MiB. Enable `--compact-host-heap` for the qualified bounded
  launcher; it pins allocator settings and trims CPU heap only at lifecycle
  boundaries. VRAM is not the limiting resource in this sweep.
- H200: four shared Image3F/student workers on `custom.gpu.4xh200`, reservation
  `cosmos-solution-architect-xcwtffiq`, no unreserved/burst fallback.
- Four data gateways: 18620–18623; Image3F ports 18500–18503. Control ports
  remain 18610–18614. Every cross-host listener is loopback/authenticated.
- One completed window: 2 × 6 × 81 = 972 transitions and 12 chunks. The tested
  learner update is one full 12×81 batch, LR 0.0002, gradient clip 1.0, no
  activation recomputation; model checkpoint saving remains enabled.
- The qualified wire path is bounded v3. **Do not remove the legacy v4
  production guard to make a four-worker run start.** Its controller, replay
  batch and multi-actor barrier are not the beta contract. The explicit new
  beta current-only integration is documented above and still needs its live
  deployment-handoff acceptance test.

The scope of the measured evidence is in
[`throughput/README.md`](throughput/README.md). It proves execution/throughput,
not improved success, numerical Isaac 5.1/6 equivalence, or a global optimum.

## CPU-only checks and plans

Use a Python environment with the RTX package's test dependencies. PyTorch is
needed for learner/BC regression tests; an environment that skips them is not
an equivalent test gate. The existing CPU-access test environment is
`runtime/test-venv-system/bin/python`; do not copy virtual environments to RTX.

```bash
python3 pipeline/deployment/plan.py --stage qualification
python3 pipeline/deployment/plan.py --stage bc
ROBORENDER_TEST_PYTHON=/path/to/test/python \
  bash test/run_cpu_checks.sh -q
```

Plans never allocate GPUs, connect to a service, read model payloads or start
training. `--stage evaluation` describes the qualified GPU1-only, B5,
ray-traced 256-case protocol (zero H200s and no optimization). `--stage dagger`
still exits 2 for the separate production-controller/replay integration guard.
Do not interpret planning output alone as a completed live test.

Create a source release in a new ignored directory:

```bash
python3 pipeline/deployment/source_snapshot.py create \
  --output runtime/source-releases/RELEASE_ID
python3 pipeline/deployment/source_snapshot.py verify \
  --manifest runtime/source-releases/RELEASE_ID/source-manifest.json
```

The output contains `source.tar.gz`, a per-file manifest and
`source-files.txt`. Back up the destination source first, copy only those
paths, and run the same verify command there. No model/data scan is involved.
Use Git for normal subsequent changes; this release mechanism detects drift
and supports the current migration, not a second permanent version-control
system.

## Historical October 10 single-actor qualification

The October 10 follow-up has a separately versioned, bounded
`roborender.beta-runtime-check.v2` topology: one beta GPU1 actor, eight
environments, a 40 GiB container RAM cap (no swap), one reserved H200, and
three persistent 81-step windows with one full 8x81 learner update each.
It requires at least 46 GiB available host RAM at admission, keeps the 6 GiB
runtime floor, records cgroup anon/file/kernel memory counters, and exercises
one lost commit acknowledgement without repeating the simulator step.
This is a diagnostic memory-layout alternative, not a promotion of the failed
two-actor profile or a relaxation of the legacy production gate.

The fresh retry `rtx-beta-migration-runtime-single-b8-retry-1010-0611` passed:
243 exactly-once vector steps, 24 learner chunks/barriers, three optimizer
updates and uninterrupted simulator/RNN lifecycle. All 70 AdamW state counters
advanced once per update, global step 4100 to 4103. Training MSE was
1.1499913, 0.5351938, 0.5625773; this is not a task-success comparison.
Peak container RAM was 30.24 GiB, GPU1 VRAM 18,219 MiB, no OOM/swap, and host
available RAM stayed above 28.94 GiB. The full CPU suite passed 506 tests without
skips. Tested source SHA was
`6bca9d48356fd6a6a33e82c9c4aa93294b3057a10e7f07daa59ce5eb8cc2567a`;
this identifies the live-tested executable source. The later health/logging
changes described below have CPU coverage but have not yet been deployed or
live-tested on beta.

One live readiness timeout and the intentional post-commit acknowledgement
loss recovered without duplicate stepping. CPU management readiness checks
also intermittently timed out and recovered; no SSH process restarted.
The inherited image's Docker healthcheck searches for a Kit `AppReady` log
and showed unhealthy despite real actor readiness/complete receipts. That
check is now overridden for connected native qualification modes by
`dagger_rtx.container_health`. It reads only the actor's bounded progress/driver
snapshots, rejects stale/mismatched/error state, and allows a bounded checkpoint
wait. It never imports Isaac/CUDA, steps, or restarts anything. The image stays
pinned; this is a Docker command override. CPU tests pass, but the replacement
still requires a native beta check before production.
The test's W&B metrics are offline, not visible in the cloud.

The same successful B8 evidence separates host memory into approximately
21.3 GiB of anonymous process memory and 8 GiB of file cache at steady state.
The failed two-actor 23 GiB cap therefore had little margin even before cache
and kernel accounting. Lowering the environment count alone is not a proven
fix. Do not remove host-RAM admission or swap guards: qualify a revised process
layout/cap against actual host availability before allocating the four H200s.

The first single-actor attempt completed one real window/update, then exposed
a qualification-only lost-ack probe that assumed a sharded client. The probe
now supports both a direct gateway and a sharded gateway without modifying any
request/receipt. An accepted learner job's transient status-404 is retried as a
bounded read only; it never submits another update. CPU controllers must not
open the H200's live SQLite WAL through the shared volume. The qualification
receipt audit runs over the pinned SSH route on the database-owning H200 host.
Failed partial windows remain diagnostic and are never resumed or learned.

1. Integrate the two-actor/four-worker profile with the persistent production
   controller, without changing teacher labels, three-view mapping, full-batch
   statistics or exactly-once stepping. Explicitly decide the production
   replay contract; do not reuse the legacy B28 schedule at B12 silently.
2. Repeat sustained windows and checkpoint/optimizer continuity for the actual
   production topology, with authenticated online W&B. The single-actor B8
   smoke has passed these runtime checks with offline logging only.
3. Port/qualify the beta evaluator and settle the Raven-approved observation,
   reset and terminal protocol. Evaluate the same checkpoint/panel before
   interpreting improvement. The old low success numbers are not new beta
   baseline measurements.
4. Check the Isaac 6 port's state/distribution behavior, and replace the
   temporary Mac/alpha-dependent management route for unattended beta use.

BC itself uses the existing corrected autoregressive rendered data on H200;
hardware migration does not require recollection or rendering. Its full
16×81 training batch is separate from DAgger's tested 12×81 batch. Allocate
one H200 for the existing single-device BC trainer; retain the allocation after
completion unless the user explicitly authorizes release of that resource.
The two-RTX/four-H200 allocation applies to concrete online DAgger work only.

Log training MSE per actual optimizer update, not per camera frame. Log BC
held-out MSE each epoch. Success-rate logging comes from completed closed-loop
episodes, with checkpoint/panel identities and numerator/denominator recorded;
never substitute MSE or synthetic test outcomes for task success.

## Online logging admission (CPU only)

Before GPU allocation, use an explicitly approved credential file and entity:

```bash
python3 -m pipeline.deployment.wandb_preflight smoke \
  --entity YOUR_WANDB_ENTITY \
  --api-key-file /approved/private/wandb-key \
  --run-id smoke-dagger-admission-UNIQUE_ID \
  --output runtime/logging-qualification/UNIQUE_ID
```

Use lowercase letters, digits and hyphens in `UNIQUE_ID`. The key file must be
owned by the current user, owner-only, and a regular file (not a symlink).
Alternatively supply `WANDB_API_KEY` in the process environment, never a config
or shell argument. This writes one clearly named diagnostic scalar to project
`dagger` and reads it back from the cloud; it does not log invented MSE or
success, allocate a GPU, or start learning. `check` authenticates only and does
not claim project write permission. No credential is written to source or run
config, and absent credentials do not trigger an interactive SDK login.

The bounded CPU controller accepts `--wandb-entity` and
`--wandb-api-key-file`. Start it before starting RTX collection. Online mode
requires authentication; explicit offline diagnostic mode remains available
but cannot qualify production logging. The controller prints the run URL at
initialization and verifies actual final MSE/global-step/update-count fields
after `finish()`. Its `.logging-admission.json` is the readback receipt;
setting `mode=online` alone is not a verified result. The new controller's
actual-training readback still requires a native run. The separate CPU-only
W&B write/readback admission passed with the existing approved shared
credential on October 10 (Pacific), as run
`nvidia/dagger/smoke-dagger-admission-bridge-repair-20261010`. It logs a
diagnostic admission scalar, not a fabricated MSE or success rate.

An existing owner-only netrc file can be reused without copying its key:

```bash
NETRC=/approved/private/wandb.netrc python3 -m pipeline.deployment.wandb_preflight check --entity nvidia
```

Only an explicit `NETRC` path is read, and only its exact `api.wandb.ai`
entry is accepted. `WANDB_API_KEY` or an explicit raw key file takes priority.
The same `NETRC` environment works for the CPU controller's online admission.
Do not commit the credential or place it in a run configuration.

## October 11 online checks and actual success smoke

The subsequent single-actor B8 retry completed three real DAgger windows with
online W&B (`smoke-dagger-beta-runtime-online-b8-retry-1011-0513`). This closes
the logging check for that bounded topology; it does not admit the two-actor
profile above.

The inference-only paired success smoke
`smoke-success-beta-bc-vs-dagger-20261011-2223` then completed on beta GPU1:

| Policy | Checkpoint SHA-256 | Completed / successes |
|---|---|---|
| Corrected BC, epoch 41 | `053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718` | 5 / 0 |
| Three-update smoke DAgger | `7ebae7db67706b9176b3ca93dd650051083d175f12320c382cdec4a768c5c020` | 5 / 0 |

Both used seed 1103, the same five held-out motion assignments and byte-identical
initial physical/state observations. Input was native ray-traced three-view RGB,
not Image3F. Each lane contributed only its first completed episode; terminal
goal flags were captured before autoreset and OR-latched over that episode.
Native Raven counts and exported counts agreed, model weights remained unchanged,
and neither arm performed an optimizer step. Runtime physics identities matched.
W&B final numerator, denominator and success-rate fields were read back from the
cloud. Both owned containers stopped; no H200 was allocated for this evaluation.

Evidence is on shared storage under
`runtime/success-validation-20261011T2223Z/`, with the original per-step trace on
beta under `runtime/success-validations/` and the matching run ID. This is a
**five-case diagnostic, not the sealed 256-case panel, a photorealistic evaluation,
or certification of production readiness**. Zero out of five is not a reliable
estimate of the full-panel policy success rate.

The requested production name is `2026-10-11-prod-1`; it has not been launched.
The existing B64-per-seed evaluator is not qualified on beta. Changing it to
small batches requires a new versioned panel with matched reset states, not
relabeling the old panel. The two-actor RAM/controller gates above also remain.
