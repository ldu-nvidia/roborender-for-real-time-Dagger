# Host deployment and migration qualification

These files configure infrastructure, not model or rollout semantics. Weights,
datasets, container layers, credentials and execution logs remain outside Git.

Current result: the dedicated Leo two-RTX/four-H200 warm campaign completed all
four bounded DAgger rounds, optimization, success evaluation and verified W&B.
Read [QUALIFICATION_20261015.md](QUALIFICATION_20261015.md) for its exact scope
and [STANDING_ISAAC.md](STANDING_ISAAC.md) for the persistent native service.
The dated Beta migration results below are historical.

The [Alpha-free connection service](connectivity/README.md) and versioned
[`beta-site.v2` example](profiles/beta_site.direct.example.json) use Beta's own
cloud login. The deployed management service passed authenticated return traffic,
binary round trips and a controlled 25.57-second reconnect on October 12 without
Alpha or the Mac. Genuine workload endpoints and the bounded native pipeline
must still be checked over the new route before declaring the migration
complete. The legacy v1 route is preserved explicitly and is never an automatic
fallback. The connection test itself allocated no GPUs.

**Start here:** [`WORKING_REPO.md`](WORKING_REPO.md) is the current source-sync,
profile and readiness runbook. The dated results below are retained as
migration evidence; the October 8 launcher is not the beta Isaac 6 entrypoint.

The current source-only implementation of four-round beta deployment is
[BETA_ROUND_AUTOMATION.md](BETA_ROUND_AUTOMATION.md). It keeps the old alpha
launcher separate and requires new native deployment acceptance before an
unattended production claim.

## Throughput follow-up

The two-RTX/four-H200 bounded sweep is described in
[`throughput/README.md`](throughput/README.md). Its run-specific results live
under `runtime/throughput-sweeps/evidence-20261009`, not in version control.
The one-GPU qualification below predates that separate allocation. A capture
PASS alone does not establish closed-loop host-RAM capacity.
The completed sweep's best full-window candidate is 6 environments per RTX
(12 total) sharing 4 H200 workers, recorded in
[`throughput/candidate_beta_2rtx_4h200.json`](throughput/candidate_beta_2rtx_4h200.json).
Both native observation branches and real learner updates were exercised.
This bounded v3 evidence does not convert the production v4/eight-gateway
launcher or certify numerical equivalence of the new simulator.

## Current functional qualification, 2026-10-09

The separately versioned Isaac 6 candidate now runs on **beta physical GPU1**.
The original Isaac 5.1 image still fails on beta; the old host manifest/launcher
below is historical evidence, not the qualified Isaac 6 entrypoint.

Verified with real models/data (no fake backend):

- Native B=2 capture: three-view controls plus privileged teacher/Homie states
  and actions, zero action steps.
- Authenticated beta → H200 → beta one-step transaction: two committed rows and
  exactly one vector `env.step`.
- One B=2 × 81-step DAgger window: 162 commits, two synchronized learner chunks,
  one actual done/reset, 122 teacher and 40 student selections at beta=0.75.
  Both sampled episode assignments used generated controls; the live raytraced
  DAgger observation branch is not covered by this particular run.
- One real DAgger optimizer update: MSE 0.7347136732, finite gradient norm
  3.1683702469 before clipping, disposable candidate saved. This is functional
  evidence, **not** a success-rate or policy-quality result.
- BC resume: epoch 41 / global step 4100, AdamW counters and RNG restored;
  one full 16×81 update, MSE 0.0304599252, sampled held-out validation executed.
- Fresh native teacher-only BC collection: 243 vector steps over two
  environments, 486 exact behavior-action/teacher-label equalities, four complete
  three-view chunks, and zero optimizer calls. Raven's original episode-tail
  discard and 81-frame chunking code is retained.
- Fresh beta chunk → 50-step teacher (CFG 5) → three-view rendered archive:
  PASS; every original state/action/control array compared equal. Rendering
  took 54.30 seconds including output publication, after 40.38 seconds loading.
- Fresh rendered chunk plus fifteen existing training-only chunks → real BC
  loader/full 16×81 update: PASS, MSE 0.0544228003, finite gradients and a
  disposable checkpoint. No sample was inserted into the production pool.

The BC checkpoint is the corrected autoregressive dataset checkpoint, SHA256
`053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718`,
not the older September 7 checkpoint.

The tests use one explicitly reserved `custom.gpu.1xh200`, not an idle eight-GPU
node. Original checkpoints and production datasets are never overwritten.
Qualification receipts are under
`runtime/migrations/alpha-to-beta-20261008/{qualification,h200}` on shared storage.
RTX evidence is under `/home/ldu/roborender/runtime/isaac6-native-qualification`.
The temporary H200 pod was deleted and its absence verified after the tests;
beta GPU1 is idle. The latest infrastructure/port/native-topology CPU suites
passed 35 tests. No production run or success-rate evaluation was launched.

The later two-actor sweep covered both observation branches and selected B6
per actor. Remaining gates are persistent multi-window/four-worker production
integration, success-rate protocol, live W&B, and an independent beta management
route. Current service forwarding is authenticated and loopback-only but still
uses the existing alpha Lepton proxy; no personal API token was copied.
Isaac 6 physics are **not** certified numerically equivalent to the old runtime.

## Historical October 8 Isaac 5.1 attempt (failed; not a beta launch recipe)

`rtx_actor/build.sh` reconstructs the missing alpha actor image from a pinned
Isaac Lab base digest and the recovered dependency recipe. It is **not** the
original byte-identical image and requires native qualification before use.

The hash-bound machine manifest is
`pipeline/dagger/rtx/config/hosts/rtx-6k-beta.json`. It pins beta's hostname,
physical GPU1 UUID, container identity and local asset roots. The launcher and
preflight accept it through `DAGGER_HOST_DEPLOYMENT_FILE` and
`DAGGER_HOST_DEPLOYMENT_SHA256`; neither option changes scientific settings.
Without an explicit manifest, existing alpha defaults are preserved.

On beta, the initial capture-only gate is:

```bash
cd /home/ldu/roborender
export DAGGER_HOST_DEPLOYMENT_FILE="$PWD/pipeline/dagger/rtx/config/hosts/rtx-6k-beta.json"
export DAGGER_HOST_DEPLOYMENT_SHA256=14fdbef0a8a6034425966f3f0805d45e3f6c4184db6810cb4013c41c5cf5e88a
bash pipeline/dagger/rtx/scripts/launch_native_raven.sh \
  capture-only 2 smoke-beta-capture-20261008 1 smoke-beta-capture-g1 0
```

Use a fresh run identity for a new native attempt. A CUDA-only preflight PASS
does not certify Isaac/Vulkan or teacher capture. Do not promote smoke output
or reuse a failed simulator's durable state.

### Qualification result, 2026-10-08

- Combined host-binding/capacity/protocol suite: 72 passed in 15.33 seconds.
- Rebuilt image dependency check: Python 3.11.13, Torch 2.7.0+cu128,
  torchvision 0.22.0+cu128, NumPy 1.26.0.
- GPU1-only CUDA preflight: PASS, one visible device, capability 12.0.
- Native B=2 capture-only: FAIL during Isaac graphics initialization, before
  capture or `env.step`.
- Clean AppLauncher without Raven/cache mounts: same startup segfault.
- Unmodified pinned base image, minimal SimulationApp, no Raven code, no
  migrated caches, no reconstructed Python environment: same
  `librtx.scenedb.plugin.so` startup segfault.

Alpha's driver is 580.173.02; beta's is 610.57.04. This is consistent with the
[documented Isaac 5.1 RTX startup issue](https://github.com/isaac-sim/IsaacSim/issues/677),
but changing the driver has not been tested here. The
[Isaac 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
list Linux 580.65.06 as tested. Driver maintenance affects all four GPUs and
requires explicit operator/user approval; do not downgrade automatically.
Upgrading to Isaac 6 instead is a simulator migration, not an equivalent
workstation move, and requires separate approval and scientific requalification.

Evidence lives at `/home/ldu/roborender_migration_20261008` on beta and
`/workspace/project/roborender/runtime/migrations/alpha-to-beta-20261008` on the
shared volume. No real transitions or optimizer updates completed in these tests.

## H200 smoke allocation

The current `h200/beta_pod.py` allocator gives each **new, concretely owned GPU
workload** an explicit queue policy: `high-9000`, `can_preempt=true`, and
`can_be_preempted=false`. It never inherits the CPU access pod's `mid-4000` or
null preemption defaults. `--priority-class high-7000` is the only lower
reviewed override; arbitrary/system levels are rejected. The approved
`cosmos-solution-architect-xcwtffiq` reservation, no-burst restriction and bounded
workload ownership remain unchanged. These fields request
scheduler behavior; they do not guarantee placement or prevent every form of
infrastructure interruption. Existing owned deployments are not automatically
modified or rejected, so old status, cleanup and recovery evidence remains
usable. Changing an existing queued owner requires a separately reviewed,
scoped control action.

### Allocation retention override (2026-10-15)

The user now requires an acquired H200 allocation to remain reserved until an
explicit user release request. This overrides the older prompt-release guidance
below and in historical run evidence. It does not authorize artificial GPU
activity, unreserved fallback, a second allocation, or automatic workload replay.

Current `BCWorkload`, `Site` and `ResidentSite` cleanup still stops only their
owned workload/services and preserves checkpoints, outputs and uncertainty
evidence. Cleanup retains the allocation by default and records `retained.json`
with `h200_allocation_released=false` in the operation receipt. It must not publish
a release proof just because processes stopped. The independent orphan guard
recognizes completed retained cleanup; process TTL enforcement is unchanged.

`h200/beta_pod.py delete` now refuses unattended deletion, including old callers
that stream the current broker source. Only a separately approved explicit
`--confirm-release EXACT_POD_NAME` permits deleting that exact attested owner.
The same optional flag is available for `beta_site.py` and
`resident_campaign_site.py` release/abort operations; Python callers use
`release(confirm_release=exact_pod)` or `abort(confirm_release=exact_pod)`.
Wrong confirmations fail before cleanup; malformed allocation status cannot
be treated as observed absence. Frozen historical source is not silently patched.

The existing multi-round controller still requires an actual allocation-release
proof before advancing. It therefore fails closed at that boundary under this
new retention policy. A reviewed retained-node service/token/checkpoint rebinding
and recovery path is still needed before another unattended multi-round campaign;
do not forge a release receipt or acquire another node to bypass this gate.
CPU monitoring deadlines may detach without releasing capacity, but must not
call workload cleanup if an approved GPU job is intended to continue running.

`h200/create_migration_smoke_pod.py` uses the authenticated Lepton SDK on the
management host. It allocates only a fresh `leo-roborender-beta-smoke-*` pod,
`custom.gpu.1xh200`, under `cosmos-solution-architect-xcwtffiq`, with no burst
fallback. Use `--dry-run` first; this now builds the plan locally and never
sends a mutation. The gateway ignored `dryrun=true` on deployment creation;
the initial API validation briefly created the smoke pod. It was deleted and
its absence confirmed. Two CPU regression tests now cover non-mutating dry-run
and exact reservation/shape/public-key-only creation. No GPU smoke ran there.
It never modifies existing workloads or launches Codex on the GPU.

The entrypoint only exposes key-authenticated SSH for the concrete smoke work;
it does not start training automatically. Confirm a usable native RTX capture
and reconciled source before allocating for the distributed test. The historical
smoke procedure released its exact pod after completion. Under the current user
override, stop the owned workload when appropriate but retain the node until the
user requests release; do not run artificial activity to keep it occupied.

Public keys may be supplied as arguments. Private keys, bearer tokens and API
credentials must not be copied into Git or placed in these manifests. The
temporary Mac/alpha management route is not an independent beta Lepton route.

## Remaining production acceptance gates

1. Qualify physics/data-distribution behavior after the explicit Isaac 6 port;
   successful execution does not establish numerical equivalence to Isaac 5.1.
2. Freeze the reconciled working source with `source_snapshot.py` and use its
   per-file manifest on both hosts. Identical working source is distinct from
   identical Git commits; a reviewed commit/push remains a separate operation.
3. Replace the temporary alpha-dependent management route with beta-owned
   authentication. The tested loopback service tunnel itself is functional.
4. Integrate and soak the two-actor/four-H200 persistent production
   path, including repeated checkpoint handoffs and online W&B. The completed
   bounded sweep covers one window/update, not long-duration certification.
   The October 10 B6 retry hit a container RAM limit before its first full
   window; a B4 retry also hit the same cap. Revise the host-RAM/process plan
   before another check. Do not promote the old single-window throughput
   results to a sustained-safe production setting.

Success-rate evaluation remains a separate scientific protocol gate. Its legacy
launcher still references alpha-only evaluation assets and is not beta-qualified.

## Versioned Isaac 6 migration candidate

The user approved completing the compatibility port on October 9. This does not
authorize a host driver downgrade or certify the new simulator as numerically
identical to alpha. The original Raven source, assets, and certified launcher
remain unchanged.

- Build: `rtx_actor/build_isaac6.sh`, pinned Isaac 6 base and IsaacLab
  `v3.0.0-beta2` commit `28a37cecdd433c22d9eabd6a5954add9f13a8951`.
- Candidate image:
  `sha256:7aa2c5aaf53e55e749c9f63b6430b9554b4ca49928cf35f376a646669da7b509`.
- PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Python 3.12, NumPy 2.2.6,
  tensordict 0.11.0 (required by the original Raven environment).
  Its complete installed dependency inventory is in the image at
  `/opt/roborender/isaac6-python-freeze.txt`.
- `materialize_isaac6.py` verifies ten exact input file hashes and writes a
  separate overlay with original/derived hashes. It refuses an existing output
  directory and never edits the original source. The original teacher/Homie
  trainer and checkpoint bytes are not transformed.
- `roborender_isaac6.py` translates only simulator API boundaries: Torch versus
  ProxyArray/Warp; WXYZ versus XYZW quaternions; PhysX COM access; external wrench
  buffers; semantic labels and prim creation. Root position/velocity and action
  values are not changed. Native Lab3 Kit replaces the obsolete copied Kit5 file.
- `qualify_native_isaac6.py` is a separate, bounded qualification runner restricted
  to beta's GPU1 UUID. It mounts the overlay/weights/code read-only, uses a fresh
  evidence directory and a separate Isaac6 cache. The default `capture-only`
  mode has no H200 endpoint or published port (outbound bridge networking permits
  original NVIDIA materials), and never calls an environment action step or
  optimizer. It requires an
  explicit source snapshot beneath `/home/ldu/roborender/runtime`.
- Explicit `isolated-one-step` and `single-window` modes reuse the same native
  owner, callbacks, v3 codec, and exactly-once batch barrier. They permit one
  genuine H200 gateway for B=1/2 qualification, with a pinned gateway PID and a
  scoped read-only bearer-token mount. Host networking reaches the SSH
  loopback forward; no application ports are published. They never submit a
  learner job or activate a candidate automatically. Production remains fixed
  to eight gateways; this qualification option does not relax its topology.
- The Lab3 semantic renderer passes the JSON string `"{}"` as a label mapping.
  Replicator 1.13.25 treats its zero labels as an RGBA mapping (`0 * 4 == 0`),
  overriding `colorize=False`. The port explicitly restores integer IDs on the
  two semantic nodes for each conditioning-camera render product after attach.
  A node-mutation test checks the scope; native ID/mask validation is still
  required. Do not infer segmentation correctness from dtype alone: packed
  RGBA can also appear as int32.

Eight CPU port tests pass, including nonidentity quaternion round trips, root
state position/velocity preservation, no input mutation, unchanged original
source hashes, and rejection of unknown sources. These tests do not replace
native capture/step/render/learner validation. Do not promote the image or
change the production host manifest until those gates pass.

## Container-only investigation (October 8, 2026, Pacific)

The temporary Mac SSH bridge was initially disconnected. Direct CPU-to-beta and
the old GCP relay were also unreachable. The revised Mac bootstrap now preserves
the user's working SSH multiplexed route, labels failed stages, and cancels only
its own reverse forwards on exit. Access was restored and the isolated graphics
test below ran on beta. No production image/manifest or host driver was changed.

`rtx_actor/build_graphics_probe.sh 5.1.0|6.0.0` builds a small diagnostic layer
on the selected official Isaac image. The pulled registry digest is resolved
and pinned before building; it does not bake in Raven code, credentials,
checkpoints, or old caches. It is intentionally separate from the actor image.

`rtx_actor/qualify_graphics.py --image-id sha256:...` requires the diagnostic
image label/immutable ID, the beta hostname, and three idle samples of beta's
GPU1 UUID. It exposes only that GPU, uses no network, bounds host memory/runtime,
and preserves console/output evidence. Its cleanup refuses unrelated containers.
The inner `graphics_probe.py` tests application startup and RGB/metric-depth
camera output before and after moving a cube; a PASS is not a Raven/teacher test.

Four CPU isolation/cleanup tests passed; Python compilation and shell syntax
passed. The `--plan-only` path was checked without Docker or GPU use.

Once access is restored:

```bash
cd /home/ldu/roborender
bash pipeline/deployment/rtx_actor/build_graphics_probe.sh 6.0.0
# Use the exact DIAGNOSTIC_IMAGE ID printed by that build:
python3 pipeline/deployment/rtx_actor/qualify_graphics.py --image-id sha256:...
```

An upstream maintainer confirms the matching 5.1 scenedb crash is fixed in
[Isaac Sim 6](https://github.com/isaac-sim/IsaacSim/issues/677). This motivates
the isolated comparison; it does not establish that beta or our pipeline works.
Isaac 6 also moves to Python 3.12 and revised APIs; see the
[migration guide](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/migration_strategy.html).
Do not overlay its native RTX libraries into the old Kit or present a 6.x image
as a source-equivalent Raven runtime. If graphics works, the next gates are a
versioned IsaacLab/Raven compatibility port, exact teacher/Homie loading,
three-view control checks, and bounded native steps before distributed training.

The eventual self-contained RTX application image should include Isaac/IsaacLab,
the pinned Raven source, teacher and Homie weights, scene/motion assets, and the
RTX service/collection/evaluation code. Anchors and changing student checkpoints
can be immutable read-only inputs; output/cache volumes must remain persistent.
The host kernel/NVIDIA driver remains a required external dependency, as described
by the [NVIDIA Container Toolkit prerequisites](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Beta graphics result, 2026-10-09 05:32 UTC

The isolated Isaac Sim 6.0.0 probe **passed** on physical GPU1, UUID
`GPU-0c1949ae-379a-2f2e-0f23-621de2eee409`, with unchanged host driver 610.57.04.
This is a graphics compatibility result, **not BC/DAgger qualification**.

- Official base: `nvcr.io/nvidia/isaac-sim@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d`.
- Tested diagnostic image: `sha256:0fc41da36b59824681c7d4593e4fd272dd45b6c1d6e639a9ab74cabb748043c2`.
- SimulationApp reached ready at 27.3 seconds; complete cold probe took 312.7
  seconds, exit 0, with no timeout. It used 4 CPU cores and a 16 GiB RAM cap.
- Camera output: RGBA `[108,192,4]`, metric depth `[108,192]`; moving the cube
  changed 16,421 RGB pixels and 8,964 depth pixels.
- No Raven environment, teacher, Homie or student inference was executed;
  `raven_env_steps=0`, `production_qualified=false`.
- The temporary container exited/was removed. GPU1 returned to 21 MiB/0%.
  GPU0 was not probed or used; no H200 allocation was created.

Evidence on beta:
`/home/ldu/roborender/runtime/rtx_container_qualification/20261009T052729Z-6ae55dc0/`.
A copy, including the executed diagnostic source, is retained on shared storage:
`/workspace/project/roborender/runtime/migrations/alpha-to-beta-20261008/qualification/20261009T052729Z-6ae55dc0/`.

The diagnostic source was subsequently hardened with per-phase messages and
traceback timers. Future probes use 8 CPU cores and 32 GiB RAM, matching the
[Isaac 6 minimum RAM requirement](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html),
and require 40 GiB host memory available. These runner changes have four passing
CPU tests; they are not the image used for the above GPU result. Rebuild before
using the updated probe. Do not silently promote the Isaac 6 diagnostic image
to production: porting the Raven/IsaacLab runtime changes the simulator version
and needs explicit approval plus teacher/observation/transition requalification.
