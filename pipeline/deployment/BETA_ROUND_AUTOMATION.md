# Beta round automation

Status: 2026-10-12. **Implemented, CPU-tested; this new unattended deployment
path has not yet completed a native four-round run.** The earlier genuine
three-window and 256-case evidence is in
[QUALIFICATION_20261012.md](QUALIFICATION_20261012.md); it must not be relabeled
as testing this new allocator/controller/evaluator handoff.

## Fixed algorithm and hardware

The new contract is `roborender.cross_gpu_dagger_round.beta.v1`, separate from
the alpha B28/eight-worker contract. No old guard was removed.

| Setting | Beta campaign |
|---|---|
| RTX | Beta physical GPUs 1 and 2, six environments each; GPU0 forbidden |
| RTX host RAM | 21 GiB/container, 12 CPU quota, no swap, compact-host-heap |
| H200 | Four genuine Image3F/student workers, v3 ports 18620–18623 |
| Allocation | `custom.gpu.4xh200`, `cosmos-solution-architect-xcwtffiq`, no burst |
| One update | Twelve 81-row chunks = 972 transitions; one full B×T forward/backward |
| Optimizer | Existing Raven learner, LR .0002, gradient clip 1, one minibatch |
| Data | Current iteration only; no BC/prior-round replay |
| Observation/action semantics | Three views; native ray-traced augmentation, randomized initial episode lengths; unchanged Image3F/teacher/Homie/selection contracts |
| Round schedule | Teacher 75%, 50%, 25%, 0%; exactly 50 updates each |
| Checkpoints | Save every update; next round uses the final iteration-50 checkpoint |
| Validation | Full 256-case beta B5-v4 success panel after update 50, inference-only |
| Logging | Project `dagger`; separate `RUN-rN-teacherPERCENT` run; actual MSE/update and success numerator/256 |

There is one persistent simulator/history/RNG lifecycle per RTX actor per
round: 4,050 vector steps. An 81-step boundary does **not** reset Isaac.
Only actual done/timeout resets an episode. A new round has an explicit new
seed and new native lifecycle after final evaluation.

The success panel is the same version across rounds. It uses native ray-traced
three-view inputs, no Image3F and no optimizer. The candidate and original BC
reference are evaluated with byte-matched initial physical/state observations.
Goal events are OR-latched before reset, including a goal followed by timeout.
All 256 predetermined cases must finish; partial results cannot advance a
round. This beta panel is not equivalent to the old B64 benchmark or Raven's
published score, and improved success is not guaranteed.

## Control flow

```text
CPU: verify source, checkpoint, host routes, online W&B admission
  -> reserve 4 H200; start genuine services; pin identities and SSH forwards
  -> register round; create its W&B run
  -> start two persistent RTX actors (6 lanes each)
       repeat 50 times:
         81 times: capture+teacher -> H200 render/student/selection
                  -> one RTX env.step -> commit outcomes
         12 sealed chunks -> one H200 update -> saved candidate
         both RTX actors wait for that exact candidate before next window
  -> actors exit; stop owned services/forwards; release H200 allocation
  -> RTX-only paired 256-case evaluation; retain checkpoint/panel evidence
  -> restore CPU-only ledger services (exclusive owner; no GPU allocation)
  -> import actual outcomes -> W&B final metrics/readback -> close round
  -> next teacher probability, new seed, final checkpoint -> repeat
```

The shared SQLite ledger has **one host owner at a time**. The CPU scheduler
uses HTTP while H200 owns it. GPU-host services must stop before CPU-only
services open it. Shared-storage paths alone do not make cross-host WAL access
safe. The bundle-level lock and immutable ledger identity enforce this.

## Components

| File | Responsibility |
|---|---|
| `pipeline/dagger/beta_campaign.py` | Durable four-round state machine, final checkpoint lineage, eval/logging gates |
| `pipeline/dagger/production_controller.py` | Existing coordinator/learner submission, now explicitly supports B12/four workers |
| `pipeline/dagger/rtx/code/src/dagger_rtx/beta_round.py` | Shared strict beta config, actors, probabilities, GPU identities |
| `pipeline/dagger/rtx/code/src/dagger_rtx/qualification_driver.py` | `BetaRoundDriver` reuses the tested persistent-window lifecycle; no production fault injection |
| `pipeline/dagger/beta_validation.py` | Small real episode receipts into the existing success ledger; no image scans |
| `pipeline/deployment/beta_site.py` | Concrete scoped host operations: reserve, launch, collect, release, evaluate, CPU control, abort |
| `pipeline/deployment/beta_operations.py` | SHA-pinned operation commands, durable identities, bounded reattachment |
| `pipeline/deployment/owned_process.py` | Workload lifetime independent of SSH, bounded deadlines, scoped stop |
| `pipeline/deployment/beta_lease_watchdog.py` | CPU-only orphan guard; missing campaign heartbeat triggers owned allocation cleanup |
| `pipeline/deployment/h200/beta_service_host.py` | Exclusive service-bundle owner, readiness, child cleanup |
| `pipeline/deployment/h200/beta_pod.py` | Approved reservation only; no Codex server or unreserved fallback |

## Preparing and inspecting a plan

First run the CPU test gate and create an immutable source release using the
[source-sync runbook](WORKING_REPO.md). Publish that exact archive/manifest to
beta's `runtime/source-releases/RELEASE/`, extracting source under `frozen/`.
No models, caches, datasets, secrets, or `.git` travel in the source archive.

Copy `profiles/beta_site.direct.example.json` into the ignored campaign workspace
and replace its `RELEASE` paths. This is a host binding, not a credential
file. It explicitly references approved existing SSH keys and W&B netrc;
their values never enter source or W&B configs.

The v2 example uses Beta's own cloud login and the Alpha-free management route.
That service passed authenticated binary traffic and controlled reconnect on
October 12; [its evidence](connectivity/README.md) is not a native production
qualification. The legacy `profiles/beta_site.example.json` retains the old
Mac/alpha-dependent v1 route for historical compatibility only. There is no
automatic fallback from v2 to v1.

```bash
cd /workspace/project/roborender
python3 -m pipeline.deployment.prepare_beta_campaign \
  --site /absolute/runtime/site.json \
  --run-id dagger-beta-YYYYMMDD-prod-1 --seed 610291 \
  --checkpoint /absolute/path/to/corrected-BC-best_validation.pt \
  --panel /absolute/path/to/panel.v4.json \
  --output /workspace/project/roborender/runtime/beta-campaigns/UNIQUE_PLAN

python3 -m pipeline.dagger.beta_campaign \
  --first-round-config /absolute/UNIQUE_PLAN/first-round-template.json \
  --root /absolute/UNIQUE_PLAN/active
```

Both commands above allocate **zero GPUs** and start **zero workloads**. The
template has no gateway PIDs. `PREPARE` must pin the four actual live PIDs in
the immutable round config before registration or native launch.

After the native deployment acceptance gate is approved, the explicit launch
command adds `--operations .../operations.json --execute`, with an approved
`NETRC` environment reference (or `--wandb-key-file`) and a Python environment
containing W&B. Keep the scheduler on the CPU access pod, not an H200 pod.
The named approved reservation must be available; no fallback is attempted.

## Monitoring and failure behavior

- CPU: `active/state.json`, `active/phases.jsonl`, per-operation `console.log`.
  State reports round, iteration, actor barrier counts and current stage.
- RTX: `runtime/beta-production/RUN/round-N/gpu{1,2}/progress.json`,
  `phase-boundaries.jsonl`, `beta-round-state.json`; one `pair-result.json`.
- W&B: the round run is created before collection; loss appears only after
  actual optimization. Initial collection is not an optimizer step. No fake
  MSE or inferred success is logged.
- All 50 checkpoints and the final success comparison stay on shared storage.
  The next round uses final-50, never an incidental best-MSE checkpoint.
- Transient metadata/SSH failures retry only identical idempotent requests.
  HTTP control has one 120-second recovery deadline, longer than the tunnel's
  75-second allowance; individual socket calls use at most 15 seconds. Recovery
  prints route/type/timing events and never nests another deadline after
  exhaustion. Short health-probe jitter does not immediately restart SSH.
  Semantic HTTP errors, changed payloads, stale identities and uncertainty
  fail closed. Renderer-worker PID changes require re-attestation.
- Restarting the CPU scheduler can reattach to an existing operation. It
  cannot recreate missing simulator/RNN/RNG state. Lost native processes,
  partial chunks and abandoned windows are never resumed or learned.
- SIGTERM routes through scoped cleanup. Deadline guards are a backstop if
  management disappears; a separate CPU watchdog requests cleanup after a
  five-minute scheduler-heartbeat loss. It records and retries cleanup failures
  rather than claiming unreachable resources were released. Cleanup never kills unknown workloads or probes RTX
  GPU0. A cleanup failure is reported, not mislabeled as released capacity.
- At success-rate evaluation, H200 capacity is released. CPU control services
  can remain until the next round; they reserve no GPU.

Cold-path checks hash source/checkpoints and small receipts only. No full
trajectory reread/recompression is added to an iteration. The existing compact
RTX seals, batched v3 wire and lean H200 persistence are preserved.

## Remaining acceptance

CPU tests cover four rounds/200 checkpoint handoffs, single native-lifecycle
calls, partial/mismatched rejection, real HTTP import/retry of 256 test
outcomes, reservation safety, scoped detached process ownership and recovery.
These are tests, **not new live success measurements**.

Before unattended promotion, exercise the new site binding and actual
GPU-to-CPU ledger ownership transfer, then the first native beta-production
checkpoint handoff. Confirm genuine process identities, W&B MSE, final panel
import and next-round checkpoint equality. The prior native qualification
does not by itself certify new deployment code or policy quality.
