# Standing Isaac workers on Leo

For the October28 explicit `run_phase` selector, current integration status and
remaining BC/collection-bound adapter work, see [phase modes](PHASE_MODES.md).
The B6/B5 descriptions below document the original implementation, not the
current capacity certificate or the requested B32 collection-bound benchmark.

This is an **opt-in runtime**, not a change to an already pinned experiment.
For a failed next-round allocation before any native command, see the separate
[round-boundary recovery procedure](ROUND_BOUNDARY_RECOVERY.md). It preserves
completed training/evaluations and never restores or replays a lost simulator.
It keeps one native Kit process per authorized RTX GPU (Leo physical 0 and 1;
display GPU2 remains forbidden). Isaac owns every physical step. CPU HTTP
threads accept small commands; only the native main thread executes them.

## What stays warm

Kit, CUDA/kernel/shader caches and teacher/Homie/student model objects stay
in the worker. Compatible six-lane training scenes can be reused; a changed
scene seed rebuilds its seed-selected object/material configuration without
restarting Kit. Scene reconfiguration is deliberately not disguised as a pure
episode reset.
An 81-step window never resets the scene or changes episode lifecycle.
Commands are sequential: collection, checkpoint barrier, success evaluation,
then a new declared round. Idle workers do not step or manufacture GPU work.
Offline BC optimization/rendering still runs on H200 without touching Isaac.

The qualified evaluator has **five lanes**, whereas training has six. The
first implementation keeps these exact batch sizes and held-out motions.
Switching profiles releases the old Lab3 context/scene and creates the new
one **inside the same Kit process**. This still has a scene-reconfiguration
cost; it is not a promise of zero initialization work. Same-profile commands
reuse their scene when the scene seed is unchanged. Never mask a sixth lane and
call it the old B5 panel.

At a real phase boundary, reset commands/history, dataset/RNG, event clocks,
goal/contact latches and controller rollout state. Native episode reset
semantics inside a collection round remain unchanged. `resident-phase-reset`
is an infrastructure lifecycle extension, not certified bit-identical cold
training RNG consumption. New scene seeds preserve native asset resampling;
reset seeds still change native sampled states/appearance.
Startup physics-randomization profiles are rejected until separately qualified.
Evaluation still requires byte-identical paired
physical/state observations, unchanged policy weights and zero optimizer steps.

## Interfaces

Authenticated loopback ports are **18640/18641**, separate from H200 gateways.
Use the existing pinned SSH route; never publish these ports externally.

- `GET /ready`, `GET /v1/status`: native identity/source/GPU/phase/progress.
- `POST /v1/commands`: version `roborender.isaac-resident.v1`, exact fields
  `schema,worker_instance_id,command_id,operation,payload`.
- `GET /v1/commands/ID`: queued/running/complete/uncertain receipt and timing.
- Operations: `capture_reset`, `prepare_scene`, `collect_round`,
  `evaluate_pair`, `shutdown`.

A duplicate command ID/payload returns the same acceptance; changed payload
or stale worker generation is rejected. No command calls an optimizer on RTX.
Collection delegates to the existing vector owner, v3 gateways and complete
round driver. Evaluation delegates to the existing B5 reset/accounting and
paired-panel code. Only H200 assembles chunks and updates student weights.

The native entry is `rtx_actor/resident_isaac_trainer.py`. The CPU-only control
boundary is `dagger/rtx/code/src/dagger_rtx/resident_service.py`. Launch with
`rtx_actor/run_resident_isaac.py`; its default prints a plan and starts nothing.
Use `resident_isaac_client.py` for individual commands and `resident_pool.py`
for the two-worker barrier. `resident_campaign_site.py` is the opt-in concrete
campaign driver; ordinary `beta_site.py` remains the cold-launch default.

## Admission and resources

Pin an identical frozen source release, both GPU UUIDs, image/assets, tokens,
worker generation IDs and a native qualification receipt. A pool cannot run
a campaign until B6 -> B5 -> B6 switching and paired reset identity have actual
native evidence. Unit tests and forged test-only receipts do not admit it.

Normal actor caps remain 21 GiB each with no swap and six GiB host headroom.
One evaluation owner may increase to 28 GiB at an all-worker idle barrier;
the other remains parked at 21 GiB. Admission credits current owned usage,
not unknown jobs. Restore B6 before lowering the cap. OOM/owned swap, missing
attribution or lost source/GPU/worker identity fails closed. Retain the H200
allocation during RTX-only evaluation unless the user explicitly authorizes
release of that resource. Never manufacture GPU activity to evade reclamation.

Use user-systemd `Restart=no`, never Docker `--restart=always`. Once a native
owner is lost, persisted receipts cannot restore physics, cameras or recurrent
state. Preserve its last durable boundary and require a fresh runtime identity.
Routine phase changes must not call AppLauncher or close Kit. Shutdown/host
failure/version replacement can still require a real cold start.

## Evidence and logging

`resident-state.json` records command digests/acceptance/completion. The
append-only `resident-events.jsonl` and atomic `resident-progress.json` record
phase/profile switches and wall time; `/ready` additionally exposes bounded
native collection progress. Existing per-step native/H200 timing, learner MSE,
success numerator/denominator and W&B logging remain authoritative.

Certification requires actual native receipts: stable PID/Kit identity,
`native_initializations=1`, exact B6/B5 shapes, zero extra physical steps during
profile/reset probes, matched cold/paired B5 initial states, no OOM/swap, a live
collection/learner barrier and inference-only success evaluation. Record what
passed versus still pending; do not call this production-ready from CPU tests.
