# Bounded beta throughput sweep

This is a performance experiment, not production training or a success-rate
evaluation. It uses beta physical GPUs 1 and 2 (GPU0 excluded) and one reserved
`custom.gpu.4xh200` workload in `cosmos-solution-architect-xcwtffiq`.

## Measured candidate, 2026-10-09

Update, 2026-10-10: **B6 did not pass sustained qualification.** A new-seed
two-actor test hit the GPU2 container's 23 GiB system-RAM cap (Docker `oom`
event, exit 137) after 360 remotely committed transitions, before any complete
window or learner update. The safe stop preserved evidence and released the
four-H200 allocation. The single-window result below remains a historical
measurement, not a robustness certification. The B4 retry also failed at the
same cap on GPU1 after 248 committed transitions (Docker `oom`, exit 137),
with zero learner chunks/updates. Both native runtimes stopped safely. Merely
lowering the environment count is insufficient for these seeds. Revise the
host-RAM/process plan before another run; the default qualification plan is
blocked rather than silently increasing limits or permitting swap.

An opt-in allocator experiment is available through `--compact-host-heap`.
It sets `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_=131072` before Python
starts, and releases unused CPU heap after native setup and sealed-window
boundaries. It does not clear GPU tensors or change actions, observations,
reset sampling, or learner batches. Default launch behavior is unchanged.
The existing RAM caps, 6-GiB host safety floor, and swap-growth guard remain
mandatory. This is **not yet a qualified memory fix**: first run a fresh
two-actor capture test, then a sustained three-window live test. Resource
receipts include cgroup anonymous/file/shared memory and OOM counters.

[`candidate_beta_2rtx_4h200.json`](candidate_beta_2rtx_4h200.json) records the
best configuration that passed a complete bounded window: **6 environments
per RTX / 12 total**, shared four-worker routing, one full 12×81 learner batch,
and activation recomputation disabled (model checkpoints are still saved).
Warm collection throughput was 5.12 committed transitions/s. Collection and
publication took 193.55 s; the separately submitted learner job took 15.36 s.
One-time parallel Isaac initialization was 309 s and is reported separately.

Four environments per RTX also passed (4.11 transitions/s). Eight per RTX
passed a 10-step shared-pool trial (6.07 transitions/s), but an 81-step
dedicated-pool trial stopped at the host-swap guard. Twelve and 16 per RTX hit
the 23 GiB per-container memory limit. Beta has about 61 GiB host RAM; VRAM alone
does not establish actor capacity. Do not treat any of the rejected cases as
a sustained-throughput winner.

This is a reference profile, **not** an accepted production-launch config.
It does not relax the production v4/eight-gateway gate. Generated/raytraced
fractions differed across batches, so the measurements are workload-specific,
not proof of a globally optimal batch or improved policy success.
Detailed timing/resource receipts and the final report are in
`runtime/throughput-sweeps/evidence-20261009/PROGRESS.md` on shared storage.

## Experiment protocol

Fixed: corrected BC epoch-41 checkpoint, three views, Image3F five steps/CFG1,
TeaCache .3, 75% teacher behavior, native Raven augmentation/random initial
episode lengths, and an 81-frame learner window. No benchmark checkpoint is
promoted. Isaac 6 remains the separately qualified migration candidate.

Stages:

1. Two simultaneous native capture processes, initially eight environments
   each. Each process has a 22–24-GiB host-memory cap and no container swap. Stop
   only the scoped actors on an error, host RAM below 6 GiB, or owned-container
   swap/OOM. The current sampler attributes large host-wide swap changes using
   fresh cgroup counters; earlier sweeps used a 128-MiB global swap-growth guard.
2. Start at 8 environments per RTX with four genuine H200 workers, then test
   12/16 and only attempt 28 if smaller batches retain safe host-RAM headroom.
   Step down to 4/6 if a longer 8-environment trial triggers the resource guard.
   Both initialized actors must reach a shared cold-path start gate before any
   measured physical transition. Ten-step capacity trials do not create
   learner chunks. Exclude cold first-step timings and report warm latency,
   generated/raytraced fractions, aggregate committed transitions/s, and
   overlap between the two actors.
3. Compare shared four-gateway fanout against two dedicated gateways per RTX
   at the best safe environment count. The renderer engine remains batch-one;
   RPC lane batching is not a new TensorRT engine shape.
4. Complete an 81-step window for the strongest safe candidate, reconcile the
   two actors' complete chunks, and execute a real disposable learner update.
5. Independently benchmark learner batches of 8/16/28/56 complete chunks with
   full B*T BatchNorm populations. Compare activation checkpointing on/off,
   preserving FP32/model/loss semantics. These are optimization-semantics
   experiments: changing batch size is not a quality-equivalent speedup.

Rank by completed-transition throughput and complete collection-to-update
wall time, not GPU utilization or optimizer FPS alone. Report failed/OOM cases
instead of selecting them. A bounded sweep identifies the best **tested**
configuration, never a mathematically global optimum. Keep cold startup,
steady-state collection, checkpoint writing, and CPU data loading separate.

Implementation reuses native Raven callbacks, the v3 binary service boundary,
the existing Image3F pool, and the real learner. Production v4/eight-gateway
gates remain unchanged; this sweep does not yet establish a v4 optimum.

Entrypoints:

- `../rtx_actor/run_throughput_pair.py`: two GPU-identity-pinned actors, host
  resource guard, isolated outputs, timeout, and scoped cleanup.
- `../h200/create_migration_smoke_pod.py --gpus 4`: reserved allocation only;
  `--dry-run` never mutates the API.
- `../h200/run_migration_services.sh`: set
  `ROBORENDER_BENCHMARK_GPU_COUNT=4` for four live workers.
- `../h200/migration_smoke_control.py`: frozen run registration, audit, and
  bounded learner submission; use `--rtx-gpus 1 2 --gateway-count 4`.
- `../h200/benchmark_learner.py`: isolated real-data forward/backward trials.
- Learner parameter `activation_checkpointing=false` opts into retaining the
  full B*T forward activations instead of recomputing them. It is permitted
  only with `raven_exact_full_batch=true`; the default remains recomputation
  enabled. This changes memory/time tradeoffs, not the BatchNorm population.

Preserve all results on shared storage. Stop only owned actors/services and
release the H200 pod after completion or abandonment. No credentials, datasets,
checkpoints, or GPU runtime outputs belong in Git.
