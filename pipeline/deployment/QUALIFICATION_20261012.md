# Beta runtime and success-evaluation qualification

This records bounded native tests, not an unattended production promotion.
The source used on both GPU hosts was frozen at SHA-256
`bd118df69c8459772db43a0d456225cd456467ad5e621e4d0a5d3c274c9a3cd7`.
Later documentation/planner edits do not alter that executable evidence.

## Two RTX / four H200: PASS

Run: `rtx-beta-migration-runtime-pair-heap-1011-prod1`.
Config SHA-256:
`83c68116c4634428079889deac7fb910fad8bb6eeb511029443f32034c5e455e`.

- Beta GPUs 1 and 2, B6 per actor, one persistent Isaac lifecycle per actor.
- Four genuine Image3F/student workers on reserved `custom.gpu.4xh200`;
  `cosmos-solution-architect-xcwtffiq`, no burst fallback. GPU0 excluded.
- Three 81-step windows: 243 physical vector steps per actor, 2,916 total
  committed transitions, 36 learner chunks, three full 12x81 updates.
- Round 1 teacher beta 0.75, LR 0.0002, gradient clipping 1.0. No replay or
  microbatching in this bounded qualification.
- All 70 AdamW state counters advanced exactly once per update; global steps
  4100 -> 4101 -> 4102 -> 4103. Both actors accepted the same checkpoint chain.
- A deliberately lost commit response recovered without a repeated physical
  step. Both actors ended COMPLETE, with no uncertain simulator state.
- Per-iteration MSE: 0.9897832674, 0.4610602553, 0.4696037158. These are
  changing-rollout training metrics, not an accuracy or success comparison.
- [W&B](https://wandb.ai/nvidia/dagger/runs/beta-runtime-aab388ed504b4b54):
  actual final MSE/global step/update count read back from the cloud.
- Exact owned containers stopped; H200 allocation deleted after verifying
  client shutdown. Lifecycle receipt has no cleanup errors.

The earlier 23 GiB-cap OOM attempts were not erased or relabeled. The passing
test used **21 GiB per actor**, no container swap, `MALLOC_ARENA_MAX=2`,
`MALLOC_TRIM_THRESHOLD_=131072`, and opt-in CPU heap compaction at native setup
and sealed-window boundaries. No per-step CUDA cache flush or algorithm change
was introduced. Peak container memory was 18.40/18.22 GiB; minimum available
host RAM was 23.45 GiB. Container OOM/swap counters stayed zero; background
host swap growth was 2.59 MiB, below the 128 MiB guard.

One-time concurrent actor startup took 494.58 s. Warm 81-step collection took
about 194–199 s, and learner job completion took 16.36–21.89 s. The complete
bounded test, including startup/handoffs/shutdown, took 1240.72 s.

BC parent:
`053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718`.
Final disposable candidate:
`a33e4c8d39c209c0d16a654fb527bc5cd776db7631e4f09819a03514cd80cd3f`.

Shared evidence roots (ignored, not checked into Git):

- `runtime/pair-runtime-qualification-20261011/`
- `runtime/migrations/alpha-to-beta-20261008/h200/leo-roborender-beta-smoke-1011-pair/attempt1/`
- Beta: `/home/ldu/roborender/runtime/throughput-sweeps/rtx-beta-migration-runtime-pair-heap-1011-prod1/`

## Full paired success evaluation: execution/accounting PASS

Completed on 2026-10-12:
[`success-beta-panel-256-v4-seq-20261011-prod1`](https://wandb.ai/nvidia/dagger/runs/success-beta-panel-256-v4-seq-20261011-prod1),
sequentially on beta GPU1, B5, 40 GiB cap; no H200 or Image3F needed for this
ray-traced observation protocol. Neither policy was optimized during evaluation.

| Frozen policy | Successes / scored completed episodes | Goal-reached rate |
|---|---:|---:|
| Corrected BC epoch 41, global step 4100 | 0 / 256 | 0% |
| Three-update runtime candidate, global step 4103 | 11 / 256 | 4.296875% |

This is the **new beta B5 panel**, not Raven's historical single-view benchmark
or the old B64 panel. The candidate scored higher here, but absolute success is
still poor; this is not a production-quality endorsement or a claim of general
improvement across simulator distributions. It is not a completed DAgger round.

- Panel `beta-isaac6-b5-heldout-256-v4`, SHA-256
  `eec00069af8e4d5387ea86c0026abebf79024db7cb01a7f0764efd910dfa4ed9`.
- Frozen evaluation source SHA-256
  `6544b028e56940b0808e440172d0e56039588bf8c0f1d880422cfe00c928f0b5`.
- Result SHA-256
  `9addbd02a2e8d0ced39d7190884143534ab539706b4c437a71bf36e0a5be7ad7`.
- 52 B5 cohorts/checkpoint: 256 predetermined scored first episodes plus four
  unscored final-cohort lanes; all 260 native episodes per checkpoint completed.
- All 52 BC/candidate cohort starts matched byte-for-byte across 15 physical
  and state-observation descriptors. Both policies had the same physics and
  goal-accounting identities. This is not a claim of bitwise GPU trajectories.
- All 104 small cohort receipts checked: zero accounting disagreements,
  unchanged weights and zero optimizer steps. No full image/trajectory rescan.
- Independent CPU replay through Raven's original metric callbacks on cohorts
  0, 10 and 51 agreed: BC 0/5 in each; candidate 1/5, 0/5 and 0/5 respectively.
- Paired outcomes: both succeed 0, BC only 0, candidate only 11, neither 245.
- Final success counts/rates read back from online W&B. Both owned containers
  stopped; GPU1 returned to 0% utilization. GPU0 was not probed or used.
- Approximate native arm wall times including setup: BC 3387 s (56.5 min),
  candidate 3347 s (55.8 min). Peaks 33.53/32.77 GiB container memory; minimum
  host available 25.54/25.59 GiB; no OOM, container swap or host swap growth.

The numerator is the number of scored completed first episodes with a native
pre-reset goal flag observed at least once; the denominator is exactly 256.
The metric is not MSE. Evidence, `final-evidence-audit.json`, sampled native
callback replays and the handoff are under
`runtime/beta-production-qualification-20261011/` on the CPU/shared volume and
`/home/ldu/roborender/runtime/success-validations/` on beta.

### Failed attempts and reset corrections retained

V1/v2 probes found command/history/contact/target state surviving a cohort
reset. V3 restored cold state and refreshed reset poses before deriving the
lift target, but its full candidate run failed closed at cohort 10: appearance
event cooldowns depended on earlier policy episode lengths and changed RNG
consumption before motion sampling. Its BC arm finished 1/256 and candidate
finished 50/256, but that incomplete pair must not be used for comparison.

V4 additionally resets event cooldown clocks/flags and the common step counter
at explicit independent-cohort boundaries. Ordinary episode resets, actions
and physics remain unchanged. The full sequential run passed the strict state
equality gate at every cohort, including the previously failing boundary.

The concurrent v4 attempt stopped **before scoring**: host swap grew 224.7 MiB
at startup, above the unchanged 128 MiB guard, with zero container OOM/swap and
25.97 GiB minimum available host RAM. This was not a proven container OOM.
Concurrent evaluation remains unqualified; the passing run is sequential.
Earlier failed evidence is preserved, not overwritten or relabeled.

## Scope and remaining decisions

- Source CPU regression: 580 passed in 108.90 s, no skips; shell syntax and
  diff checks passed. The planner distinguishes GPU1-only success evaluation
  from two-RTX/four-H200 training; passing evaluation does not remove the
  separate production-controller guard.
- Three persistent windows qualify the tested runtime layout, not an unlimited
  run or the legacy v4/eight-gateway production controller.
- Production B12 replay/update scheduling must be explicit; do not silently
  inherit the legacy B28 replay counts.
- Isaac 5/6 distribution comparison remains a separate issue. The historical
  GPU tests above used the Mac/alpha management route. At 23:18 UTC the new
  Beta-owned service separately passed authenticated return traffic, binary
  round trips and a 25.57-second controlled reconnect without Alpha or the Mac.
  See [connection evidence and scope](connectivity/README.md). It does not
  retroactively qualify the new route for native GPU production; that bounded
  workload check remains outstanding.
