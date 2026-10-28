# Real five-stage smoke: 2026-10-18

`smoke-10-17-e2e-03` completed fresh BC preparation, genuine offline rendering,
BC optimization/held-out MSE, corrected success testing, and all four DAgger
rounds. Each DAgger round completed two real full-batch optimizer updates,
baseline plus per-update held-out checks, best-checkpoint/AdamW handoff, and
the same two complete success-test cases. W&B online readback passed at all
five stages. This is execution evidence, not a policy-quality benchmark.

| Stage | Baseline → selected held-out MSE | Success test |
|---|---:|---:|
| BC | 0.993959 → 0.983614 | 0/2 |
| D1, teacher75% | 0.989526 → 0.982021 | 0/2 |
| D2, teacher50% | 1.116743 → 1.110241 | 0/2 |
| D3, teacher25% | 3.051277 → 3.037449 | 0/2 |
| D4, teacher0% | 3.128088 → 3.085653 | 0/2 |

BC used 16 training and 2 validation chunks. Online collection used two Leo
RTX actors with two environments each, four genuine H200 Image3F workers,
81 sequential steps per window, and four independently collected validation
chunks per DAgger round. Different rounds have different held-out trajectory
populations; their MSE values are not directly comparable. The tiny training
budget and two-case panel cannot estimate production quality.

## Evidence and recovery scope

- Runtime audit: `runtime/overnight-20261018/completed-smoke-audit.json`, SHA256
  `7a166d2966b73c5123b1339ed6e6c10b7c480585f4c5dc29dedd83a7f3871fe7`.
- Native/evaluation source snapshot:
  `e7170a5f56e2ba27a95f0692e284cceb843647a1c3be308844537fc6dd566021`.
- Completed BC receipts originated from snapshot
  `2ec4c97e16b5bcdec8718f1b04c24557d3da2bf65ca881f865aad5987d7cb7ad`.
  They were explicitly adopted before any test/DAgger command was accepted;
  data, seed, checkpoint and test identity were preserved. This was a recovered
  workflow, not an uninterrupted single-source launch.
- A later control-transport outage interrupted checkpoint delivery before the
  round-2 test was accepted. Original idle native journals proved nonexecution;
  byte transfer was retried without native or optimizer replay. Failed owned
  forwards were recovered before round 3. Both Kit owners stayed warm.
- All checkpoints/results were retained. No Cosmos allocation was released.
- Core source commit: `883cd4c`. Later capacity/transport hardening is separately
  tested; this smoke does **not** qualify a larger batch or a changed reset rule.

W&B project `nvidia/dagger`: BC `ao2q5k77`; D1
`beta-96afb7d2bb4aa51dcc2f945b`; D2 `beta-bbe954cdb9873cc3c42db5cc`;
D3 `beta-57653c158f3b26da6d0dc763`; D4 `beta-8915731c4808fd6e76c4a89f`.

The production plan reuses the admitted 2,560/320 BC populations and tests all
295 accepted cases after BC and every DAgger round. Larger RTX batches require
paired native resource evidence **and** a joint H200 full-batch learner check.
Use [the mode contract](../pipeline/deployment/RUN_MODES.md); do not relabel
smoke results or choose capacity from free VRAM alone.
