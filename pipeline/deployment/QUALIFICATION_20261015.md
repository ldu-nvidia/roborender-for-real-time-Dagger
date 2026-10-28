# Leo warm BC/DAgger smoke qualification

Completed October 14, 2026 (Pacific); evidence/run identifiers use October 15
UTC. Campaign: `dagger-leo-smoke-standing-20261015-v9`.

## Scope and result

The four-round warm infrastructure smoke completed with real Isaac observations,
teacher/Homie labels, H200 Image3F/student inference, behavior selection,
action return, native stepping, optimization/checkpoints and success evaluation.
Training and evaluation metrics were read back from online W&B.

- Two Leo RTX GPUs (physical 0 and 1), six environments each.
- Four H200s only for concrete online work, shape `custom.gpu.4xh200`, approved
  reservation `cosmos-solution-architect-xcwtffiq`.
- 81 steps per lane/window; 12 chunks and 972 rows per full-batch update.
- Three updates per round, teacher probabilities 0.75, 0.50, 0.25 and 0.
- Current-iteration data only; next round starts from the previous final policy.
- Ten-case success panel after each round, with paired BC/candidate arms,
  ray-traced three-view student input and no evaluation optimization.
- Totals: **12 updates, 144 learner chunks, 11,664 real committed transitions**.
- Each RTX native worker retained `native_initializations=1` across collection,
  evaluation and round changes. H200 allocations were released before RTX-only
  evaluation and verified absent at completion.

The previously completed BC smoke was reused with explicit checkpoint/result
provenance. It had exercised fresh teacher collection, offline 50-step rendering,
one disposable BC update and success evaluation. No BC update was repeated in
this warm campaign, and no smoke checkpoint was promoted to production.

| Round | Teacher probability | Training MSE for its three updates | Final success |
|---|---:|---|---:|
| 1 | 75% | 1.240757, 0.677372, 0.643241 | 1/10 |
| 2 | 50% | 1.585332, 0.777547, 0.570434 | 0/10 |
| 3 | 25% | 1.321783, 0.998938, 0.992897 | 1/10 |
| 4 | 0% | 3.051043, 5.981317, 7.037290 | 0/10 |

All 2,916 round-4 behavior actions came from the student. Its rising loss is
recorded as observed, not evidence of improvement. Ten cases and three updates
per round establish functional coverage, not policy convergence or a reliable
production success estimate. The 50-update/256-case production budget and
Isaac 6 versus Isaac 5 distribution equivalence are not certified by this run.

## Recovery and test evidence

An allocation-startup failure before round-3 activity was recovered at PREPARE.
A later interrupted checkpoint transfer left an unpublished partial file;
bounded, hash-verified chunk transfer completed it before evaluation began.
The pending-evaluation admission retained completed training and the original
resident owners. Neither recovery replayed native steps or optimizer updates.
This is evidence for those guarded recovery paths, not arbitrary GPU-loss or
mid-episode simulator restoration.

The final CPU regression suite passed **910 tests**:

```bash
runtime/test-venv-system/bin/python -m pytest -q \
  pipeline/tests pipeline/deployment pipeline/dagger/rtx/tests
```

The final source was identical across CPU/H200 storage and Leo (422 records)
before this documentation update. Source identities distinguish the frozen
native implementation, live management code and final logging metadata fix:

| Source | SHA-256 |
|---|---|
| Qualified native | `9c17c8e77fafaf1661328f7b6b92d3b873802b11c39db0d591aef157c95ec08a` |
| Live control release v11 | `f3773248bed9467cf8405503f74c66f34fbcbe4da357228fdc431c506f31576f` |
| Final control release v13 | `2c131d8ac77ac7fa1885fa35c3b1f7e442af87e1002338a9b8dc1c0b794ddcfb` |

The final control-only change corrected W&B collection/round-completion summary
flags; it did not change reported MSE, success outcomes or scientific behavior.
The commit containing this record adds documentation after that source release.

## Durable evidence and dashboards

Shared evidence root:
`/workspace/project/roborender/runtime/leo-smokes/dagger-leo-smoke-standing-20261015-v9`.

- `certification-20261015.json`: audited counts, results and provenance.
- `monitor-control-v11/completion.json`: all-round completion and allocation absence.
- `campaign/state.json`: completed campaign and final-checkpoint lineage.
- `completed-bc-reuse.json`: reused BC result and exact checkpoint provenance.

These runtime receipts, checkpoints and trajectories remain outside Git.

W&B: [round 1](https://wandb.ai/nvidia/dagger/runs/beta-bcd847e0b017d7627f41d588),
[round 2](https://wandb.ai/nvidia/dagger/runs/beta-09d19864ed4f48d22532c74e),
[round 3](https://wandb.ai/nvidia/dagger/runs/beta-9a6fa19ce98d7fe9899acfb8),
[round 4](https://wandb.ai/nvidia/dagger/runs/beta-39399286b7cb4b490ac99c95).
