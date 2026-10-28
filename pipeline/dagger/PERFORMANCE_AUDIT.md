# DAgger performance audit

Active evidence: `smoke-dagger-lean-v4-gpu1-b28-20260910T185659Z`, B=28,
81 vector steps, 2,268 transitions, eight H200 workers.

## Current measured critical path

Excluding the cold first prepare (19.839 s), the warm vector cycle averaged
4.852 s:

| Exclusive phase | Per vector step | Per 81-step iteration | Share |
|---|---:|---:|---:|
| RTX synchronized capture + privileged teacher | 1.157 s | 93.7 s | 23.9% |
| framing/transfer + H200 Image3F/student/selection | 3.153 s | 255.4 s | 65.0% |
| RTX `env.step([28,26])` | .489 s | 39.6 s | 10.1% |
| bookkeeping | .052 s | 4.2 s | 1.0% |

Observed collection wall time was approximately seven minutes. The learner's
core update took 1.029 s and checkpoint writing .602 s. Final RTX compact seal
is a few seconds rather than tens of minutes.

The smoke completed 664/664 cross-host requests on the first attempt with no
error, retry, or OOM. Its single learner update reported MSE
`0.6420315584540845` and gradient norm `0.6199788451194763`.

## Removed overhead

The original scalar path spent roughly 49 minutes in the 81-step loop and 29
minutes repackaging full RTX chunks. The active v4 path removes that avoidable
work through:

- eight parallel data ports instead of 28 scalar requests on one stream;
- binary `RDB4` framing instead of base64 JSON arrays;
- zlib level 1 on the wire;
- one anchor registration per digest instead of retransmitting anchor RGB;
- omission of metric/raw archival depth arrays from the online payload;
- previous-step commit pipelined with current-step prepare;
- an H200 in-memory current-window cache with no per-lane hot-path fsync;
- H200-only learner-row assembly, with no RGB round trip back to RTX; and
- a compact RTX identity/outcome seal instead of loading, stacking, and
  recompressing full image/control rows.

These optimizations retain the same current observations, teacher labels,
student action, beta choice, behavior action, environment transition, and
learner tensors.

## Work that remains essential

The following cannot be removed without changing DAgger or weakening
exactly-once behavior:

- current Isaac capture and privileged teacher/Homie evaluation;
- Image3F for generated-control episodes;
- student inference and per-decision beta selection;
- the all-28 prepare barrier before one vector environment step;
- the physical `env.step([28,26])`;
- durable identities sufficient to prevent a double step;
- per-row episode/reset/outcome lineage;
- final 81-row chunk reconciliation;
- one current-window learner update and immutable checkpoint; and
- closed-loop success validation at its configured boundary.

## Remaining optimization targets

The dominant warm cost is still the render/student prepare span. Safe future
work should profile inside each H200 worker and preserve output identity:

1. overlap RTX capture/encoding with work that does not depend on the next
   simulator state;
2. reduce Python copies between decoded controls, Image3F, and student input;
3. batch compatible student forwards after independent Image3F completion;
4. keep checkpoint activation out of the per-step path and prewarm all replicas;
5. investigate a faster lossless codec only if end-to-end bytes and CPU time
   improve on the real B=28 payload; and
6. keep W&B and rich diagnostics asynchronous and scalar-only.

Do not weaken the all-lane barrier, idempotency keys, action construction,
teacher-label alignment, or success metric for throughput.
