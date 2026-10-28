# H200 DAgger services

These services are the H200 half of distributed DAgger. RTX owns Isaac and the
privileged teacher. H200 owns Image3F, student inference, synchronized learner
rows, one-update learning, immutable checkpoints, and evaluation records.

## Roles

| Role | Port | Authority |
|---|---:|---|
| `coordinator` | 18610 | run/round/iteration state and actor barriers |
| `trajectory` | 18611 | validated immutable learner chunks |
| `registry` | 18612 | content-addressed student checkpoints |
| `learner` | 18613 | fixed worker jobs and result receipts |
| `evaluation` | 18614 | fixed-panel success results and promotion state |
| `render-student` | 18620–18627 | Image3F, student inference, beta selection, step ledger |

Control roles share transactional SQLite metadata beneath the ignored runtime
root. Large chunks and checkpoints remain immutable files. The eight data
gateways each own one Image3F process, one student replica, and one H200 GPU;
they use the shared trajectory authority only at safe publication boundaries.

All live SQLite access, including diagnostics and learner metadata reads, must
run on the bound owner host. Shared storage is not permission for cross-host
SQL. New ledgers use DELETE/FULL journals and fail closed on foreign host/boot
identity or unowned legacy databases. See [ledger durability](../../../docs/LEDGER_DURABILITY.md).
The current October18 MSE campaign explicitly pins four gateways and durable
v3 commits; the legacy v4 profile below does not override its frozen contract.

Every non-health request requires a bearer token. Endpoints bind to loopback
and are reached from RTX through SSH forwards. Never expose Image3F's legacy
pickle socket or any service port publicly.

## Active render/student protocol

Production uses `roborender.dagger.rtx.v4`, content type
`application/vnd.roborender.ndarray-batch.v4`, and `RDB4` framing. It keeps the
scientific step contract while removing avoidable hot-path work:

- current controls/state/teacher labels only;
- anchors registered once by content digest;
- 28 lanes sharded across eight independent gateway connections;
- zlib-level-1 binary blobs with canonical descriptors;
- previous commit combined with the next prepare;
- per-window in-memory assembly instead of per-row SQLite/blob persistence;
- no hot-path fsync; and
- final publication only after the complete 81-step ledger is reconciled.

The response remains per-lane and idempotent. H200 caches/replays the same
selection in memory for a repeated request identity and rejects a changed body. RTX does
not step until every lane validates. At step 80, the compact RTX audit manifest
is cross-checked against H200's rows before 28 learner chunks are published.
Any gateway PID or instance-generation replacement before finalization
invalidates the incomplete window and requires fail-closed abandonment; the
simulator action sequence is never replayed.

The v1 and v3 endpoints remain covered for conformance and recovery
compatibility. The historical v2 implementation is retained on RTX as tested
protocol provenance and as the transport foundation used by v3/v4, but this
H200 server does not expose v2 routes. None of these older paths is the
recommended production data path.

## Observation and action contract

The model view order is `head_color`, `hand_left`, `hand_right`.

- Generated episodes: anchor RGB, Raven `h5_v4` depth, and textured
  robot-on-black controls are 240×416 per view. Image3F produces policy RGB.
- Ray-traced episodes: RTX supplies Raven's float32 `[0,1]` policy tiles at
  108×192 per view and Image3F is bypassed.
- H200 stacks all policy views vertically in the same order.
- The student predicts 11 float32 action values; the supplied 15-D Homie tail
  forms the complete 26-D student action.
- The teacher's 26-D action is always stored as the supervised target.

Selection is teacher iff the persisted finite draw is strictly less than the
round beta. The selected 26-D behavior action is returned to RTX; reward,
`done`, and `timeout` arrive only after RTX performs the one physical step.

## Iteration and learner

An iteration requires 28 actor barriers, one for each 81-row lane window. A
window may cross episode boundaries, so episode ID/index and terminal flags are
per-row. The learner consumes exactly the chunks admitted for that iteration,
runs one optimizer update under Raven's one-minibatch contract, and writes a
content-addressed candidate checkpoint.

The production worker is launched by
`../run_raven_learner_worker_h200.sh`. It rejects a missing input checkpoint,
non-finite loss/gradients, wrong chunk lineage, or mismatched action shapes.
Between validation boundaries, a successful candidate becomes the immutable
input to the next fresh rollout iteration.

W&B project `dagger` records `train/iteration_mse`, gradient norm, checkpoint
identity, and phase timing per optimizer iteration. At the configured cadence,
the candidate is frozen while RTX performs the fixed-panel closed-loop success
evaluation. Validation logs `validation/goal_reached_rate` and Wilson bounds;
it performs no backpropagation and computes no validation MSE.

## Launch

Start one control role per process:

```bash
DAGGER_LEARNER_WORKER="$PWD/pipeline/dagger/run_raven_learner_worker_h200.sh" \
  pipeline/dagger/run_h200_service.sh learner

pipeline/dagger/run_h200_service.sh coordinator
pipeline/dagger/run_h200_service.sh trajectory
pipeline/dagger/run_h200_service.sh registry
pipeline/dagger/run_h200_service.sh evaluation
```

Start all eight Image3F/gateway pairs with:

```bash
pipeline/dagger/run_h200_render_pool.sh
```

Runtime paths, token location, devices, ports, and telemetry roots have
`DAGGER_*`/`IMAGE3F_*` environment overrides. Asset identities must be
validated before a live run; a routine service restart may skip repeated
multi-gigabyte hashing only after that deployment attestation exists.

## Tests

```bash
python -m pytest -q \
  test/pipeline/test_dagger_h200_services.py \
  test/pipeline/test_dagger_render_student_service.py \
  test/pipeline/test_dagger_round_saturation.py \
  test/pipeline/test_raven_learner_batching.py \
  test/pipeline/test_dagger_stream_v4.py
```

The tests cover authorization, exact schemas, hashes, idempotent retry,
route-scoped keys, selection correctness, no partial publish, v4 pipelining,
ephemeral cache behavior, final seal reconciliation, learner/checkpoint gates,
exact flattened-BxT SyncBatchNorm gradients/running statistics, index-safe
three-view max-pooling, and validation-controlled promotion. `--fake` is
test-only and must use an
isolated sink; a production identity must never advertise it.
